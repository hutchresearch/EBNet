import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import os
import yaml
import tempfile
from astropy.table import Table
import scipy.stats
from enum import Enum
from ebnet.dataset import Dataset
from ebnet.runner import Runner
from ebnet.denorm import denormalize_labels, denormalize_std
from ebnet.chop_model import franken_load
from ebnet.model import EBModelPlus, LoadedModelWrapper

class ModelType(str, Enum):
    MODEL1 = "tf_model"
    MODEL2 = "pt_model"
    MIXED = "mixed"

def open_yaml(path: str) -> dict:
    with open(os.path.expanduser(path), "r") as handle:
        return yaml.safe_load(handle)

def compute_orbital_angles(pred, std, targets):
    num_samples = pred.shape[0]

    eccsin_idx = targets.index("eccsin")
    ecccos_idx = targets.index("ecccos")
    t0sin_idx = targets.index("T0sin")
    t0cos_idx = targets.index("T0cos")

    eccsin_pred = pred[:, eccsin_idx]
    ecccos_pred = pred[:, ecccos_idx]
    eccsin_std = std[:, eccsin_idx]
    ecccos_std = std[:, ecccos_idx]

    t0sin_pred = pred[:, t0sin_idx]
    t0cos_pred = pred[:, t0cos_idx]
    t0sin_std = std[:, t0sin_idx]
    t0cos_std = std[:, t0cos_idx]

    random_noise_x = np.random.normal(size=num_samples)
    random_noise_y = np.random.normal(size=num_samples)

    # per0
    per0_pred = np.arctan2(eccsin_pred, ecccos_pred) * 180.0 / np.pi

    per0_std = np.zeros(num_samples)
    for i in range(num_samples):
        perturbed_angle_deg = np.arctan2(
            eccsin_pred[i] + random_noise_x * eccsin_std[i],
            ecccos_pred[i] + random_noise_y * ecccos_std[i]
        ) * 180.0 / np.pi
        per0_std[i] = scipy.stats.circstd(
            perturbed_angle_deg, low=-180, high=180, normalize=False
        )

    # phase0
    phase0_pred = np.arctan2(t0sin_pred, t0cos_pred) / (2.0 * np.pi)

    phase0_std = np.zeros(num_samples)
    for i in range(num_samples):
        perturbed_phase_cycles = np.arctan2(
            t0sin_pred[i] + random_noise_x * t0sin_std[i],
            t0cos_pred[i] + random_noise_y * t0cos_std[i]
        ) / (2.0 * np.pi)
        phase0_std[i] = scipy.stats.circstd(
            perturbed_phase_cycles, low=-0.5, high=0.5, normalize=False
        )

    return per0_pred, per0_std, phase0_pred, phase0_std


def predict(data, model_type="mixed", verbose=False, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    entry_point_path = os.path.split(os.path.abspath(__file__))[0]
    config = open_yaml(os.path.join(entry_point_path, "config.yaml"))
    targets = config["targets"]

    lwa = np.array(config["lwa"])
    lcflux = np.array(config["lcflux"])[np.argsort(lwa)].tolist()
    colflux = config["colflux"]

    if isinstance(data, Table):
        temp_dir = tempfile.TemporaryDirectory()
        temp_fits_path = os.path.join(temp_dir.name, "temp_table.fits")
        data.write(temp_fits_path, overwrite=True)
        data = temp_dir.name

    dataset = Dataset(
        lcflux=lcflux,
        colflux=colflux,
        data_dir_path=data,
        verbose=verbose,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda batch: batch[0],
        shuffle=False,
        num_workers=0,
    )

    target_to_model_type = config["target_to_model_type"]

    model1 = None
    model2 = None

    if model_type in [ModelType.MODEL1, ModelType.MIXED]:
        model1_path = os.path.join(entry_point_path, "model_files/model.pth")
        model1 = LoadedModelWrapper(model_path=model1_path)

    if model_type in [ModelType.MODEL2, ModelType.MIXED]:
        model2_path = os.path.join(entry_point_path, "model_files/eb_model_plus")
        model2 = EBModelPlus(
            flux_in_channels=50,
            rv_in_channels=2,
            output_dim=42,
            kernel_size=2,
            flux_backbone_dim=32,
            num_flux_backbone_conv_layers=5,
            flux_conv_block_step=1,
            rv_backbone_dim=64,
            num_rv_backbone_conv_layers=5,
            rv_conv_block_step=1,
            meta_backbone_dim=128,
            num_meta_backbone_conv_layers=2,
            meta_conv_block_step=2,
            num_final_backbone_conv_layers=4,
            final_conv_block_step=3,
            num_linear_layers=2,
            linear_layer_dim=2048,
            drop_p=0.0,
        )
        model_state_dict = franken_load(model2_path, chunks=2)
        model2.load_state_dict(model_state_dict, strict=False)

    model1_pred = model1_std = None
    model2_pred = model2_std = None

    if model_type == ModelType.MODEL1:
        model1_pred, model1_std = Runner(
            loader=data_loader,
            model=model1,
            verbose=verbose,
        ).run(f"Evaluating {ModelType.MODEL1.value}")

    elif model_type == ModelType.MODEL2:
        model2_pred, model2_std = Runner(
            loader=data_loader,
            model=model2,
            verbose=verbose,
        ).run(f"Evaluating {ModelType.MODEL2.value}")

    elif model_type == ModelType.MIXED:
        model1_pred, model1_std = Runner(
            loader=data_loader,
            model=model1,
            verbose=verbose,
        ).run(f"{ModelType.MODEL1.value} Pass")
        model2_pred, model2_std = Runner(
            loader=data_loader,
            model=model2,
            verbose=verbose,
        ).run(f"{ModelType.MODEL1.value} Pass")

    # Assemble mixed pred
    if model_type == ModelType.MIXED:
        combined_pred = []
        combined_std = []

        for i, target in enumerate(targets):
            source_model_type = target_to_model_type[target]
            if source_model_type == ModelType.MODEL1:
                combined_pred.append(model1_pred[:, i])
                combined_std.append(model1_std[:, i])
            elif source_model_type == ModelType.MODEL2:
                combined_pred.append(model2_pred[:, i])
                combined_std.append(model2_std[:, i])
            else:
                raise ValueError(f"Unknown model type for target '{target}': {source_model_type}")

        pred = torch.stack(combined_pred, dim=1)
        std = torch.stack(combined_std, dim=1)
    else:
        if model_type == ModelType.MODEL1:
            pred, std = model1_pred, model1_std
        else:
            pred, std = model2_pred, model2_std

    std = denormalize_std(std, pred).numpy()
    pred = denormalize_labels(pred).numpy()

    result_table = Table()
    for i, target in enumerate(targets):
        result_table[f"{target}_pred"] = pred[:, i]
        result_table[f"{target}_std"] = std[:, i]
    
    per0_pred, per0_std, phase0_pred, phase0_std = compute_orbital_angles(pred, std, targets)
    
    result_table["per0_pred"] = per0_pred
    result_table["per0_std"] = per0_std
    result_table["phase0_pred"] = phase0_pred
    result_table["phase0_std"] = phase0_std

    for col in ["T0sin_pred", "T0sin_std", "T0cos_pred", "T0cos_std"]:
        result_table.remove_column(col)
    
    return result_table
