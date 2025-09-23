import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import os
import yaml
import tempfile
from astropy.table import Table
from scipy.stats import circstd
from ebnet.dataset import Dataset
from ebnet.runner import Runner
from ebnet.denorm import denormalize_labels, denormalize_std
from ebnet.chop_model import franken_load
from ebnet.model import EBModelPlus, LoadedModelWrapper

def open_yaml(path: str) -> dict:
    with open(os.path.expanduser(path), "r") as handle:
        return yaml.safe_load(handle)

import numpy as np
import scipy.stats

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

    ebnet_model = None
    ebnet_plus_model = None

    if model_type in ["ebnet", "mixed"]:
        ebnet_model_path = os.path.join(entry_point_path, "model_files/model.pth")
        ebnet_model = LoadedModelWrapper(model_path=ebnet_model_path)

    if model_type in ["ebnet+", "mixed"]:
        ebnet_plus_model_path = os.path.join(entry_point_path, "model_files/eb_model_plus")
        ebnet_plus_model = EBModelPlus(
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
        model_state_dict = franken_load(ebnet_plus_model_path, chunks=2)
        ebnet_plus_model.load_state_dict(model_state_dict, strict=False)

    ebnet_pred = ebnet_std = None
    ebnet_plus_pred = ebnet_plus_std = None

    if model_type == "ebnet":
        ebnet_pred, ebnet_std = Runner(
            loader=data_loader,
            model=ebnet_model,
            verbose=verbose,
        ).run("Evaluating EBNet")

    elif model_type == "ebnet+":
        ebnet_plus_pred, ebnet_plus_std = Runner(
            loader=data_loader,
            model=ebnet_plus_model,
            verbose=verbose,
        ).run("Evaluating EBNet+")

    elif model_type == "mixed":
        ebnet_pred, ebnet_std = Runner(
            loader=data_loader,
            model=ebnet_model,
            verbose=verbose,
        ).run("EBNet Pass")
        ebnet_plus_pred, ebnet_plus_std = Runner(
            loader=data_loader,
            model=ebnet_plus_model,
            verbose=verbose,
        ).run("EBNet+ Pass")

    # Assemble mixed pred
    if model_type == "mixed":
        combined_pred = []
        combined_std = []

        for i, target in enumerate(targets):
            source_model_type = target_to_model_type[target]
            if source_model_type == "ebnet":
                combined_pred.append(ebnet_pred[:, i])
                combined_std.append(ebnet_std[:, i])
            elif source_model_type == "ebnet+":
                combined_pred.append(ebnet_plus_pred[:, i])
                combined_std.append(ebnet_plus_std[:, i])
            else:
                raise ValueError(f"Unknown model type for target '{target}': {source_model_type}")

        pred = torch.stack(combined_pred, dim=1)
        std = torch.stack(combined_std, dim=1)
    else:
        if model_type == "ebnet":
            pred, std = ebnet_pred, ebnet_std
        else:
            pred, std = ebnet_plus_pred, ebnet_plus_std

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
