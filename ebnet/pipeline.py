import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import os
import yaml
from astropy.table import Table
from ebnet.dataset import Dataset
from ebnet.runner import Runner
from ebnet.denorm import denormalize_labels, denormalize_std
from ebnet.chop_model import franken_load
from ebnet.model import EBModelPlus, LoadedModelWrapper


def open_yaml(path: str) -> dict:
    with open(os.path.expanduser(path), "r") as handle:
        return yaml.safe_load(handle)

def predict(data, model_type="mixed", verbose=False):
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

    ebnet_predictions = ebnet_uncertainties = None
    ebnet_plus_predictions = ebnet_plus_uncertainties = None

    if model_type == "ebnet":
        ebnet_predictions, ebnet_uncertainties = Runner(
            loader=data_loader,
            model=ebnet_model,
            verbose=verbose,
        ).run("Evaluating EBNet")

    elif model_type == "ebnet+":
        ebnet_plus_predictions, ebnet_plus_uncertainties = Runner(
            loader=data_loader,
            model=ebnet_plus_model,
            verbose=verbose,
        ).run("Evaluating EBNet+")

    elif model_type == "mixed":
        ebnet_predictions, ebnet_uncertainties = Runner(
            loader=data_loader,
            model=ebnet_model,
            verbose=verbose,
        ).run("EBNet Pass")
        ebnet_plus_predictions, ebnet_plus_uncertainties = Runner(
            loader=data_loader,
            model=ebnet_plus_model,
            verbose=verbose,
        ).run("EBNet+ Pass")

    # Assemble mixed predictions
    if model_type == "mixed":
        combined_predictions = []
        combined_uncertainties = []

        for i, target in enumerate(targets):
            source_model_type = target_to_model_type[target]
            if source_model_type == "ebnet":
                combined_predictions.append(ebnet_predictions[:, i])
                combined_uncertainties.append(ebnet_uncertainties[:, i])
            elif source_model_type == "ebnet+":
                combined_predictions.append(ebnet_plus_predictions[:, i])
                combined_uncertainties.append(ebnet_plus_uncertainties[:, i])
            else:
                raise ValueError(f"Unknown model type for target '{target}': {source_model_type}")

        pred = torch.stack(combined_predictions, dim=1)
        std = torch.stack(combined_uncertainties, dim=1)
    else:
        if model_type == "ebnet":
            pred, std = ebnet_predictions, ebnet_uncertainties
        else:
            pred, std = ebnet_plus_predictions, ebnet_plus_uncertainties

    std = denormalize_std(std, pred)
    pred = denormalize_labels(pred)

    result_table = Table()
    for i, target in enumerate(targets):
        result_table[f"{target}_pred"] = pred[:, i].numpy()
        result_table[f"{target}_std"] = std[:, i].numpy()

    return result_table
