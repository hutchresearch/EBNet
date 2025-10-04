"""
Pipeline for model evaluation. 

MIT License
Copyright (c) 2025 hutchresearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import tempfile
from enum import Enum
from typing import List, Tuple, Union

import numpy as np
import scipy.stats
import torch
import yaml
from astropy.table import Table

from ebnet.chop_model import franken_load
from ebnet.dataset import Dataset
from ebnet.denorm import denormalize_labels, denormalize_std
from ebnet.model import EBModelPlus, LoadedModelWrapper
from ebnet.runner import Runner

class ModelType(str, Enum):
    """
    Identifiers for available prediction backends.

    Values:
        MODEL1: str, TensorFlow-backed model ("tf_model").
        MODEL2: str, PyTorch-backed model ("pt_model").
        MIXED: str, Hybrid mode routing targets to different backends ("mixed").
    """
    MODEL1 = "tf_model"
    MODEL2 = "pt_model"
    MIXED = "mixed"

def open_yaml(path: str) -> dict:
    """
    Loads a YAML configuration file.

    Args:
        path: str, Path to the YAML file. User home marker '~' is supported.

    Returns:
        dict, Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the path does not exist.
        PermissionError: If the file cannot be read due to permissions.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(os.path.expanduser(path), "r") as handle:
        return yaml.safe_load(handle)

def compute_orbital_angles(
    pred: np.ndarray, 
    std: np.ndarray, 
    targets: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes argument of periastron (per0) in degrees and phase zero (phase0) in cycles
    with circular standard deviations derived from one-sigma perturbations.

    Args:
        pred: np.ndarray, Predicted normalized targets with shape (n, t).
        std: np.ndarray, Predicted one-sigma uncertainties with shape (n, t).
        targets: list of str, Target names including "eccsin", "ecccos", "T0sin", "T0cos".

    Returns:
        tuple of np.ndarray, (per0_pred_deg, per0_std_deg, phase0_pred_cycles, phase0_std_cycles).

    Notes:
        per0 is computed as arctan2(eccsin, ecccos) in degrees.
        phase0 is computed as arctan2(T0sin, T0cos) divided by 2π in cycles.
        Circular standard deviations are computed with scipy.stats.circstd on perturbed samples
        within [-180, 180] degrees for per0 and [-0.5, 0.5] cycles for phase0.
    """
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

def predict(
    data: Union[str, Table],
    model_type: str = "mixed",
    meta_type: bool = "magnitude",
    download_flux: bool = False,
    num_workers: int = 1,
    verbose: bool = False,
    seed: int = 0,
) -> Table:
    """
    Runs the EBNet inference pipeline and returns predictions as an Astropy Table.

    Args:
        data: str or Table, Path to a FITS file or directory of FITS files, or an in-memory Table.
        model_type: str, Backend selection. One of {"tf_model", "pt_model", "mixed"}.
        verbose: bool, Whether to print progress and file status messages.
        seed: int, Random seed applied to NumPy and PyTorch.

    Returns:
        Table, Output table with prediction means and one-sigma uncertainties per target,
        plus derived orbital angle columns.

    Notes:
        If an Astropy Table is provided, a temporary FITS is written for dataset loading.
        Configuration is read from "config.yaml" colocated with this module.
        In mixed mode, each target is routed to the backend specified by "target_to_model_type".
        Output columns include "<target>_pred" and "<target>_std" for each configured target,
        and the derived columns "per0_pred", "per0_std", "phase0_pred", "phase0_std".
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    entry_point_path = os.path.split(os.path.abspath(__file__))[0]
    config = open_yaml(os.path.join(entry_point_path, "config.yaml"))
    targets = config["targets"]

    lwa = np.array(config["lwa"])
    lcflux = np.array(config["lcflux"])[np.argsort(lwa)].tolist()
    colflux = config["colflux"]
    colwa = config["colwa"]
    zero_points=config["photometric_zero_points"]

    # Convert Table input to temp directory
    if isinstance(data, Table):
        temp_dir = tempfile.TemporaryDirectory()
        temp_fits_path = os.path.join(temp_dir.name, "temp_table.fits")
        data.write(temp_fits_path, overwrite=True)
        data = temp_dir.name

    dataset = Dataset(
        data_dir_path=data,
        lcflux=lcflux,
        colflux=colflux,
        colwa=colwa,
        zero_points=zero_points,
        meta_type=meta_type,
        download_flux=download_flux,
        num_workers=num_workers,
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

    # MODEL1 only
    if model_type == ModelType.MODEL1:
        model1_path = os.path.join(entry_point_path, "model_files/tf_model.pth")
        model1 = LoadedModelWrapper(model_path=model1_path)

        pred, std = Runner(
            loader=data_loader,
            model=model1,
            verbose=verbose,
        ).run(f"Evaluating {ModelType.MODEL1.value}")

    # MODEL2 only
    elif model_type == ModelType.MODEL2:
        model2_path = os.path.join(entry_point_path, "model_files/pt_model")
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

        pred, std = Runner(
            loader=data_loader,
            model=model2,
            verbose=verbose,
        ).run(f"Evaluating {ModelType.MODEL2.value}")

    # MIXED mode
    elif model_type == ModelType.MIXED:
        model1 = LoadedModelWrapper(
            model_path=os.path.join(entry_point_path, "model_files/tf_model.pth")
        )

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
        model2.load_state_dict(
            franken_load(os.path.join(entry_point_path, "model_files/pt_model"), chunks=2),
            strict=False,
        )

        model1_pred, model1_std = Runner(
            loader=data_loader,
            model=model1,
            verbose=verbose,
        ).run(f"{ModelType.MODEL1.value} Pass")

        model2_pred, model2_std = Runner(
            loader=data_loader,
            model=model2,
            verbose=verbose,
        ).run(f"{ModelType.MODEL2.value} Pass")

        combined_pred: list[torch.Tensor] = []
        combined_std: list[torch.Tensor] = []

        for i, target in enumerate(targets):
            source_model_type = target_to_model_type[target]
            if source_model_type == ModelType.MODEL1:
                combined_pred.append(model1_pred[:, i])
                combined_std.append(model1_std[:, i])
            elif source_model_type == ModelType.MODEL2:
                combined_pred.append(model2_pred[:, i])
                combined_std.append(model2_std[:, i])
            else:
                raise ValueError(
                    f"Unknown model type for target '{target}': {source_model_type}"
                )

        pred = torch.stack(combined_pred, dim=1)
        std = torch.stack(combined_std, dim=1)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # denormalize predictions
    std = denormalize_std(std, pred)
    pred = denormalize_labels(pred)

    # Build output table
    result_table = Table()
    for i, target in enumerate(targets):
        result_table[f"{target}_pred"] = pred[:, i].numpy()
        result_table[f"{target}_std"] = std[:, i].numpy()

    per0_pred, per0_std, phase0_pred, phase0_std = compute_orbital_angles(
        pred.numpy(), std.numpy(), targets
    )

    result_table["per0_pred"] = per0_pred
    result_table["per0_std"] = per0_std
    result_table["phase0_pred"] = phase0_pred
    result_table["phase0_std"] = phase0_std

    # Drop uneeded columns
    for col in ["T0sin_pred", "T0sin_std", "T0cos_pred", "T0cos_std"]:
        if col in result_table.colnames:
            result_table.remove_column(col)

    return result_table