import os
import glob
import torch
import numpy as np
from typing import Union, List
from astropy.table import Table

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir_path: Union[str, List[str]],
        lcflux: List[str],
        colflux: List[str],
    ) -> None:
        super(Dataset, self).__init__()
        
        if isinstance(data_dir_path, str):
            pattern = os.path.join(data_dir_path, "*.fits")
            self.flux_paths = glob.glob(pattern)
        else:
            self.flux_paths = []
            for path in data_dir_path:
                pattern = os.path.join(path, "*.fits")
                self.flux_paths.extend(glob.glob(pattern))

        self.lx = 512
        self.lcflux = lcflux
        self.colflux = colflux

    def _load_from_fits(self, path):
        data = Table.read(path)
        n = len(data)

        # Load flux data
        flux = np.zeros((n, self.lx, len(self.lcflux), 1))
        for i, band in enumerate(self.lcflux):
            flux[:, :, i, 0] = data[band].data

        # Load RV data and scale by 100
        rv = np.zeros((n, self.lx, 2, 1))
        rv[:, :, 0, 0] = data["rv1"].data / 100
        rv[:, :, 1, 0] = data["rv2"].data / 100

        # Load period
        period = data["period"].data
        period = np.log10(period) - 1

        # Load parallax and error, handling key naming differences
        if "plx" in data.colnames:
            parallax = np.nan_to_num(data["plx"])
            parallax_error = np.nan_to_num(data["e_plx"])
        elif "parallax" in data.colnames:
            parallax = np.nan_to_num(data["parallax"])
            parallax_error = np.nan_to_num(data["parallax_error"])
        else:
            parallax = np.zeros(n)
            parallax_error = np.zeros(n)

        # Load metadata columns
        meta = np.zeros((n, len(self.colflux), 1))
        for i, col in enumerate(self.colflux):
            meta[:, i, 0] = data[col]
        
        flux = torch.from_numpy(flux.astype(np.float32)) # batch, flux_lengh, num_flux, 1
        rv = torch.from_numpy(rv.astype(np.float32)) # batch, rv_lengh, num_rvs, 1
        meta = torch.from_numpy(meta.astype(np.float32)) # batch, meta_len, 1
        period = torch.from_numpy(period.astype(np.float32)) # batch, period
        parallax = torch.from_numpy(parallax.astype(np.float32)) # batch, parallax
        parallax_error = torch.from_numpy(parallax_error.astype(np.float32)) # batch, parallax_error

        return flux, rv, meta, period, parallax, parallax_error


    def __getitem__(self, i: int):
        flux_path = self.flux_paths[i]
        flux_batch, rv_batch, meta_batch, period_batch, parallax_batch, parallax_error_batch = self._load_from_fits(flux_path)

        flux_batch = flux_batch.permute(0, 2, 1, 3).squeeze()
        rv_batch = rv_batch.permute(0, 2, 1, 3).squeeze()
        meta_batch = meta_batch.squeeze()

        period_parallax_batch = torch.hstack((period_batch[:, None], parallax_batch[:, None], parallax_error_batch[:, None]))

        return flux_batch, rv_batch, meta_batch, period_parallax_batch

    def __len__(self):
        return len(self.flux_paths)


def prebatched_collate(batch):
    flux_batch, rv_batch, meta_batch, period_batch = batch[0]
    return flux_batch, rv_batch, meta_batch, period_batch