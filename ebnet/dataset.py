"""
Pytorch Dataset class used for loading data into memory, preprocessing the inputs, 
and normalizing the inputs. 

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

import glob
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from astropy.table import Table

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir_path: Union[str, List[str]],
        lcflux: List[str],
        colflux: List[str],
        verbose: bool = True,
    ) -> None:
        """
        Initializes the Dataset with paths to FITS files, flux column names,
        and metadata column names.

        Args:
            data_dir_path: str or list of str, Path to a FITS file, a directory
                containing FITS files, or a list of such paths.
            lcflux: list of str, Names of light-curve flux columns.
            colflux: list of str, Names of metadata flux columns.
            verbose: bool, Whether to print warnings when columns are missing.
        """
        super(Dataset, self).__init__()

        self.flux_paths = []

        if isinstance(data_dir_path, str):
            if data_dir_path.endswith(".fits") and os.path.isfile(data_dir_path):
                self.flux_paths = [data_dir_path]
            else:
                pattern = os.path.join(data_dir_path, "*.fits")
                self.flux_paths = glob.glob(pattern)

        elif isinstance(data_dir_path, list):
            for path in data_dir_path:
                if path.endswith(".fits") and os.path.isfile(path):
                    self.flux_paths.append(path)
                else:
                    pattern = os.path.join(path, "*.fits")
                    self.flux_paths.extend(glob.glob(pattern))

        self.lx = 512
        self.lcflux = lcflux
        self.colflux = colflux
        self.verbose = verbose

    def _load_flux_from_table(self, table: Table) -> np.ndarray:
        """
        Loads flux values from an Astropy Table.

        Args:
            table: Table, The input FITS table.

        Returns:
            np.ndarray, Flux array of shape (n, lx, num_flux, 1).
        """
        n = len(table)

        # Load flux data
        flux = np.zeros((n, self.lx, len(self.lcflux), 1))
        for i, band in enumerate(self.lcflux):
            if band in table.colnames:
                flux[:, :, i, 0] = table[band].data
            elif self.verbose:
                print(f"[Missing Column] '{band}' not found.")
        
        return flux
    
    def _load_rv_from_table(self, table: Table) -> np.ndarray:
        """
        Loads radial velocity values from an Astropy Table.

        Args:
            table: Table, The input FITS table.

        Returns:
            np.ndarray, RV array of shape (n, lx, 2, 1).
        """
        n = len(table)
        
        # Load RV data and scale by 100
        rv = np.zeros((n, self.lx, 2, 1))
        for i, rv_col in enumerate(["rv1", "rv2"]):
            if rv_col in table.colnames:
                rv[:, :, i, 0] = table[rv_col].data / 100 # Bad to norm in this function. But, whatever.
            elif self.verbose:
                print(f"[Missing Column] '{rv_col}' not found.")
        
        return rv

    def _load_parallax_from_table(self, table: Table) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads parallax and parallax error from an Astropy Table.

        Args:
            table: Table, The input FITS table.

        Returns:
            tuple of np.ndarray, Arrays for parallax and parallax error.
        """
        n = len(table)

        if "plx" in table.colnames:
            parallax = np.nan_to_num(table["plx"].data)
        elif "parallax" in table.colnames:
            parallax = np.nan_to_num(table["parallax"].data)
        else:
            parallax = np.zeros(n)
            if self.verbose:
                print(f"[Missing Column] 'parallax' not found.")

        if "e_plx" in table.colnames:
            parallax_error = np.nan_to_num(table["e_plx"].data)
        elif "parallax_error" in table.colnames:
            parallax_error = np.nan_to_num(table["parallax_error"].data)
        else:
            parallax_error = np.zeros(n)
            if self.verbose:
                print(f"[Missing Column] 'parallax_error' not found.")

        return parallax, parallax_error


    def _load_meta_from_table(self, table: Table) -> np.ndarray:
        """
        Loads metadata values from an Astropy Table.

        Args:
            table: Table, The input FITS table.

        Returns:
            np.ndarray, Metadata array of shape (n, num_meta, 1).
        """
        n = len(table)

        # Load metadata columns
        meta = np.zeros((n, len(self.colflux), 1))
        for i, meta_col in enumerate(self.colflux):
            if meta_col in table.colnames:
                meta[:, i, 0] = table[meta_col].data
            elif self.verbose:
                print(f"[Missing Column] '{meta_col}' not found.")
        
        return meta

    def _load_period_from_table(self, table: Table) -> np.ndarray:
        """
        Loads orbital period from an Astropy Table.

        Args:
            table: Table, The input FITS table.

        Returns:
            np.ndarray, Array containing orbital period values.

        Raises:
            KeyError: If the required 'period' column is missing.
        """

        # Period is a required column. 
        if "period" not in table.colnames:
            raise KeyError("[Missing Column] Required column 'period' not found.")
        
        return table["period"].data

    def _load_from_fits(
        self, 
        path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads and formats flux, RV, metadata, period, parallax, and parallax error
        from a FITS file into PyTorch tensors.

        Args:
            path: str, Path to the FITS file.

        Returns:
            tuple of torch.Tensor, Contains flux, RV, metadata, period, parallax,
            and parallax error tensors.
        """
        data = Table.read(path)

        period = self._load_period_from_table(data)

        flux = self._load_flux_from_table(data)
        rv = self._load_rv_from_table(data)
        parallax, parallax_error = self._load_parallax_from_table(data)
        meta = self._load_meta_from_table(data)

        flux_formatted = torch.from_numpy(flux.astype(np.float32)) # batch, flux_lengh, num_flux, 1
        flux_formatted  = self.normalize_flux(flux_formatted)

        rv_formatted = torch.from_numpy(rv.astype(np.float32)) # batch, rv_lengh, num_rvs, 1

        meta_formatted = torch.from_numpy(meta.astype(np.float32)) # batch, meta_len, 1
        meta_formatted = self.normalize_meta(meta_formatted)

        period_formatted = torch.from_numpy(period.astype(np.float32)) # batch, period
        period_formatted = self.normalize_period(period_formatted)

        parallax_formatted = torch.from_numpy(parallax.astype(np.float32)) # batch, parallax
        parallax_formatted = self.normalize_parallax(parallax_formatted)

        parallax_error_formatted = torch.from_numpy(parallax_error.astype(np.float32)) # batch, parallax_error

        return flux_formatted, rv_formatted, meta_formatted, period_formatted, parallax_formatted, parallax_error_formatted


    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves and formats the i-th sample from the dataset.

        Args:
            i: int, Index of the sample.

        Returns:
            tuple of torch.Tensor, Contains flux, RV, metadata, and a stacked
            tensor of period, parallax, and parallax error.
        """
        flux_path = self.flux_paths[i]
        flux_batch, rv_batch, meta_batch, period_batch, parallax_batch, parallax_error_batch = self._load_from_fits(flux_path)

        flux_batch = flux_batch.permute(0, 2, 1, 3).squeeze(-1)
        rv_batch = rv_batch.permute(0, 2, 1, 3).squeeze(-1)
        meta_batch = meta_batch.squeeze(-1)

        period_parallax_batch = torch.hstack((period_batch[:, None], parallax_batch[:, None], parallax_error_batch[:, None]))

        return flux_batch, rv_batch, meta_batch, period_parallax_batch

    def __len__(self) -> int:
        """
        Returns the number of FITS files in the dataset.

        Returns:
            int, Number of samples.
        """
        return len(self.flux_paths)
    
    def normalize_flux(self, flux: torch.Tensor) -> torch.Tensor:
        """
        Normalizes flux values by dividing by the median value.

        Args:
            flux: torch.Tensor, Flux tensor.

        Returns:
            torch.Tensor, Normalized flux tensor with NaNs replaced by zero.
        """
        normalized_flux  = flux / torch.nanmedian(flux, dim=1, keepdim=True).values
        normalized_flux[normalized_flux.isnan()] = 0
        return normalized_flux
    
    def normalize_period(self, period: torch.Tensor) -> torch.Tensor:
        """
        Normalizes orbital period values by applying log scaling.

        Args:
            period: torch.Tensor, Period tensor.

        Returns:
            torch.Tensor, Normalized period tensor.
        """
        period = torch.log10(period) - 1
        return period
    
    def normalize_meta(self, meta: torch.Tensor) -> torch.Tensor:
        """
        Normalizes metadata values by dividing by 20.

        Args:
            meta: torch.Tensor, Metadata tensor.

        Returns:
            torch.Tensor, Normalized metadata tensor.
        """
        meta = meta / 20
        return meta
    
    def normalize_parallax(self, parallax: torch.Tensor) -> torch.Tensor:
        """
        Normalizes parallax values by applying log scaling.

        Args:
            parallax: torch.Tensor, Parallax tensor.

        Returns:
            torch.Tensor, Normalized parallax tensor.
        """
        parallax = torch.log10(parallax + 5)
        return parallax