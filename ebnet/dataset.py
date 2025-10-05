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
from enum import Enum
import glob
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from astropy.table import Table, Column

from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse

import requests
from io import BytesIO

class MetaType(str, Enum):
    MAGNITUDE = "magnitude"
    FLUX = "flux"
    FLUX_JY = "flux_jy"

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir_path: Union[str, List[str]],
        lcflux: List[str],
        colflux: List[str],
        colwa: List[str],
        zero_points: List[str],
        meta_type: MetaType,
        download_flux: bool,
        num_workers: int = 1,
        verbose: bool = False,
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
        self.colwa = colwa
        self.zero_points = zero_points
        self.meta_type = meta_type
        self.download_flux = download_flux
        self.num_workers = num_workers
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

        if self.download_flux:
            if self.verbose:
                print("Downloading SED metadata from vizier. Overwriting table values.")
            data = self.download_fill_metadata_flux(data)
            self.meta_type = MetaType.FLUX_JY

        period = self._load_period_from_table(data)

        flux = self._load_flux_from_table(data)
        rv = self._load_rv_from_table(data)
        parallax, parallax_error = self._load_parallax_from_table(data)
        meta = self._load_meta_from_table(data)

        if self.meta_type == MetaType.FLUX_JY:
            meta = self.jy_to_log_lambda_flux(meta)
        elif self.meta_type == MetaType.MAGNITUDE:
            meta = self.mags_to_log_lambda_flux(meta)

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

    def __getitem__(self, i: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return flux_path, flux_batch, rv_batch, meta_batch, period_parallax_batch

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
        period = torch.clamp(period, min=1e-6)
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
    
    def mags_to_log_lambda_flux(self, mag: np.ndarray) -> np.ndarray:
        """
        Converts magnitudes into log-scaled metadata values.

        The conversion applies the standard relation between magnitude and
        flux using bandpass zero points, then multiplies by the central
        wavelength and takes base-10 logarithm to produce values in
        log10(lambda * F_lambda).

        Args:
            mag: np.ndarray, Input magnitudes.

        Returns:
            np.ndarray, Converted log-scaled metadata.
        """
        zero_points = np.array(self.zero_points)[None, :, None]
        colwa = np.array(self.colwa)[None, :, None]
        return np.log10((10 ** (-mag / 2.5)) * zero_points * colwa)

    def jy_to_log_lambda_flux(self, flux_jy):
        """
        Convert fluxes in Jy to log10(lambda * F_lambda) in cgs units.

        Args:
            flux_jy : array_like, Flux values in Jy.
            lambda_angstrom : array_like, Effective wavelengths of the bands in Angstrom.

        Returns:
            np.ndarray
                log10(lambda * F_lambda) values.
        """
        c_ang_s = 2.99792458e18  # speed of light in Angstrom/s
        lambda_angstrom = np.array(self.colwa)[None, :, None]
        flux_cgs_hz = np.array(flux_jy) * 1e-23
        f_lambda = flux_cgs_hz * (c_ang_s / (np.array(lambda_angstrom) ** 2))
        log_lambda_flux  = np.log10(f_lambda * lambda_angstrom)
        return np.nan_to_num(log_lambda_flux, nan=0, posinf=0, neginf=0)

    def download_fill_metadata_flux(self, data: Table) -> Table:
        """
        Download flux metadata for each target in the input table and
        add the requested flux columns (self.colflux) to the table.
        This will overwrite any flux values that currently exist in the table. 

        Args:
            data: Table, Input table containing at least 'RAJ2000' and 'DEJ2000'.

        Returns:
            Table, Input table with additional columns for flux metadata.
        """    
        def fetch_sed(ra, dec, radius="1"):
            target = f"{ra} {dec}"
            target_enc = urllib.parse.quote(target)
            url = f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={target_enc}&-c.rs={radius}&-out.form=VOTable"
            try:
                if self.verbose:
                    print(f"Downloading {url}", end=" ")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()  # raise HTTPError if bad response
                t = Table.read(BytesIO(resp.content), format="votable")
                if self.verbose:
                    print("[Done]")
                return t
            except Exception as e:
                return e

        ra_batch = data["RAJ2000"]
        dec_batch = data["DEJ2000"]

        # prepare empty columns for requested fluxes
        for col in self.colflux:
            if col in data.colnames:
                data[col][:] = 0
            else:
                data.add_column(Column(np.full(len(data), 0), name=col))

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(fetch_sed, ra, dec): idx
                for idx, (ra, dec) in enumerate(zip(ra_batch, dec_batch))
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()

                if isinstance(result, Exception):
                    if self.verbose:
                        print(f"SED query failed for row {idx}: {result}")
                    continue
                
                sed_table = result
                for filter_name in self.colflux:
                    try:
                        filters = np.char.replace(sed_table["sed_filter"].astype(str), ":", ".")
                        filters = np.char.replace(filters, "/", ".")
                        mask = filters == filter_name

                        if np.any(mask):
                            rows = np.where(mask)[0]

                            if "sed_eflux" in sed_table.colnames:
                                # prefer rows with finite, positive error
                                errs = sed_table["sed_eflux"][rows].astype(float)
                                good = np.isfinite(errs) & (errs > 0)
                                if np.any(good):
                                    # choose row with smallest error
                                    chosen_idx = rows[np.argmin(errs[good])]
                                else:
                                    chosen_idx = rows[0]
                            else:
                                chosen_idx = rows[0]

                            flux_value = sed_table["sed_flux"][chosen_idx]
                            data[filter_name][idx] = flux_value
                    except Exception as e:
                        if self.verbose:
                            print(f"Could not update {filter_name} for row {idx}: {e}", flush=True)
        return data