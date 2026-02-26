# EBNet
[![DOI](https://zenodo.org/badge/1024347138.svg)](https://doi.org/10.5281/zenodo.18790154)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Credit: Logan Sizemore, Marina Kounkel*

**EBNet** is a pipeline for predicting stellar parameters for eclipsing binaries using light curves.

## Installation
Clone the repository and run:
```bash
pip install .
```
from within the `EBNet` directory.

## Usage
Run the pipeline with:

```bash
python -m ebnet <data_path>
```

### Positional Arguments
* **`data_path`**: Path to an input FITS file or a directory containing FITS files.

### Optional Arguments
* **`-h, --help`**: Show help information and exit.
* **`-m, --model_type`**: Model type for prediction. Default is `mixed`. Options:
  * `tf_model`
  * `pt_model`
  * `mixed`
* **`-mt, --meta_type`**: Metadata representation type. Default is `magnitude`. Options:
  * `magnitude` — raw magnitudes (catalog-style).
  * `flux` — log10(lambda * F_lambda) representation.
    When `--download_flux` is used, metadata are automatically converted from flux densities (Jy) to log-scaled fluxes.
* **`-o, --output`**: Path to save results as a FITS table. If not provided, results are printed as CSV to standard output.
* **`-d, --download_flux`**: Download SED flux metadata from VizieR and overwrite or fill metadata flux columns in the input table.
* **`--num_workers`**: Number of parallel download workers for SED flux retrieval (default: 1). Increasing this may speed up downloads but can trigger rate limits.
* **`--device`**: PyTorch device to run inference on. Default is `"cpu"`.
* **`-v, --verbose`**: Enable verbose output, including SED download status and missing column warnings.

## Input File Requirements
An input FITS file must contain a **`period`** column.

It may include any number of the following **light curve columns**:
* Kepler_mean
* TESS_T
* Johnson_U, Johnson_B, Johnson_V, Johnson_R, Johnson_I, Johnson_J, Johnson_H, Johnson_K
* Cousins_R, Cousins_I
* SDSS_u, SDSS_g, SDSS_r, SDSS_i, SDSS_z
* SDSS_uprime, SDSS_gprime, SDSS_rprime, SDSS_iprime, SDSS_zprime
* Gaia_BP, Gaia_G, Gaia_RP, Gaia_RVS
* PanStarrs_g, PanStarrs_r, PanStarrs_w, PanStarrs_open, PanStarrs_i, PanStarrs_z, PanStarrs_y
* Stromgren_u, Stromgren_v, Stromgren_b, Stromgren_y
* LSST_u, LSST_g, LSST_r, LSST_i, LSST_z, LSST_y3
* SWASP_default
* ZTF_g, ZTF_r, ZTF_i
* Tycho_B, Tycho_V
* Hipparcos_Hp

It may also include any number of the following **metadata columns**, either in magnitudes (in which case, set meta_type="magnitude") or converted to log10(λF_λ) format (in which case, set meta_type="flux" ):
* GALEX.FUV, GALEX.NUV
* Johnson.U, Johnson.B, Johnson.V, Johnson.R, Johnson.I
* Cousins.U, Cousins.B, Cousins.V, Cousins.R, Cousins.I
* SDSS.u, SDSS.g, SDSS.r, SDSS.i, SDSS.z
* PAN-STARRS.PS1.g, PAN-STARRS.PS1.r, PAN-STARRS.PS1.i, PAN-STARRS.PS1.z, PAN-STARRS.PS1.y
* GAIA.GAIA3.Gbp, GAIA.GAIA3.G, GAIA.GAIA3.Grp
* 2MASS.J, 2MASS.H, 2MASS.Ks
* WISE.W1, WISE.W2, WISE.W3

It may include any number of the following **radial velocity columns**:
* rv1
* rv2

It may include any number of the following **parallax columns**:
* parallax
* parallax_error

## Example Commands

1. Predict stellar parameters for all stars in `sample.fits` and write results to `output.csv`:
   ```bash
   python -m ebnet sample.fits > output.csv
   ```

2. Predict stellar parameters for all stars in all FITS files in a directory and write results to `output.fits`:
   ```bash
   python -m ebnet ./path/to/data/ -o output.fits
   ```

3. Run predictions using a GPU:
   ```bash
   python -m ebnet sample.fits --device cuda > output.csv
   ```

4. Download missing SED metadata from VizieR using four workers:
   ```bash
   python -m ebnet sample.fits --download_flux --num_workers 4 --verbose
   ```

---

## Programmatic Usage
EBNet can also be used directly from Python without calling the command line interface.

```python
import ebnet
from astropy.table import Table

# Load input data from a FITS file
path = "./sample.fits"
data = Table.read(path)

# Run predictions with the default "mixed" backend
result = ebnet.predict(
    data,
    model_type="mixed",
    meta_type="flux",
    device="cuda",
    download_flux=False,
    num_workers=1,
    verbose=True
)

# Save results to a FITS file
result.write("output.fits", format="fits", overwrite=True)
```

### Inputs
The `predict` function accepts three types of inputs:
* An in-memory Astropy `Table`
* A path to a single FITS file
* A path to a directory containing one or more FITS files

### Options
* `model_type`: Model backend for predictions. Options:
  * `"tf_model"` – TensorFlow-based model
  * `"pt_model"` – PyTorch-based model
  * `"mixed"` – Uses both models, selecting the backend per target
* `meta_type`: Metadata representation type. Options:
  * `"magnitude"` — raw magnitudes (catalog-style)
  * `"flux"` — log10(λ * F_λ) representation
  * Automatically set to `"flux_jy"` when `download_flux=True`
* `device`: PyTorch device for computation (e.g., `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`).
* `download_flux`: If `True`, queries VizieR for SED fluxes and replaces existing metadata flux columns.
* `num_workers`: Number of threads for concurrent SED metadata downloads.
* `verbose`: Prints progress messages and missing column warnings to standard error.
* `seed`: Sets NumPy and PyTorch random seeds for reproducibility.

### Outputs
The function returns an Astropy `Table` with:
* `<target>_pred`: Predicted value for each target parameter
* `<target>_std`: One-sigma uncertainty for each prediction
* `per0_pred` / `per0_std`: Derived argument of periastron in degrees and its uncertainty
* `phase0_pred` / `phase0_std`: Derived orbital phase of periastron and its uncertainty

## Cite
```bibtex
@software{logan_sizemore_2025_14728812,
  author       = {Logan Sizemore},
  title        = {hutchresearch/BOSSNet: BOSS Net v2.0.0},
  month        = jan,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {2.0.0},
  doi          = {10.5281/zenodo.14728812},
  url          = {https://doi.org/10.5281/zenodo.14728812},
}
```
