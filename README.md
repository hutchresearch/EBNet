# EBNet
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
python -m ebnet <data_path> --model_type <model_type> --output <output_path> --verbose
```

### Positional Arguments

* **`data_path`**: Path to an input file or a directory containing input data (e.g., FITS files).

### Optional Arguments

* **`-h, --help`**: Show help information and exit.
* **`-m, --model_type`**: Model type for prediction. Default is `mixed`. Options:
  * `tf_model`
  * `pt_model`
  * `mixed`
* **`-o, --output`**: Path to save results as a FITS table. If not provided, results are printed as CSV to standard output.
* **`-v, --verbose`**: Enable verbose output.

## Input File Requirements

An input FITS file must contain a `period` column.

It can include any number of the following **light curve columns**:

* Kepler\_mean
* TESS\_T
* Johnson\_U, Johnson\_B, Johnson\_V, Johnson\_R, Johnson\_I, Johnson\_J, Johnson\_H, Johnson\_K
* Cousins\_R, Cousins\_I
* SDSS\_u, SDSS\_g, SDSS\_r, SDSS\_i, SDSS\_z
* SDSS\_uprime, SDSS\_gprime, SDSS\_rprime, SDSS\_iprime, SDSS\_zprime
* Gaia\_BP, Gaia\_G, Gaia\_RP, Gaia\_RVS
* PanStarrs\_g, PanStarrs\_r, PanStarrs\_w, PanStarrs\_open, PanStarrs\_i, PanStarrs\_z, PanStarrs\_y
* Stromgren\_u, Stromgren\_v, Stromgren\_b, Stromgren\_y
* LSST\_u, LSST\_g, LSST\_r, LSST\_i, LSST\_z, LSST\_y3
* SWASP\_default
* ZTF\_g, ZTF\_r, ZTF\_i
* Tycho\_B, Tycho\_V
* Hipparcos\_Hp

It can also include any number of the following **metadata columns**:

* GALEX.FUV, GALEX.NUV
* Johnson.U, Johnson.B, Johnson.V, Johnson.R, Johnson.I
* Cousins.U, Cousins.B, Cousins.V, Cousins.R, Cousins.I
* SDSS.u, SDSS.g, SDSS.r, SDSS.i, SDSS.z
* PAN-STARRS.PS1.g, PAN-STARRS.PS1.r, PAN-STARRS.PS1.i, PAN-STARRS.PS1.z, PAN-STARRS.PS1.y
* GAIA.GAIA3.Gbp, GAIA.GAIA3.G, GAIA.GAIA3.Grp
* 2MASS.J, 2MASS.H, 2MASS.Ks
* WISE.W1, WISE.W2, WISE.W3

It can include any number of the following **radial velocity columns**:

* rv1
* rv2

It can include any number of the following **parallax columns**:

* parallax
* parallax\_error

## Example Commands

1. Predict stellar parameters for all stars in `sample.fits` and write results to `output.csv`:

   ```bash
   python -m ebnet sample.fits > output.csv
   ```

2. Predict stellar parameters for all stars in all FITS files in `./path/to/data/` and write results to `output.fits`:

   ```bash
   python -m ebnet ./path/to/data/ -o output.fits
   ```
Here’s an updated version of that section with more detail on inputs, options, and outputs:

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
result = ebnet.predict(data)

# Save results to a FITS file
result.write("output.fits", format="fits", overwrite=True)
```

### Inputs

The `predict` function accepts three types of inputs:

* An in-memory Astropy `Table`
* A path to a single FITS file
* A path to a directory containing one or more FITS files

### Options

* `model_type`: Choose the backend for predictions. Options are:
  * `"tf_model"` – TensorFlow-based model
  * `"pt_model"` – PyTorch-based model
  * `"mixed"` – Uses both models, selecting the backend per target for best predictions.
* `verbose`: If `True`, prints progress messages and missing column warnings.
* `seed`: Sets NumPy and PyTorch random seeds for reproducibility.

### Outputs

The function returns an Astropy `Table` with:
* `<target>_pred`: Predicted value for each target parameter
* `<target>_std`: One-sigma uncertainty for each prediction
* `per0_pred` / `per0_std`: Derived argument of periastron in degrees and its uncertainty
* `phase0_pred` / `phase0_std`: Derived orbital phase zero in cycles and its uncertainty