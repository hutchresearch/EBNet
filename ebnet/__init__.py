import torch
import yaml
import numpy as np
from astropy.table import Table
from model import EBModelPlus, LoadedModelWrapper
from dataset import Dataset
from chop_model import franken_load
from runner import Runner
from denorm import denormalize_labels, denormalize_std

def predict(fits_path: str) -> Table:
    model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "./config.yaml")
    return None

    # --- Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lcflux = config["lcflux"]
    colflux = config["colflux"]
    lwa = np.array(config["lwa"])
    targets = config["targets"]
    model_type = config["model"]

    # --- Sort lcflux by lwa (important for input channel ordering)
    sorted_indices = np.argsort(lwa)
    lcflux = np.array(lcflux)[sorted_indices].tolist()

    # --- Load model
    if model_type == "ebnet":
        model = torch.load("model_files/model.pth", map_location="cpu")
    elif model_type == "ebnet+":
        model = EBModelPlus(**config["eb_model_plus"])
        weights = franken_load("model_files/eb_model_plus", 2)
        model.load_state_dict(weights, strict=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.eval()

    # --- Load FITS as one-entry dataset
    tab = Table.read(fits_path)
    dataset = Dataset(table=tab, lcflux=lcflux, colflux=colflux)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda batch: batch[0],
        shuffle=False,
        num_workers=0,
    )

    # --- Run model
    with torch.no_grad():
        pred, std = Runner(loader=loader, model=model, verbose=False).run("Evaluating 1 FITS")

    pred = denormalize_labels(pred)
    std = denormalize_std(std, pred)

    # --- Append predictions to table
    for i, name in enumerate(targets):
        tab[f"{name}_pred"] = pred[:, i]
        tab[f"{name}_std"] = std[:, i]

    return tab
