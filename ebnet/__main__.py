import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import os
import io
import tempfile
import skeletonkey as sk
from collections import OrderedDict
from dataset import prebatched_collate
from runner import Runner

def franken_load(load_path: str, chunks: int) -> OrderedDict:
    """
    Loads a PyTorch model from multiple binary files that were previously split.

    Args:
        load_path: str, The directory where the model chunks are located.
        chunks: int, The number of model chunks to load.

    Returns:
        A ordered dictionary containing the PyTorch model state.
    """

    def load_member(load_path: str, file_out: io.BufferedReader, file_i: int) -> None:
        """
        Reads a single chunk of the model from disk and writes it to a buffer.

        Args:
            load_path: str, The directory where the model chunk files are located.
            file_out: io.BufferedReader, The buffer where the model chunks are written.
            file_i: int, The index of the model chunk to read.

        """
        load_name = os.path.join(load_path, f"model_chunk_{file_i}")
        with open(load_name, "rb") as file_in:
            file_out.write(file_in.read())

    with tempfile.TemporaryDirectory() as tempdir:
        # Create a temporary file to write the model chunks.
        model_path = os.path.join(tempdir, "model.pt")
        with open(model_path, "wb") as file_out:
            # Load each model chunk and write it to the buffer.
            for i in range(chunks):
                load_member(load_path, file_out, i)
        
        # Load the PyTorch model from the buffer.
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    return state_dict


def print_results_as_csv(pred, std, targets):
    print(",".join([t + "_pred" for t in targets]), end=",")
    print(",".join([t + "_std" for t in targets]), end="\n")

    for i in range(len(pred)):
        print(",".join(map(str, pred[i].tolist())), end=",")
        print(",".join(map(str, std[i].tolist())), end="\n")

@sk.unlock("./config.yaml")
def main(args) -> None:
    lwa = np.array(args.lwa)
    lwa_sorted_indicies = np.argsort(lwa)
    lcflux = np.array(args.lcflux)
    lcflux = lcflux[lwa_sorted_indicies].tolist()

    if args.model == "ebnet":
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "model_files/model.pth")
        model = sk.instantiate(args.eb_model, model_path=model_path)
    elif args.model == "ebnet+":
        model = sk.instantiate(args.eb_model_plus)
        model_file = franken_load(os.path.join(os.path.split(os.path.abspath(__file__))[0], "model_files/eb_model_plus"), 2)
        model.load_state_dict(model_file, strict=False)

    dataset = sk.instantiate(args.dataset, lcflux=lcflux, colflux=args.colflux)

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        collate_fn=prebatched_collate, 
        shuffle=False,
        num_workers=0,
    )

    pred, std = Runner(
        loader=loader,
        model=model,
        verbose=args.verbose,
    ).run("Evaluating Model")

    print_results_as_csv(pred, std, args.targets)

if __name__ == "__main__":
    main()