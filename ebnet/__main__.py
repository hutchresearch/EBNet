import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import os
import skeletonkey as sk
from runner import Runner
from denorm import denormalize_labels, denormalize_std
from chop_model import franken_load

def print_results_as_csv(pred, std, targets):
    print(",".join([t + "_pred" for t in targets]), end=",")
    print(",".join([t + "_std" for t in targets]), end="\n")

    for i in range(len(pred)):
        print(",".join(map(str, pred[i].tolist())), end=",")
        print(",".join(map(str, std[i].tolist())), end="\n")

@sk.unlock("./config.yaml")
def main(args) -> None:

    if args.model == "ebnet":
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "model_files/model.pth")
        model = sk.instantiate(args.eb_model, model_path=model_path)
    elif args.model == "ebnet+":
        model = sk.instantiate(args.eb_model_plus)
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "model_files/eb_model_plus")
        model_file = franken_load(model_path, 2)
        model.load_state_dict(model_file, strict=False)
    
    lwa = np.array(args.lwa)
    lwa_sorted_indicies = np.argsort(lwa)
    lcflux = np.array(args.lcflux)
    lcflux = lcflux[lwa_sorted_indicies].tolist()

    dataset = sk.instantiate(args.dataset, lcflux=lcflux, colflux=args.colflux)

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        collate_fn=lambda batch: batch[0], 
        shuffle=False,
        num_workers=0,
    )

    pred, std = Runner(
        loader=loader,
        model=model,
        verbose=args.verbose,
    ).run("Evaluating Model")

    std = denormalize_std(std, pred)
    pred = denormalize_labels(pred)
    
    print_results_as_csv(pred, std, args.targets)

if __name__ == "__main__":
    main()