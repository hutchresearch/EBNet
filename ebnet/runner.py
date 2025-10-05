"""
Code for doing a single run through all the data in a dataloader. 

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
import os
from typing import Tuple, List

import torch
from tqdm import tqdm

def to_device(
    *args: torch.Tensor, 
    device: torch.device=torch.device("cpu")
) -> Tuple[torch.Tensor, ...]:
    """
    Moves one or more tensors to the specified device.

    Args:
        *args: torch.Tensor, One or more tensors to move.
        device: torch.device, The device to move the tensors to.
            Defaults to CPU.

    Returns:
        tuple of torch.Tensor, The tensors on the specified device.
    """
    cast_tensors = []
    for tensor in args:
        cast_tensors.append(tensor.to(device))
    return tuple(cast_tensors)

class Runner:
    """
    Utility for running a single pass of a dataset through a model.
    """
    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ) -> None:

        self.loader = loader
        self.model = model.to(device)
        self.device = device
        self.verbose = verbose

    def run(self, desc: str) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Runs inference on the entire dataset.

        Args:
            desc: str, Description string used for the progress bar
                if verbose output is enabled.

        Returns:
            tuple of torch.Tensor, Contains two tensors:
                - Predictions for each sample in the dataset.
                - Estimated one-sigma uncertainties derived from the
                  log-variance outputs of the model.
        """
        self.model.train(False)
        paths = []
        batch_predictions = []

        loader = tqdm(self.loader, desc=desc) if self.verbose else self.loader

        for batch in loader:
            path, prediction = self._run_batch(batch)
            paths.extend([os.path.basename(path)] * prediction.shape[0])
            batch_predictions.append(prediction.detach().cpu())

        pred, pred_log_var = torch.chunk(torch.vstack(batch_predictions).detach(), chunks=2, dim=1)
        pred_std = torch.sqrt(torch.exp(pred_log_var))
        return paths, pred, pred_std

    def _run_batch(self, batch: Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[str, torch.Tensor]:
        """
        Runs inference on a single batch of data.

        Args:
            batch: tuple of torch.Tensor, A batch containing model inputs
                (flux, radial velocity, metadata, and period).

        Returns:
            torch.Tensor, Model predictions for the batch.
        """
        path, flux, rv, meta, period = batch
        flux, rv, meta, period = to_device(flux, rv, meta, period, device=self.device)
        prediction = self.model(flux, rv, meta, period)
        return path, prediction
