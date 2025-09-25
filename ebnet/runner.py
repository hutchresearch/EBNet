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

import torch
from typing import Tuple
from tqdm import tqdm

def to_device(*args, device=torch.device("cpu")):
    cast_tensors = []
    for tensor in args:
        cast_tensors.append(tensor.to(device))
    return tuple(cast_tensors)

class Runner:
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

    def run(self, desc: str) -> torch.Tensor:
        self.model.train(False)
        batch_predictions = []

        loader = tqdm(self.loader, desc=desc) if self.verbose else self.loader

        for batch in loader:
            prediction = self._run_batch(batch)
            batch_predictions.append(prediction.detach().cpu())

        pred, pred_log_var = torch.chunk(torch.vstack(batch_predictions).detach(), chunks=2, dim=1)
        pred_std = torch.sqrt(torch.exp(pred_log_var))
        return pred, pred_std

    def _run_batch(self, batch) -> Tuple[torch.Tensor]:
        flux, rv, meta, period = to_device(*batch, device=self.device)
        prediction = self.model(flux, rv, meta, period)
        return prediction
