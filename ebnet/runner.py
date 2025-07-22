import torch
from typing import Optional, Tuple, List
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
