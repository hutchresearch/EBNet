import torch

def denormalize_labels(
    labels: torch.Tensor,
) -> torch.Tensor:

    denorm_labels = torch.zeros_like(labels)
    denorm_labels[:, 0] = 10 ** ((labels[:, 0] * 0.4) + 3.85)
    denorm_labels[:, 1] = 10 ** ((labels[:, 1] * 0.4) + 3.85)
    denorm_labels[:, 2] = 10 ** (labels[:, 2])
    denorm_labels[:, 3] = 10 ** (labels[:, 3])
    denorm_labels[:, 4] = labels[:, 4]
    denorm_labels[:, 5] = labels[:, 5]
    denorm_labels[:, 6] = 10 ** (labels[:, 6] * 1.5)
    denorm_labels[:, 7] = 10 ** (labels[:, 7] * 1.5)
    denorm_labels[:, 8] = 10 ** (labels[:, 8] * 2)
    denorm_labels[:, 9] = 10 ** (labels[:, 9] * 2)
    denorm_labels[:, 10] = 10 ** ((labels[:, 10] * 2) + 1.5)
    denorm_labels[:, 11] = labels[:, 11]
    denorm_labels[:, 12] = 90.1 - (10 ** (labels[:, 12] + 0.6))
    denorm_labels[:, 13] = labels[:, 13] * 100
    denorm_labels[:, 14] = (10 ** (labels[:, 14] - 0.7)) - 0.02
    denorm_labels[:, 15] = denorm_labels[:, 14] * torch.sin(torch.zeros_like(labels[:, 15])) #??
    denorm_labels[:, 16] = denorm_labels[:, 14] * torch.cos(torch.zeros_like(labels[:, 16])) #??
    denorm_labels[:, 17] = 10 ** (labels[:, 17] + 1.5)
    denorm_labels[:, 18] = 10 ** (labels[:, 18] + 1.5)
    denorm_labels[:, 19] = labels[:, 19]
    denorm_labels[:, 20] = labels[:, 20]

    return denorm_labels

def denormalize_std(
    std_devs: torch.Tensor, 
    labels: torch.Tensor,      
) -> torch.Tensor:
    labels_upper = labels + std_devs
    denorm_labels = denormalize_labels(labels)
    denorm_labels_upper = denormalize_labels(labels_upper)
    denorm_std = denorm_labels_upper - denorm_labels
    return denorm_std