import torch
from torch import Tensor
from utils import generate_weight, weighted_corrcoef

def mse_loss(output_data: Tensor, target_data: Tensor) -> Tensor:
    """
    Calculates the Mean Squared Error (MSE) loss between output and target tensors.

    Args:
        output_data: The predicted output tensor from the model.
        target_data: The ground truth target tensor.

    Returns:
        The calculated MSE loss as a single-value tensor.
    """
    return torch.pow(output_data - target_data, 2).mean()

def ic_loss(output_data: Tensor, target_data: Tensor, device, method=None) -> Tensor:
    """
    Calculates the negative weighted Information Coefficient (IC) loss.

    IC measures the correlation between predictions and actual returns.
    This loss function aims to maximize the weighted IC.

    Args:
        output_data: The predicted output tensor from the model.
        target_data: The ground truth target tensor.
        device: The torch device (e.g., 'cuda' or 'cpu') to perform calculations on.
        method: The weighting method to use for calculating the correlation.
                Passed to `generate_weight`. Defaults to None (likely uniform weights).

    Returns:
        The negative weighted IC loss as a single-value tensor.
    """
    weight = generate_weight(output_data.size(0), method=method).to(device)
    return -weighted_corrcoef(output_data, target_data, weight)
