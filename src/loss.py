

import torch
from torch import Tensor

from utils import generate_weight, weighted_corrcoef




def mse_loss(output_data: Tensor, target_data: Tensor):
    diff = output_data - target_data
    square_diff = torch.pow(diff, 2)

    return square_diff.mean()


def ic_loss(output_data: Tensor, target_data: Tensor, device, method=None):
    weight = generate_weight(output_data.size(0), method=method).to(device)
    ic = weighted_corrcoef(output_data, target_data, weight)
    return -ic
