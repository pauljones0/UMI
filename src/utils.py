import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing import Union, Optional, Dict, Type, List

# Small epsilon to avoid division by zero
EPSILON = 1e-8

def weighted_corrcoef(
    y_pred: Union[Tensor, np.ndarray],
    y_true: Union[Tensor, np.ndarray],
    weight: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:
    """
    Calculates the weighted Pearson correlation coefficient.

    Supports both PyTorch Tensors and NumPy arrays. The weight vector
    should sum to 1.

    Args:
        y_pred: Predicted values tensor/array. Shape (n_samples,).
        y_true: True values tensor/array. Shape (n_samples,).
        weight: Weight vector for samples. Shape (n_samples,). Should sum to 1.

    Returns:
        The weighted correlation coefficient.
    """
    if not isinstance(y_pred, type(y_true)):
        raise TypeError(f"Input types mismatch: y_pred is {type(y_pred)}, y_true is {type(y_true)}")

    if isinstance(y_pred, torch.Tensor):
        # Ensure weight is on the same device and dtype
        weight = weight.to(y_pred.device, dtype=y_pred.dtype)

        # Calculate weighted means
        y_pred_mean = torch.matmul(weight, y_pred)
        y_true_mean = torch.matmul(weight, y_true)

        # Calculate weighted covariance
        cov = torch.matmul(weight, (y_pred - y_pred_mean) * (y_true - y_true_mean))

        # Calculate weighted standard deviations
        std_pred = torch.sqrt(torch.matmul(weight, (y_pred - y_pred_mean) ** 2))
        std_true = torch.sqrt(torch.matmul(weight, (y_true - y_true_mean) ** 2))

        # Calculate correlation coefficient
        corr = cov / (std_pred * std_true + EPSILON)
        return corr

    elif isinstance(y_pred, np.ndarray):
        # Ensure weight is a numpy array
        if isinstance(weight, torch.Tensor):
            weight = weight.detach().cpu().numpy()

        # Calculate weighted means
        y_pred_mean = np.dot(weight, y_pred)
        y_true_mean = np.dot(weight, y_true)

        # Calculate weighted covariance
        cov = np.dot(weight, (y_pred - y_pred_mean) * (y_true - y_true_mean))

        # Calculate weighted standard deviations
        std_pred = np.sqrt(np.dot(weight, (y_pred - y_pred_mean) ** 2))
        std_true = np.sqrt(np.dot(weight, (y_true - y_true_mean) ** 2))

        # Calculate correlation coefficient
        corr = cov / (std_pred * std_true + EPSILON)
        return corr
    else:
        raise TypeError(f"Unsupported input type: {type(y_pred)}")


def generate_weight(stock_num: int, method: Optional[str] = None) -> Tensor:
    """
    Generates a weight tensor for a given number of stocks based on a method.

    Args:
        stock_num: The total number of stocks.
        method: Weighting method. Options:
                None (default): Uniform weights.
                'exp_decay': Exponential decay based on deciles (0.9^decile).
                'linear_decay': Linear decay based on deciles ((10-decile)/10).

    Returns:
        A FloatTensor of shape (stock_num,) containing weights that sum to 1.
    """
    if method is None:
        # Uniform weights
        weights = torch.ones(stock_num)
    else:
        weights = torch.zeros(stock_num)
        one_decile = stock_num // 10
        indices = torch.arange(stock_num)
        deciles = indices // one_decile
        deciles[deciles > 9] = 9 # Cap decile index at 9

        if method == "exp_decay":
            decay_factors = 0.9 ** deciles.float()
            weights = decay_factors
        elif method == "linear_decay":
            decay_factors = (10.0 - deciles.float()) / 10.0
            weights = decay_factors
        else:
            raise ValueError(f"Unknown weighting method: {method}")

    # Normalize weights to sum to 1
    weights /= weights.sum()
    assert abs(weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
    assert len(weights) == stock_num, "Weight length mismatch"
    return weights.float()


# Precompute log2 values for efficiency in DCG calculations
_LOG2_TABLE = np.log2(np.arange(2, 5002)) # Max k=5000? Adjust if needed

def _dcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
    """Calculates Discounted Cumulative Gain (DCG) at rank k."""
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size:
        # Use precomputed log2 table, ensure size matches
        log2_ranks = _LOG2_TABLE[:relevance_scores.size]
        return np.sum(np.divide(np.power(2, relevance_scores) - 1, log2_ranks))
    return 0.0

def ndcg(
    true_relevance: Union[List[float], np.ndarray],
    predicted_relevance: Union[List[float], np.ndarray],
    k: int = -1
) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Args:
        true_relevance: Ground truth relevance scores (ideally sorted).
        predicted_relevance: Predicted relevance scores corresponding to the order
                             for which DCG should be calculated.
        k: Rank cutoff. If -1, uses the length of predicted_relevance.

    Returns:
        The NDCG score.
    """
    true_relevance = np.asarray(true_relevance)
    predicted_relevance = np.asarray(predicted_relevance)
    rank_k = len(predicted_relevance) if k == -1 else k

    # Calculate Ideal DCG (IDCG) using the sorted true relevance scores
    ideal_dcg = _dcg_at_k(np.sort(true_relevance)[::-1], rank_k)

    # Calculate DCG using the predicted order
    actual_dcg = _dcg_at_k(predicted_relevance, rank_k)

    # Calculate NDCG
    if ideal_dcg == 0:
        return 0.0  # Avoid division by zero if ideal DCG is zero
    else:
        return actual_dcg / ideal_dcg


def _idcg_perfect(n_relevant: int) -> float:
    """Calculates the Ideal DCG for a list of n perfectly relevant items (score=1)."""
    if n_relevant <= 0:
        return 0.0
    # Assumes relevance score of 1 for all top items (2^1 - 1 = 1)
    # Denominators are log2(rank+1), precomputed table starts at log2(2)
    denominators = _LOG2_TABLE[:n_relevant]
    return np.sum(1.0 / denominators)


def lambdaRank_update(
    args: object, # Typically argparse.Namespace or similar object
    x_features: Tensor, # Unused in this function?
    y_true: Tensor,
    scores: Tensor,
    device: torch.device
) -> tuple[float, Tensor]:
    """
    Performs a LambdaRank update step to calculate gradients (lambdas).

    Reference: "Learning to Rank using Gradient Descent" by Burges et al.

    Args:
        args: Configuration object containing parameters like `lambda_topk`.
        x_features: Input features (currently unused in the calculation shown).
        y_true: True relevance scores/labels. Shape (n_stocks, 1).
        scores: Predicted scores from the ranking model. Shape (n_stocks,).
        device: The torch device to perform calculations on.

    Returns:
        A tuple containing:
        - Negative NDCG@k value (loss to be minimized).
        - Lambda gradients for backpropagation. Shape (n_stocks,).
    """
    scores = scores.unsqueeze(1) # Shape (n_stocks, 1)
    n_stocks = scores.size(0)
    if n_stocks <= 1: # Cannot rank with 0 or 1 stock
        return 0.0, torch.zeros_like(scores.squeeze())

    # Sort scores to get ranks
    _, sorted_indices = scores.sort(dim=0, descending=True)

    # Determine relevant and irrelevant sets based on top-k
    # Ensure k is not larger than available items
    k = min(getattr(args, 'lambda_topk', n_stocks), n_stocks)
    n_relevant = min(k, n_stocks) # Number of items considered relevant (top k)
    # Ensure we don't index out of bounds if k == n_stocks
    n_irrelevant = n_stocks - n_relevant

    if n_relevant == 0 or n_irrelevant == 0: # Cannot compare if one set is empty
         # Calculate NDCG anyway for monitoring
         ndcg_value = ndcg(y_true.cpu().numpy().ravel(),
                           scores[sorted_indices].cpu().numpy().ravel(),
                           k=k)
         return -ndcg_value, torch.zeros_like(scores.squeeze()) # Return 0 gradients

    # Get ranks (1-based)
    stock_ranks = torch.zeros(n_stocks, 1, device=device)
    stock_ranks[sorted_indices] = torch.arange(1, n_stocks + 1, device=device).float().unsqueeze(1)

    # Separate relevant (top-k predicted) and irrelevant items
    relevant_indices = sorted_indices[:n_relevant]
    irrelevant_indices = sorted_indices[n_relevant:]

    # Calculate pairwise score differences (relevant_i vs irrelevant_j)
    score_i = scores[relevant_indices] # Shape (n_relevant, 1)
    score_j = scores[irrelevant_indices].t() # Shape (1, n_irrelevant)
    score_diffs = score_i - score_j # Shape (n_relevant, n_irrelevant)

    # Calculate lambda updates based on score differences and DCG changes
    exp_score_diffs = torch.exp(score_diffs)
    lambda_factor = 1.0 / (1.0 + exp_score_diffs) # Sigmoid factor

    # Calculate ideal DCG for normalization factor N
    # Using simplified IDCG assuming top-k are relevant (score=1)
    ideal_dcg_k = _idcg_perfect(n_relevant)
    if ideal_dcg_k == 0:
        N = 0.0
    else:
        N = 1.0 / ideal_dcg_k

    # Calculate change in DCG if items i and j were swapped
    rank_i = stock_ranks[relevant_indices] # Shape (n_relevant, 1)
    rank_j = stock_ranks[irrelevant_indices].t() # Shape (1, n_irrelevant)
    dcg_gain_i = 1.0 / torch.log2(1.0 + rank_i)
    dcg_gain_j = 1.0 / torch.log2(1.0 + rank_j)
    delta_dcg = torch.abs(dcg_gain_i - dcg_gain_j) # Shape (n_relevant, n_irrelevant)

    # Combine factors to get lambda updates for each pair
    pair_lambdas = lambda_factor * N * delta_dcg # Shape (n_relevant, n_irrelevant)

    # Aggregate lambdas for each item
    lambdas = torch.zeros_like(scores) # Shape (n_stocks, 1)

    # Sum updates for relevant items (rows in pair_lambdas)
    lambdas.scatter_add_(0, relevant_indices, pair_lambdas.sum(dim=1, keepdim=True))
    # Sum updates for irrelevant items (columns in pair_lambdas) - need transpose and negative sign
    lambdas.scatter_add_(0, irrelevant_indices, -pair_lambdas.sum(dim=0, keepdim=True).t())

    # Calculate NDCG value for monitoring/loss
    # Use true relevance scores for NDCG calculation
    ndcg_value = ndcg(y_true.cpu().numpy().ravel(),
                      y_true[sorted_indices.squeeze()].cpu().numpy().ravel(), # Use true relevance in predicted order
                      k=k)

    return -ndcg_value, lambdas.squeeze() # Return loss and gradients


def df_to_grouped_dict(
    df: pd.DataFrame,
    group_col: str = 'Date',
    sort_cols: List[str] = ['Date', 'StkCode'],
    key_type: Type = str
) -> Dict:
    """
    Groups a DataFrame by a specified column and returns a dictionary.

    Args:
        df: Input DataFrame.
        group_col: Column name to group by (e.g., 'Date').
        sort_cols: List of columns to sort by before grouping.
        key_type: The desired type for the dictionary keys (e.g., str, int).

    Returns:
        A dictionary where keys are unique values from `group_col` (cast to `key_type`)
        and values are the corresponding sub-DataFrames.
    """
    grouped_dict = {}
    df_sorted = df.sort_values(by=sort_cols, ascending=True)
    for group_key, group_df in df_sorted.groupby(group_col):
        try:
            typed_key = key_type(group_key)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert group key '{group_key}' to {key_type}. Using original type. Error: {e}")
            typed_key = group_key
        grouped_dict[typed_key] = group_df
    return grouped_dict


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    initial_learning_rate: float,
    schedule: Optional[Dict[int, float]] = None
):
    """
    Adjusts the learning rate based on the current epoch according to a schedule.

    Args:
        optimizer: The PyTorch optimizer.
        epoch: The current epoch number (0-based or 1-based, depends on usage).
        initial_learning_rate: The starting learning rate.
        schedule: An optional dictionary mapping epoch thresholds to learning rate
                  multipliers. If None, uses a default schedule.
                  Example: {5: 0.5, 15: 0.1, 75: 0.05} means LR = initial * 0.5
                  at epoch 5, initial * 0.1 at epoch 15, etc.
    """
    if schedule is None:
        # Default schedule
        schedule = {5: 0.5, 15: 0.1, 75: 0.05}

    lr_multiplier = 1.0
    # Find the applicable multiplier based on the schedule
    # Sort keys to ensure correct application
    sorted_epochs = sorted(schedule.keys())
    for threshold_epoch in sorted_epochs:
        if epoch >= threshold_epoch:
            lr_multiplier = schedule[threshold_epoch]
        else:
            # Since keys are sorted, no need to check further
            break

    new_lr = initial_learning_rate * lr_multiplier

    # Apply the new learning rate to all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
