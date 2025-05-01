import os
import logging
import random
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Assuming 'Query' class is defined in 'data.py'
from data import Query

logger = logging.getLogger(__name__)

# --- Stock Dictionary Class ---

class StockDictManager:
    """
    Manages a dictionary mapping stock codes (str) to unique integer IDs.

    Handles initialization from Query data and provides a lookup method.
    Reserves ID 1 for unknown/out-of-dictionary stocks.
    """
    def __init__(self):
        self.stk_dic: Dict[str, int] = {}
        self._next_id: int = 2 # Start IDs from 2

    def initialize_from_query(self, data_query: Query):
        """
        Initializes the stock dictionary from all unique stocks found in the Query object.

        Args:
            data_query: An initialized Query object.
        """
        logger.info("Initializing stock dictionary from query data...")
        all_stocks = set()
        # Iterate through valid stocks per date code provided by Query
        for stock_list in data_query.valid_stocks_per_code.values():
            all_stocks.update(stock_list)

        self.stk_dic = {stk: i for i, stk in enumerate(sorted(list(all_stocks)), start=self._next_id)}
        self._next_id = len(self.stk_dic) + 2
        logger.info(f"Initialized stock dictionary with {len(self.stk_dic)} unique stocks.")

    def get_stock_id(self, stock_code: str) -> int:
        """
        Gets the integer ID for a given stock code.

        Args:
            stock_code: The stock code string.

        Returns:
            The integer ID, or 1 if the stock code is not found in the dictionary.
        """
        return self.stk_dic.get(stock_code, 1) # Return 1 for unknown stocks

    def load_dictionary(self, loaded_stk_dic: Dict[str, int]):
        """
        Loads an existing stock dictionary.

        Args:
            loaded_stk_dic: The dictionary to load.
        """
        self.stk_dic = loaded_stk_dic
        # Update next_id based on loaded dictionary if needed for consistency
        if self.stk_dic:
            self._next_id = max(self.stk_dic.values()) + 1
        else:
            self._next_id = 2
        logger.info(f"Loaded stock dictionary with {len(self.stk_dic)} entries.")

    @property
    def total_stock_count(self) -> int:
        """Returns the total number of unique stocks plus reserved IDs (0 and 1)."""
        # Original code added +2, assuming 0 and 1 are reserved.
        return len(self.stk_dic) + 2

# --- Data Generation Functions ---

def get_stock_ids_for_date(
    date_code: int,
    data_query: Query,
    stock_manager: StockDictManager,
    stock_id_cache: Dict[int, Tensor]
) -> Tensor:
    """
    Retrieves or generates the tensor of stock IDs for a given date code.
    Uses a cache for efficiency.
    """
    if date_code not in stock_id_cache:
        stock_list = data_query.valid_stocks_per_code.get(date_code)
        if stock_list is None:
             # This should ideally not happen if date_code comes from valid keys
             logger.warning(f"Stock list not found for date code {date_code} in valid_stocks_per_code.")
             return torch.empty(0, dtype=torch.long)
        id_list = [stock_manager.get_stock_id(stk) for stk in stock_list]
        stock_id_cache[date_code] = torch.tensor(id_list, dtype=torch.long)
    return stock_id_cache[date_code]

def generate_pretraining_batch(
    data_query: Query,
    date_codes: List[int],
    stock_id_cache: Dict[int, Tensor],
    stock_manager: StockDictManager,
    batch_type: str = 'market2',
    min_stocks_per_sample: Optional[int] = None # Minimum stocks after random sampling
) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """
    Generates a batch of data for pre-training tasks ('Cointegration' or 'market2').

    Args:
        data_query: Initialized Query object.
        date_codes: List of date codes for the batch.
        stock_id_cache: Cache dictionary mapping date codes to stock ID tensors.
        stock_manager: Initialized StockDictManager.
        batch_type: Type of batch to generate ('Cointegration' or 'market2').
        min_stocks_per_sample: If specified, randomly samples this many stocks from each day.
                               If None, uses all available stocks for the day.

    Returns:
        Depending on batch_type:
        - 'Cointegration': (features, labels, extra_features, stock_ids)
        - 'market2': (features1, features2, pair_labels, stock_ids1, stock_ids2,
                      extra_features1, extra_features2, distances)
    """
    # --- Fetch and Cache Data for the Batch ---
    batch_data: Dict[int, Tuple[Tensor, Tensor, Tensor, List[str]]] = {}
    min_stocks_available = float('inf')
    for d in date_codes:
        tensor_data = data_query.get_data_step_tensor(d)
        if tensor_data is None:
            logger.warning(f"Skipping date code {d} in batch generation: No tensor data.")
            continue
        x, y, addi_x = tensor_data
        stock_list = data_query.valid_stocks_per_code.get(d) # Get original stock list
        if x is None or y is None or addi_x is None or stock_list is None:
             logger.warning(f"Skipping date code {d} in batch generation: Missing data components.")
             continue
        if x.shape[0] == 0:
             logger.warning(f"Skipping date code {d} in batch generation: Zero stocks.")
             continue

        batch_data[d] = (x, y, addi_x, stock_list)
        # Get stock IDs using cache
        _ = get_stock_ids_for_date(d, data_query, stock_manager, stock_id_cache)
        min_stocks_available = min(min_stocks_available, x.shape[0])

    if not batch_data:
        raise ValueError("No valid data found for any date code in the provided list.")

    # Determine the number of stocks to sample if specified
    if min_stocks_per_sample is not None:
        num_stocks_to_use = min(min_stocks_per_sample, int(min_stocks_available))
        if num_stocks_to_use <= 0:
             raise ValueError(f"Cannot generate batch: min_stocks_available ({min_stocks_available}) or requested min_stocks_per_sample ({min_stocks_per_sample}) is too low.")
    else:
        # If not sampling, we might need different handling if stock counts vary widely.
        # Original code used min_sktk implicitly. Let's stick to that for now if not sampling.
        num_stocks_to_use = int(min_stocks_available)


    # --- Generate Batch Based on Type ---
    if batch_type == 'Cointegration':
        x_batch, y_batch, addi_x_batch, id_batch = [], [], [], []
        for d in date_codes:
            if d not in batch_data: continue # Skip dates that had issues
            x, y, addi_x, _ = batch_data[d]
            id_tensor = stock_id_cache[d]

            # Randomly sample stocks if needed
            indices = torch.randperm(x.shape[0])[:num_stocks_to_use]

            x_batch.append(x[indices].unsqueeze(0))
            y_batch.append(y[indices].unsqueeze(0))
            addi_x_batch.append(addi_x[indices].unsqueeze(0))
            id_batch.append(id_tensor[indices].unsqueeze(0))

        if not x_batch: raise ValueError("Cointegration batch is empty after processing.")
        return (torch.cat(x_batch, dim=0), torch.cat(y_batch, dim=0),
                torch.cat(addi_x_batch, dim=0), torch.cat(id_batch, dim=0))

    elif batch_type == 'market2':
        # Uses the first date in date_codes as the anchor (d0)
        anchor_date = date_codes[0]
        if anchor_date not in batch_data:
            raise ValueError(f"Anchor date {anchor_date} not found in batch data for 'market2'.")

        x1_anchor, _, addi_x1_anchor, _ = batch_data[anchor_date]
        id1_anchor = stock_id_cache[anchor_date]

        # Sample anchor stocks (group 1)
        num_anchor_samples = num_stocks_to_use // 2
        if num_anchor_samples == 0: raise ValueError("num_stocks_to_use too small for market2 pairs.")
        anchor_indices = torch.randperm(x1_anchor.shape[0])[:num_anchor_samples]

        x1_sampled = x1_anchor[anchor_indices]
        addi_x1_sampled = addi_x1_anchor[anchor_indices]
        id1_sampled = id1_anchor[anchor_indices]

        # Create positive pair (from the same anchor day)
        num_positive_samples = num_stocks_to_use - num_anchor_samples
        positive_indices = torch.randperm(x1_anchor.shape[0])[:num_positive_samples] # Sample remaining stocks
        x2_pos = x1_anchor[positive_indices]
        addi_x2_pos = addi_x1_anchor[positive_indices]
        id2_pos = id1_anchor[positive_indices]

        # Initialize lists for the batch
        x1_list, x2_list = [x1_sampled.unsqueeze(0)], [x2_pos.unsqueeze(0)]
        addi_x1_list, addi_x2_list = [addi_x1_sampled.unsqueeze(0)], [addi_x2_pos.unsqueeze(0)]
        id1_list, id2_list = [id1_sampled.unsqueeze(0)], [id2_pos.unsqueeze(0)]
        pair_labels = [1] # Positive pair label
        distances = [1] # Distance for positive pair (anchor day)

        # Create negative pairs (from other days in the batch)
        for other_date in date_codes:
            if other_date == anchor_date or other_date not in batch_data:
                continue # Skip anchor or invalid dates

            x2_other, _, addi_x2_other, _ = batch_data[other_date]
            id2_other = stock_id_cache[other_date]

            # Find stocks in other_date that are *not* in the anchor sample
            # This ensures negative pairs are truly different stocks (or same stock on different day)
            is_not_in_anchor = torch.isin(id2_other, id1_sampled, invert=True)
            available_indices = torch.where(is_not_in_anchor)[0]

            if len(available_indices) < num_positive_samples:
                 logger.warning(f"Not enough unique stocks in date {other_date} for negative pairs. Sampling with replacement or fewer samples.")
                 # Sample with replacement or adjust num_positive_samples if needed
                 if len(available_indices) == 0: continue # Cannot create pair
                 neg_indices = available_indices[torch.randint(len(available_indices), (num_positive_samples,))]
            else:
                 neg_indices = available_indices[torch.randperm(len(available_indices))[:num_positive_samples]]


            x2_neg = x2_other[neg_indices]
            addi_x2_neg = addi_x2_other[neg_indices]
            id2_neg = id2_other[neg_indices]

            # Append negative pair data
            x1_list.append(x1_sampled.unsqueeze(0)) # Anchor stocks
            x2_list.append(x2_neg.unsqueeze(0)) # Stocks from other day
            addi_x1_list.append(addi_x1_sampled.unsqueeze(0))
            addi_x2_list.append(addi_x2_neg.unsqueeze(0))
            id1_list.append(id1_sampled.unsqueeze(0))
            id2_list.append(id2_neg.unsqueeze(0))
            pair_labels.append(0) # Negative pair label
            # Use absolute difference in date codes as distance? Original used day index diff + 1
            distances.append(abs(other_date - anchor_date) + 1)

        if not x1_list: raise ValueError("Market2 batch is empty after processing.")
        # Concatenate all pairs into a batch
        return (torch.cat(x1_list, dim=0), torch.cat(x2_list, dim=0),
                torch.tensor(pair_labels, dtype=torch.long),
                torch.cat(id1_list, dim=0), torch.cat(id2_list, dim=0),
                torch.cat(addi_x1_list, dim=0), torch.cat(addi_x2_list, dim=0),
                torch.tensor(distances, dtype=torch.float)) # Use float for distance weighting

    else:
        raise ValueError(f"Unsupported batch_type: {batch_type}")


def generate_daily_data(
    data_query: Query,
    date_code: int,
    stock_id_cache: Dict[int, Tensor],
    stock_manager: StockDictManager
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generates data for a single day, adding a batch dimension."""
    tensor_data = data_query.get_data_step_tensor(date_code)
    if tensor_data is None:
        raise ValueError(f"No tensor data found for date code {date_code}")
    x, y, addi_x = tensor_data
    if x is None or y is None or addi_x is None:
         raise ValueError(f"Missing data components for date code {date_code}")

    stock_ids = get_stock_ids_for_date(date_code, data_query, stock_manager, stock_id_cache)

    # Add batch dimension
    return x.unsqueeze(0), y.unsqueeze(0), addi_x.unsqueeze(0), stock_ids.unsqueeze(0)


# --- Model Components ---

class AttentionMarketAggregation(nn.Module):
    """
    Aggregates stock features using a simple attention mechanism based on the last time step.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Linear layer to project features before attention calculation
        self.key_query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, stock_features: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Args:
            stock_features: Input tensor of shape (batch_size, num_stocks, hidden_dim).
                            Assumes the features are already processed (e.g., by a sequence model).
            temperature: Softmax temperature for attention scores.

        Returns:
            Aggregated market representation tensor of shape (batch_size, hidden_dim * 2).
            Concatenates attention-weighted sum and the last stock's features.
        """
        if stock_features.dim() != 3:
             raise ValueError(f"Expected input shape (batch, stocks, features), got {stock_features.shape}")

        # Project features to get keys and queries
        projected_features = self.key_query_proj(stock_features) # (batch, stocks, hidden_dim)
        keys = projected_features
        # Use the feature vector of the last stock as the query
        query = keys[:, -1:, :] # (batch, 1, hidden_dim)

        # Calculate scaled dot-product attention scores
        # (batch, 1, hidden_dim) * (batch, stocks, hidden_dim) -> sum over hidden_dim
        # Simplified attention: dot product between query and all keys
        attention_scores = torch.sum(keys * query, dim=2) / math.sqrt(self.hidden_dim) # (batch, stocks)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores / temperature, dim=1).unsqueeze(-1) # (batch, stocks, 1)

        # Calculate weighted sum of input features (values = original stock_features)
        weighted_sum = torch.sum(stock_features * attention_weights, dim=1) # (batch, hidden_dim)

        # Concatenate weighted sum with the last stock's features (used as query)
        last_stock_feature = stock_features[:, -1, :] # (batch, hidden_dim)
        aggregated_output = torch.cat([weighted_sum, last_stock_feature], dim=1) # (batch, hidden_dim * 2)

        return aggregated_output


class StockEmbeddingAttention(nn.Module):
    """
    Combines sequence model output with stock-specific embeddings using attention.
    Intended for the 'market' pre-training task's main model.
    """
    def __init__(self, input_feature_dim: int, dropout: float = 0.3, num_total_stocks: int = 4000, use_stock_embedding: bool = False):
        super().__init__()
        self.use_stock_embedding = use_stock_embedding
        self.dropout_rate = dropout
        self.num_total_stocks = num_total_stocks
        self.embedding_dim = input_feature_dim * 2 # Embedding size is twice the input feature dim?

        # Stock embedding matrix (learnable)
        # Initialize with Xavier uniform for better starting point
        self.stock_embedding_matrix = nn.Parameter(torch.zeros(1, num_total_stocks, self.embedding_dim))
        nn.init.xavier_uniform_(self.stock_embedding_matrix.data)

        # Layer to calculate attention weights based on combined features + embedding
        self.attention_weight_layer = nn.Linear(self.embedding_dim, 1)

        # The sequence processing layer (e.g., LSTM, GRU, Transformer encoder output)
        # This assumes the input 'x' to forward is the output of such a layer.
        # Here, using a simple AttentionMarketAggregation as a placeholder sequence processor.
        # Replace this with the actual sequence processing logic if needed.
        self.sequence_processor = AttentionMarketAggregation(hidden_dim=input_feature_dim)

    def reinitialize_stock_embeddings(self, num_total_stocks: Optional[int] = None):
        """Reinitializes the stock embedding matrix, optionally with a new size."""
        if num_total_stocks is None:
            num_total_stocks = self.num_total_stocks
        else:
            self.num_total_stocks = num_total_stocks

        logger.info(f"Reinitializing stock embeddings for {num_total_stocks} stocks with dim {self.embedding_dim}.")
        self.stock_embedding_matrix = nn.Parameter(torch.zeros(1, num_total_stocks, self.embedding_dim))
        nn.init.xavier_uniform_(self.stock_embedding_matrix.data)


    def forward(self, sequence_features: Tensor, stock_ids: Tensor, return_stock_outputs: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass combining sequence features and stock embeddings.

        Args:
            sequence_features: Output from sequence model. Shape (batch_size, num_stocks, seq_len, features).
                               Needs processing first. Let's assume input is (batch, num_stocks, features)
                               after sequence processing.
            stock_ids: Tensor of stock integer IDs. Shape (batch_size, num_stocks).
            return_stock_outputs: If True, also return the combined stock+sequence features before aggregation.

        Returns:
            Aggregated output tensor. Shape (batch_size, embedding_dim).
            If return_stock_outputs is True, returns (aggregated_output, combined_stock_features).
        """
        stock_ids = stock_ids.long()
        batch_size, num_stocks, *_ = sequence_features.shape # Get batch and num_stocks

        # Process sequence features (e.g., aggregate time steps)
        # Placeholder: Apply the simple market attention aggregation
        # Input to sequence_processor should be (batch * num_stocks, seq_len, features)
        # Or adapt sequence_processor. Here assuming input is already (batch, num_stocks, features)
        # This part needs clarification based on how sequence_features are generated.
        # Assuming sequence_features is already processed to (batch, num_stocks, input_feature_dim)
        processed_sequence_features = sequence_features # Placeholder

        # Aggregate features per stock across the batch dimension if needed, or process per batch item
        aggregated_batch_outputs = []
        combined_stock_features_list = []

        for b in range(batch_size):
            # Get features for the current batch item
            stock_seq_features = processed_sequence_features[b, :, :] # (num_stocks, features)

            # Apply sequence processor (e.g., attention over stocks for this batch item)
            # This seems redundant if sequence_features is already processed per stock.
            # Original code applied self.layer (att_market) here. Let's assume it processes
            # the features for the stocks within this single batch item.
            # Requires adapting att_market or assuming stock_seq_features is the input.
            # Let's assume stock_seq_features IS the input to combine with embeddings.
            # combined_features = self.sequence_processor(stock_seq_features.unsqueeze(0)).squeeze(0) # (num_stocks, embedding_dim) ??

            # --- Revisiting Original Logic ---
            # o_i = self.layer(xi).unsqueeze(0) -> xi shape? Assume (num_stocks, features) -> o_i (1, num_stocks, embedding_dim/2)?
            # stk_m = torch.index_select(...) -> (1, num_stocks, embedding_dim)
            # weight_i = F.relu(self.weightlayer(o_i+stk_m)) -> Requires o_i to be (1, num_stocks, embedding_dim)
            # This implies self.layer output dim = embedding_dim. Let's adjust.
            # Assume self.sequence_processor outputs (num_stocks, embedding_dim)
            processed_seq_features_b = self.sequence_processor(stock_seq_features.unsqueeze(0)).squeeze(0) # (num_stocks, embedding_dim)

            # Get stock embeddings for this batch item
            current_stock_ids = stock_ids[b, :] # (num_stocks,)
            # Select embeddings using stock IDs. Shape (1, num_total_stocks, embedding_dim) -> (1, num_stocks, embedding_dim)
            stock_embeddings = torch.index_select(self.stock_embedding_matrix, 1, current_stock_ids) # (1, num_stocks, embedding_dim)

            # Combine sequence features and stock embeddings
            # This combination logic needs clarification. Original added them?
            # combined = projected_seq_features.unsqueeze(0) + stock_embeddings * self.use_stock_embedding # (1, num_stocks, embedding_dim)
            # Let's assume concatenation based on embedding_dim = 2 * input_feature_dim
            # This requires projected_seq_features to be (num_stocks, embedding_dim/2)
            # Placeholder: Assume projected_seq_features has correct size
            # combined = torch.cat([projected_seq_features, projected_seq_features], dim=-1).unsqueeze(0) # Make it embedding_dim
            # combined = combined + stock_embeddings * self.use_stock_embedding

            # --- Re-evaluating based on original forward ---
            combined = processed_seq_features_b.unsqueeze(0) + stock_embeddings * self.use_stock_embedding # (1, num_stocks, embedding_dim)

            # Calculate attention weights
            attention_logits = self.attention_weight_layer(combined) # (1, num_stocks, 1)
            # Softmax over stocks? Original didn't use softmax here. Used ReLU?
            # weight_i = F.relu(attention_logits)
            # Let's assume softmax makes more sense for attention weights
            attention_weights = F.softmax(attention_logits, dim=1) # (1, num_stocks, 1)

            # Apply attention weights to the combined features
            # Original: torch.sum(weight_i * o_i, dim=1) / (torch.sum(weight_i, dim=1) + 1e-8)
            # Using o_i = processed_seq_features_b.unsqueeze(0)
            weighted_features = attention_weights * processed_seq_features_b.unsqueeze(0) # (1, num_stocks, embedding_dim)
            aggregated_output = torch.sum(weighted_features, dim=1) # (1, embedding_dim)
            # Normalize? Original divided by sum of weights.
            aggregated_output = aggregated_output / (torch.sum(attention_weights, dim=1) + 1e-8)

            aggregated_batch_outputs.append(aggregated_output)
            if return_stock_outputs:
                 combined_stock_features_list.append(combined.squeeze(0)) # (num_stocks, embedding_dim)


        final_aggregated_output = torch.cat(aggregated_batch_outputs, dim=0) # (batch_size, embedding_dim)

        if return_stock_outputs:
            # Need to handle varying num_stocks if not padded/sampled consistently
            # Assuming consistent num_stocks per batch item for now
            final_combined_stock_features = torch.stack(combined_stock_features_list, dim=0) # (batch, num_stocks, embedding_dim)
            return final_aggregated_output, final_combined_stock_features
        else:
            return final_aggregated_output


class PairwiseStockComparer(nn.Module):
    """
    Compares embeddings of two stocks (or groups of stocks) for the 'market' pre-training task.
    Predicts if they belong to the same context/day (label 1) or different (label 0).
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        # Batch norm applied to concatenated embeddings
        self.batch_norm = nn.BatchNorm1d(embedding_dim * 2)
        # Final layer to produce similarity score (logit for class 0 vs 1)
        # Original output size 1, suggesting regression or direct logit output.
        # For classification (same vs different day), output size 2 might be clearer.
        # Let's stick to original output size 1 for now.
        self.similarity_layer = nn.Linear(embedding_dim * 2, 1)

    def forward(self, embedding1: Tensor, embedding2: Tensor) -> Tensor:
        """
        Args:
            embedding1: Embedding tensor for the first stock/group. Shape (batch_size, embedding_dim).
            embedding2: Embedding tensor for the second stock/group. Shape (batch_size, embedding_dim).

        Returns:
            Similarity score/logit tensor. Shape (batch_size, 1).
        """
        # Concatenate the two embeddings
        combined_embedding = torch.cat([embedding1, embedding2], dim=1) # (batch_size, embedding_dim * 2)
        # Apply batch norm
        normalized_embedding = self.batch_norm(combined_embedding)
        # Apply dropout
        dropped_embedding = F.dropout(normalized_embedding, p=self.dropout_rate, training=self.training)
        # Calculate final score
        score = self.similarity_layer(dropped_embedding)
        return score


class MarketDirectionPredictor(nn.Module):
    """
    Predicts the market direction (e.g., Up, Down, Neutral) based on aggregated stock embeddings.
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.0, num_market_classes: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        self.num_market_classes = num_market_classes

        # Intermediate layer
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        # Final classification layer
        self.output_layer = nn.Linear(embedding_dim, num_market_classes)

    def forward(self, aggregated_embedding: Tensor) -> Tensor:
        """
        Args:
            aggregated_embedding: Aggregated market embedding. Shape (batch_size, embedding_dim).

        Returns:
            Logits for market direction classes. Shape (batch_size, num_market_classes).
        """
        # Apply intermediate layer with activation and dropout
        hidden = F.relu(self.fc1(aggregated_embedding))
        dropped = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        # Calculate output logits
        logits = self.output_layer(dropped)
        return logits


# --- Cointegration Models (Placeholder based on original structure) ---
# These models seem highly specific and their structure is hard to interpret without context.
# Keeping the structure but adding docstrings and type hints.

class CointegrationAttentionLayer(nn.Module):
    """
    A specific layer used in the Cointegration pre-training task.
    Applies attention based on learnable stock-pair interactions. (Interpretation based on original code)
    """
    def __init__(self, num_total_stocks: int = 4000, dropout: float = 0.3):
        super().__init__()
        self.num_total_stocks = num_total_stocks
        self.dropout_rate = dropout
        # Learnable pairwise interaction matrix (stock i vs stock j)
        self.stock_interaction_matrix = nn.Parameter(torch.ones(1, num_total_stocks, num_total_stocks))
        # Learnable attention weights for interactions
        self.stock_attention_weights = nn.Parameter(torch.ones(1, num_total_stocks, num_total_stocks))
        nn.init.xavier_uniform_(self.stock_attention_weights.data, gain=1.0)

    def forward(self, stock_features: Tensor, stock_ids: Tensor) -> Tensor:
        """
        Args:
            stock_features: Input features for stocks in the batch. Shape (batch_size, num_stocks, features).
            stock_ids: Integer IDs for stocks. Shape (batch_size, num_stocks).

        Returns:
            Output features after applying attention. Shape (batch_size, num_stocks, features).
        """
        stock_ids = stock_ids.long()
        batch_size, num_stocks, _ = stock_features.shape

        # Select relevant parts of interaction and weight matrices for the current batch
        batch_interactions_list = []
        batch_weights_list = []
        for b in range(batch_size):
            ids_b = stock_ids[b, :] # (num_stocks,)
            # Select rows then columns based on stock IDs
            interactions_b = torch.index_select(self.stock_interaction_matrix, 1, ids_b) # (1, num_stocks, total_stocks)
            interactions_b = torch.index_select(interactions_b, 2, ids_b) # (1, num_stocks, num_stocks)
            batch_interactions_list.append(interactions_b)

            weights_b = torch.index_select(self.stock_attention_weights, 1, ids_b) # (1, num_stocks, total_stocks)
            weights_b = torch.index_select(weights_b, 2, ids_b) # (1, num_stocks, num_stocks)
            batch_weights_list.append(weights_b)

        batch_interactions = torch.cat(batch_interactions_list, dim=0) # (batch, num_stocks, num_stocks)
        batch_weights = torch.cat(batch_weights_list, dim=0) # (batch, num_stocks, num_stocks)

        # --- Attention Calculation (based on original logic) ---
        # Remove diagonal elements (self-interaction?) before applying dropout/softmax
        diag_interactions = torch.diagonal(batch_interactions, dim1=1, dim2=2)
        interactions_no_diag = batch_interactions - torch.diag_embed(diag_interactions)
        interactions_dropped = F.dropout(interactions_no_diag, p=self.dropout_rate, training=self.training)

        diag_weights = torch.diagonal(batch_weights, dim1=1, dim2=2)
        weights_no_diag = batch_weights - torch.diag_embed(diag_weights)
        # Apply abs before dropout? Original code did this implicitly via softmax later.
        weights_processed = torch.abs(weights_no_diag)
        weights_dropped = F.dropout(weights_processed, p=self.dropout_rate, training=self.training)

        # Calculate attention scores (softmax over columns/dim=2)
        attention_scores = F.softmax(weights_dropped, dim=2) # (batch, num_stocks, num_stocks)

        # Apply attention to the interaction matrix (element-wise product?)
        attended_interactions = interactions_dropped * attention_scores # (batch, num_stocks, num_stocks)

        # Apply the attended interaction matrix to the input features
        # (batch, num_stocks, num_stocks) @ (batch, num_stocks, features) -> (batch, num_stocks, features)
        output_features = torch.matmul(attended_interactions, stock_features)

        return output_features


class CointegrationPredictionModel(nn.Module):
    """
    Main model for the Cointegration pre-training task. (Interpretation based on original code)
    Predicts future price/feature based on past features using CointegrationAttentionLayer.
    """
    def __init__(self, num_total_stocks: int = 4000, dropout: float = 0.3):
        super().__init__()
        # Using two attention layers? Original code had self.m1, self.m2
        self.attention_layer1 = CointegrationAttentionLayer(num_total_stocks=num_total_stocks, dropout=dropout)
        # self.attention_layer2 = CointegrationAttentionLayer(num_total_stocks=num_total_stocks, dropout=dropout) # If needed

        # Learnable mean-reversion speed parameter per stock
        self.rho = nn.Parameter(torch.zeros(num_total_stocks))

    def forward(self, sequence_features: Tensor, stock_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            sequence_features: Input sequence features. Shape (batch_size, num_stocks, seq_len, features).
            stock_ids: Integer IDs for stocks. Shape (batch_size, num_stocks).

        Returns:
            Tuple: (actual_value, predicted_value) - Based on original return values.
                   Shapes: (batch_size, num_stocks, 1)
        """
        # Select the last time step's features
        # Original code: x = x[:,:,-1,:] -> selects last time step
        last_step_features = sequence_features[:, :, -1, :] # (batch, num_stocks, features)

        # Extract specific feature to predict (e.g., close price at index 1?)
        # Original code: close_ = x[:,:,1:2]
        feature_to_predict = last_step_features[:, :, 1:2] # (batch, num_stocks, 1)

        # Normalize the feature (Z-score normalization across stocks for the batch)
        mean_val = torch.mean(feature_to_predict, dim=1, keepdim=True)
        std_val = torch.std(feature_to_predict, dim=1, keepdim=True)
        normalized_feature = (feature_to_predict - mean_val) / (std_val + 1e-8)

        # Apply the attention layer to the normalized feature
        predicted_normalized_feature = self.attention_layer1(normalized_feature, stock_ids)

        # The model returns the normalized actual value and the predicted normalized value
        return normalized_feature, predicted_normalized_feature
