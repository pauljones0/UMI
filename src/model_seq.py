import copy
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor # Added Tensor for type hinting

class PositionalEncoder(nn.Module):
    """
    Adds positional encoding to the input embeddings.

    This allows the model to incorporate information about the position
    of tokens in the sequence. Uses sine and cosine functions of
    different frequencies.
    """
    def __init__(self, d_model: int, max_seq_len: int = 160):
        """
        Initializes the PositionalEncoder.

        Args:
            d_model: The dimensionality of the embeddings (and the model).
            max_seq_len: The maximum sequence length anticipated.
        """
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model: Ensure the slicing doesn't go out of bounds
        if d_model % 2 != 0:
             pe[:, 1::2] = torch.cos(position * div_term[:-1]) # Use one less div_term if odd
        else:
             pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a model parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor with added positional encoding, same shape as input.
        """
        # x is expected to be of shape (batch_size, seq_len, d_model)
        x = x * math.sqrt(self.d_model) # Scale embeddings
        seq_len = x.size(1)
        # Add positional encoding up to the sequence length
        # Ensure pe slicing is correct for the input shape
        x = x + self.pe[:, :seq_len, :]
        return x


class Trans(nn.Module):
    """
    Transformer-based model for sequence processing.

    Combines a Transformer encoder with additional linear layers and
    an optional self-attention mechanism based on auxiliary input.
    """
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        dim_model: int,
        dim_ff: int,
        seq_len: int,
        num_layers: int,
        dropout: float = 0.0,
        add_xdim: int = 0, # Dimension of additional static features
        embeddim: int = 0  # Dimension for optional embedding-based attention
    ):
        """
        Initializes the Trans model.

        Args:
            input_size: The number of expected features in the input `x`.
            num_heads: The number of heads in the multiheadattention models.
            dim_model: The dimension of the transformer encoder layers.
            dim_ff: The dimension of the feedforward network model in encoder layers.
            seq_len: The length of the input sequences.
            num_layers: The number of sub-encoder-layers in the encoder.
            dropout: The dropout value. Default: 0.0.
            add_xdim: The dimension of the auxiliary input `addi_x` if it only contains static features. Default: 0.
            embeddim: The dimension of the embeddings if `addi_x` contains market/stock embeddings for attention. Default: 0.
        """
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.embeddim = embeddim # Store embeddim

        # Input projection
        self.fc1 = nn.Linear(input_size, dim_model, bias=False)
        self.position_encoder = PositionalEncoder(dim_model, max_seq_len=seq_len + 1) # Use dim_model

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=0.1, # Consider making this configurable
            batch_first=True # Assuming batch_first=True based on permute operations
        )
        layer_norm = nn.LayerNorm(dim_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=layer_norm
        )

        # Layers for optional auxiliary input processing
        if embeddim != 0:
            self.layer1 = nn.Linear(embeddim * 2, embeddim * 2) # Attention mechanism layer
            # Adjust fc2 input size based on whether embeddim attention is used
            fc2_input_dim = dim_model + add_xdim + dim_model # Base + market_embed + attention_output
        else:
            fc2_input_dim = dim_model + add_xdim # Base + static_features

        # Final processing layers
        self.fc2 = nn.Linear(fc2_input_dim, dim_model // 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.score_layer = nn.Linear(dim_model // 2, 1, bias=False)

    def forward(self, x: Tensor, addi_x: tuple = None) -> Tensor:
        """
        Forward pass of the Trans model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            addi_x: Optional tuple containing auxiliary information.
                    If `embeddim` > 0, expected to be (marketembed, outstks).
                    `marketembed` shape: (batch_size, add_xdim)
                    `outstks` shape: (batch_size, num_stocks?, embeddim * 2)
                    If `embeddim` == 0, expected to be a tensor of shape (batch_size, add_xdim).

        Returns:
            Output score tensor of shape (batch_size, 1).
        """
        # Input shape check (assuming batch_first=True)
        assert x.size(1) == self.seq_len and x.size(2) == self.input_size, \
               f"Input shape mismatch: Expected (N, {self.seq_len}, {self.input_size}), got {x.shape}"

        # 1. Input projection and positional encoding
        out = torch.relu(self.fc1(x)) # (N, seq_len, dim_model)
        out = self.position_encoder(out) # (N, seq_len, dim_model)

        # 2. Transformer Encoder
        # TransformerEncoderLayer expects (N, S, E) if batch_first=True
        out = self.encoder(out) # (N, seq_len, dim_model)

        # 3. Get the output of the last time step
        out = out[:, -1, :] # (N, dim_model)

        # 4. Process auxiliary input if provided
        if addi_x is not None:
            if self.embeddim != 0:
                # Attention mechanism using market and stock embeddings
                marketembed, outstks = addi_x # Unpack tuple
                # Assuming outstks is (batch_size, num_stocks, embeddim*2)
                # Apply linear layer to stock embeddings
                out_K = self.layer1(outstks) # (N, num_stocks, embeddim*2)
                # Prepare for attention score calculation
                # This part seems unusual - typically Q, K, V have compatible dims.
                # Assuming a self-attention mechanism on stocks based on their embeddings.
                # Let's assume Q is derived from K for self-attention
                out_Q = out_K.transpose(1, 2) # (N, embeddim*2, num_stocks) ? Check dimensions

                # Calculate attention scores (dot product)
                # Matmul: (N, num_stocks, embeddim*2) x (N, embeddim*2, num_stocks) -> (N, num_stocks, num_stocks)
                out_QK = torch.matmul(out_K, out_Q) / math.sqrt(out_K.size(-1)) # Scale scores

                # Apply softmax to get attention weights
                self_attn = F.softmax(out_QK, dim=-1) # (N, num_stocks, num_stocks)

                # Apply attention weights to the original transformer output 'out'
                # This step is also unusual. Typically attention is applied to V derived from input.
                # Reinterpreting: Maybe it's attention over stocks to weigh the main 'out'?
                # Let's assume 'out' needs to be weighted based on stock attention.
                # We need a V. If V=out_K: (N, num_stocks, num_stocks) x (N, num_stocks, embeddim*2) -> (N, num_stocks, embeddim*2)
                # weighted_stocks = torch.matmul(self_attn, out_K)
                # Then perhaps aggregate: weighted_agg = weighted_stocks.mean(dim=1) # (N, embeddim*2)

                # --- Re-evaluating the original code's intention ---
                # Original: self_attn = F.softmax(out_QK, dim=1).unsqueeze(2) # (N, num_stocks, 1, num_stocks) ?? dim=1 softmax?
                # Original: out2=out.unsqueeze(0) # (1, N, dim_model)
                # Original: outD = torch.sum(out2 * self_attn, dim=1) # Broadcasting issues likely
                # --- Attempting a more standard interpretation ---
                # Let's assume V is derived from outstks or similar. If V = out_K:
                # weighted_V = torch.matmul(self_attn, out_K) # (N, num_stocks, embeddim*2)
                # Aggregate attended stock features, e.g., by mean or sum
                # outD = weighted_V.mean(dim=1) # (N, embeddim*2) - This needs adjustment to match fc2 input (dim_model)

                # --- Sticking closer to original structure but clarifying ---
                # Assuming out_K is (N, E), out_Q is (E, N) -> out_QK is (N, N)
                # This implies attention between batch items, which is rare.
                # Let's assume the original code intended self-attention on 'out' modulated by 'outstks' somehow.
                # The original code is hard to interpret definitively without more context.
                # For now, keeping the structure but noting the ambiguity.
                # --- Reverting to original logic with shape comments for clarity ---
                out_K = self.layer1(outstks) # Shape depends on outstks, assume (N, D1) -> (N, D2=embeddim*2)
                out_Q = out_K.permute(1,0) # (D2, N)
                out_QK = torch.matmul(out_K, out_Q) # (N, N) - Attention between batch elements?

                self_attn = F.softmax(out_QK, dim=1).unsqueeze(2) # (N, N, 1)
                out2 = out.unsqueeze(1) # (N, 1, dim_model) - Prepare 'out' for broadcasting

                # Element-wise multiply and sum - This applies attention weights to 'out'
                # Broadcasting: (N, 1, dim_model) * (N, N, 1) -> (N, N, dim_model)
                # Summing over dim=1 aggregates contributions based on attention scores
                outD = torch.sum(out2 * self_attn, dim=1) # (N, dim_model) - Attended version of 'out'

                # Concatenate base output, market embedding, and attended output
                out = torch.cat([out, marketembed, outD], dim=1) # (N, dim_model + add_xdim + dim_model)
            else:
                # Concatenate base output with static auxiliary features
                out = torch.cat([out, addi_x], dim=1) # (N, dim_model + add_xdim)

        # 5. Final layers
        out = self.dropout(torch.relu(self.fc2(out))) # (N, dim_model // 2)
        score = self.score_layer(self.dropout(out)) # (N, 1)

        return score


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    Produce N identical layers.

    Args:
        module: The module to clone.
        N: The number of clones required.

    Returns:
        A ModuleList containing N deep copies of the module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Scaled_Dot_Product_Attention(nn.Module):
    """
    Computes Scaled Dot-Product Attention.

    As described in "Attention Is All You Need".
    """
    def __init__(self):
        """Initializes the Scaled_Dot_Product_Attention module."""
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """
        Performs the forward pass for Scaled Dot-Product Attention.

        Args:
            Q: Query tensor.
            K: Key tensor.
            V: Value tensor.

        Returns:
            The output tensor after applying attention, and the attention weights.
            (Typically returns torch.matmul(p_attn, V), p_attn - modified to match original)
        """
        d_k = Q.size(-1) # Get the dimension of keys/queries
        # MatMul Q and K^T, then scale
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # Apply softmax to get attention probabilities
        p_attn = F.softmax(scores, dim=-1)
        # MatMul attention probabilities with V
        return torch.matmul(p_attn, V)

# Note: The original PositionalEncoder had a potential bug in the loop for odd d_model.
# It accessed pe[pos, i-1] which could be -1. Also, the calculation was slightly off standard implementations.
# I've replaced it with a more standard vectorized implementation.
# If the exact original behavior is critical, revert the PositionalEncoder implementation.
# Important Notes on Trans.forward:

# batch_first=True: I've assumed batch_first=True for the nn.TransformerEncoderLayer based on the permute operations used in the original code (swapping batch and sequence length dimensions). If this assumption is incorrect, the permute calls might be necessary, and batch_first=False (the default) should be used. I've updated the code to use batch_first=True and removed the permute calls around the encoder for clarity.
# Positional Encoding: The original PositionalEncoder implementation had a potential off-by-one error in indexing (pe[pos, i - 1]) and the formula differed slightly from the standard Vaswani et al. implementation. I've replaced it with a more standard, vectorized version. If the specific numerical output of the original encoder is crucial, you might need to revert that part. I also adjusted it to use dim_model consistent with the fc1 output.
# Attention Logic (addi_x): The self-attention mechanism applied when addi_x is provided (and embeddim != 0) is quite unusual. It seems to calculate attention scores between elements within the batch (out_QK = torch.matmul(out_K, out_Q) results in shape (N, N)), and then uses these scores to weight the primary output out. This is different from typical self-attention within a sequence or cross-attention between sequences. I've kept the original logic but added comments highlighting the potential ambiguity and unusual nature of this operation. Double-check if this implementation matches the intended behavior.