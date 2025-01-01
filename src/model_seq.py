
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn, sigmoid



class Trans(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        dim_model: int,
        dim_ff: int,
        seq_len: int,
        num_layers: int,
        dropout: float = 0.0,add_xdim=0,embeddim=0):
        super().__init__()
        self.position_encoder = PositionalEncoder(input_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=num_heads, dim_feedforward=dim_ff,dropout = 0.1,
        )
        layer_norm = nn.LayerNorm(dim_model)
        self.seq_len = seq_len
        self.input_size = input_size
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=layer_norm
        )
        self.fc1 = nn.Linear(input_size, dim_model,bias=False)


        if(embeddim!=0):
            self.layer1 = nn.Linear(embeddim*2, embeddim*2)
            self.fc2 = nn.Linear(dim_model + add_xdim + dim_model, dim_model // 2, bias=False)
        else:
            self.fc2 = nn.Linear(dim_model + add_xdim , dim_model // 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.score_layer = nn.Linear(dim_model // 2, 1,bias=False)

    def forward(self, x,addi_x=None):
        assert x.size(1) == self.seq_len and x.size(2) == self.input_size
        x = self.position_encoder(x)
        out = torch.relu(self.fc1(x))
        out = out.permute(1, 0, 2)
        out = self.encoder(out)
        out = out.permute(1, 0, 2)
        out = out[:, -1, :]
        if (addi_x is not None):
            marketembed,outstks=addi_x
            out_K = self.layer1(outstks)
            out_Q = out_K.permute(1,0)

            out_QK = torch.matmul(out_K,out_Q)


            self_attn = F.softmax(out_QK, dim=1).unsqueeze(2)
            out2=out.unsqueeze(0)
            outD = torch.sum(out2 * self_attn, dim=1)


            out = torch.cat([out, marketembed,outD], dim=1)
        out = self.dropout(torch.relu(self.fc2(out)))


        score = self.score_layer(self.dropout(out))

        return score




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, V)






class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 160):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i - 1] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x


