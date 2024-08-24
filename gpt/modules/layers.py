import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, d_model, n_head, max_len, dropout):
        super().__init__()
        assert d_model % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)).view(
                1, 1, max_len, max_len
            ),
        )

    def forward(self, x, mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        casual_mask = self.bias[:, :, :T, :T]
        if mask is not None:
            casual_mask = casual_mask & mask
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=casual_mask,
                dropout_p=self.dropout if self.training else 0,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(casual_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, d_model, n_head, max_len, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = Attention(
            d_model=d_model, n_head=n_head, max_len=max_len, dropout=dropout
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(d_model, 4 * d_model),
                act=nn.GELU(approximate="tanh"),
                c_proj=nn.Linear(4 * d_model, d_model),
                dropout=nn.Dropout(dropout),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, mask=None):
        x = x + self.ln_1(self.attn(x, mask))
        x = x + self.ln_2(self.mlpf(x))
        return x
