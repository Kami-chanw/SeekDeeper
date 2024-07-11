import torch
from torch import nn
import math


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            # Learnable parameters
            self.gamma = nn.Parameter(torch.ones(normalized_shape, device=device))
            self.beta = nn.Parameter(torch.zeros(normalized_shape, device=device))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        # x: [batch_size, ..., normalized_shape]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)

        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, n_head, seq_len, d_key = k.size()

        # 1. Compute the dot product between query and key^T
        k_t = k.transpose(-2, -1)
        scores = q @ k_t / math.sqrt(d_key)

        # 2. Apply mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 3. Apply softmax to get attention weights
        attn = self.softmax(scores)

        # 4. Compute the weighted sum of values
        out = attn @ v

        return out, attn


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, n_head, device):
        super(MultiheadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model, device=device)
        self.w_k = nn.Linear(d_model, d_model, device=device)
        self.w_v = nn.Linear(d_model, d_model, device=device)
        self.w_concat = nn.Linear(d_model, d_model, device=device)

    def forward(self, q, k, v, mask=None):
        # 1. Linear projections, [batch_size, length, d_model]
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Split tensor by number of heads, [batch_size, length, n_head, d_key]
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. Apply attention
        out, attn = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer, [batch_size, length, d_model]
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_key = d_model // self.n_head
        return tensor.view(batch_size, length, self.n_head, d_key).transpose(1, 2)

    def concat(self, tensor):
        batch_size, head, length, d_key = tensor.size()
        d_model = head * d_key

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, dropout, device):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden, device=device)
        self.linear2 = nn.Linear(hidden, d_model, device=device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
