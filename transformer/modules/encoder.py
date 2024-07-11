from torch import nn

from .embedding import TransformerEmbedding
from .layers import *


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, dropout, device):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model=d_model, n_head=n_head, device=device
        )
        self.ln_1 = LayerNorm(normalized_shape=d_model, device=device)
        self.dropout = nn.Dropout(p=dropout)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, dropout=dropout, device=device
        )
        self.ln_2 = LayerNorm(normalized_shape=d_model, device=device)

    def forward(self, x, src_mask):
        # 1. Self-Attention sublayer
        # x: [batch_size, seq_len, d_model]
        residual = x
        x = self.self_attn(x, x, x, src_mask)
        x = self.dropout(x)

        # 2. Add and norm
        x = self.ln_1(x + residual)

        # 3. Feed-Forward sublayer
        residual = x
        x = self.dropout(self.ffn(x))

        # 4. Add and norm
        x = self.ln_2(x + residual)

        return x


class Encoder(nn.Module):

    def __init__(
        self,
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layer,
        dropout,
        device,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            dropout=dropout,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
