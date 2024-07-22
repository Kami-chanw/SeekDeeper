import torch
from torch import nn

from .layers import *


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, dropout, device):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model=d_model, n_head=n_head, device=device
        )
        self.ln_1 = LayerNorm(normalized_shape=d_model, device=device)
        self.dropout = nn.Dropout(p=dropout)

        self.enc_dec_attention = MultiheadAttention(
            d_model=d_model, n_head=n_head, device=device
        )
        self.ln_2 = LayerNorm(normalized_shape=d_model, device=device)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, dropout=dropout, device=device
        )
        self.ln_3 = LayerNorm(normalized_shape=d_model, device=device)

    def forward(self, dec, enc, tgt_mask, src_mask):
        # 1. Apply self attention
        residual = dec
        x = self.self_attn(q=dec, k=dec, v=dec, mask=tgt_mask)
        x = self.dropout(x)

        # 2. Add and norm
        x = self.ln_1(x + residual)

        if enc is not None:
            # 3. Cross attention
            residual = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout(x)

            # 4. add and norm
            x = self.ln_2(x + residual)

        # 5. positionwise feed forward network
        residual = x
        x = self.dropout(self.ffn(x))

        # 6. add and norm
        x = self.ln_3(x + residual)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        n_head,
        n_layer,
        dropout,
        device,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        for layer in self.layers:
            tgt = layer(tgt, enc_src, tgt_mask, src_mask)

        return tgt
