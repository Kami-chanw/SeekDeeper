from builtins import staticmethod
import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder


def make_src_mask(src, pad_idx, device=None):
    # [batch_size, 1, 1, src_len]
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(device)

def make_tgt_mask(tgt, pad_idx, device=None):
    batch_size, tgt_len = tgt.shape
    # [batch_size, 1, 1, tgt_len]
    tgt_pad_mask = Transformer.mask_src_mask(tgt, pad_idx, device)
    # [tgt_len, tgt_len]
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
    # [batch_size, 1, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
    return tgt_mask


class Transformer(nn.Module):

    def __init__(
        self,
        src_pad_idx,
        tgt_pad_idx,
        tgt_sos_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        dropout,
        device,
    ):
        """
        Constructor for the Transformer model.

        :param src_pad_idx: Padding index for the source sequences.
        :param tgt_pad_idx: Padding index for the target sequences.
        :param tgt_sos_idx: Start-of-sequence index for the target sequences.
        :param enc_voc_size: Vocabulary size of the encoder.
        :param dec_voc_size: Vocabulary size of the decoder.
        :param d_model: Dimensionality of the model.
        :param n_head: Number of attention heads.
        :param max_len: Maximum sequence length.
        :param ffn_hidden: Dimensionality of the feed-forward network.
        :param n_layers: Number of layers in the encoder and decoder.
        :param dropout: Dropout probability.
        :param device: Device to run the model on.
        """

        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.device = device
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            dropout=dropout,
            n_layers=n_layers,
            device=device,
        )

        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            dropout=dropout,
            n_layers=n_layers,
            device=device,
        )

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return output

    def make_src_mask(self, src):
        return make_src_mask(src, self.src_pad_idx, self.device)

    def make_tgt_mask(self, tgt):
        return make_tgt_mask(tgt, self.tgt_pad_idx, self.device)
