import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # Initialize position encoding matrix (shape: [max_len, d_model])
        pe = torch.zeros(max_len, d_model, device=device)

        # Create a tensor of shape [max_len, 1] with position indices
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(
            1
        )

        # Compute the div_term (shape: [d_model//2]) for the sin and cos functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * (-math.log(10000.0) / d_model)
        )

        # Apply sin/cos to even/odd indices in the position encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer, not a parameter (no gradients needed)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_len = x.size()
        # [seq_len, d_model]
        return self.pe[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout, device):
        super(TransformerEmbedding, self).__init__()

        self.tok_emb = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]
        tok_emb = self.tok_emb(x)  # [batch_size, seq_len, d_model]
        pos_emb = self.pos_emb(x)  # [seq_len, d_model]

        # [batch_size, seq_len, d_model]
        return self.drop_out(tok_emb + pos_emb)
