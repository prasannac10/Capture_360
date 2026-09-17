"""Attention-based aggregation over a variable number of high-resolution tiles."""
import torch
import torch.nn as nn


class TileAttentionAggregator(nn.Module):
    def __init__(self, dim: int, heads: int = 8, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                                       dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, mask=None):
        # tokens [B,T,D], mask True for valid tiles.
        key_padding = None if mask is None else ~mask.bool()
        x = tokens
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding)
        return self.norm(x)


class SceneToken(nn.Module):
    """Pool variable tile tokens into one global scene token without losing per-tile tokens."""
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, mask=None):
        q = self.query.expand(tokens.shape[0], -1, -1)
        key_padding = None if mask is None else ~mask.bool()
        out, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding)
        return self.norm(out[:, 0])
