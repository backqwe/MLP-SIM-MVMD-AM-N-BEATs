"""Multi-Head Attention mechanism module."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        embed_dim: Embedding / model dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout probability applied to the summed embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: embed_dim // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention / Cross-Attention module.

    Wraps ``torch.nn.MultiheadAttention`` with optional positional encoding
    and a residual feed-forward sub-layer (Transformer encoder block style).

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability for attention weights and feed-forward.
        use_positional_encoding: Whether to inject sinusoidal positional encoding.
        ff_dim: Hidden dimension of the feed-forward sub-layer.
            Defaults to ``4 * embed_dim``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        ff_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if use_positional_encoding:
            self.pos_enc: Optional[PositionalEncoding] = PositionalEncoding(
                embed_dim, dropout=dropout
            )
        else:
            self.pos_enc = None

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        ff_hidden = ff_dim if ff_dim is not None else 4 * embed_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention followed by a feed-forward block.

        Args:
            query: Tensor of shape ``(batch, seq_q, embed_dim)``.
            key: Key tensor. Defaults to ``query`` (self-attention).
            value: Value tensor. Defaults to ``query`` (self-attention).
            key_padding_mask: Boolean mask for padding in keys.
            attn_mask: Additive mask for attention scores.

        Returns:
            Tuple of:
                - Output tensor ``(batch, seq_q, embed_dim)``.
                - Attention weights ``(batch, seq_q, seq_k)``.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        if self.pos_enc is not None:
            query = self.pos_enc(query)
            if key is not query:
                key = self.pos_enc(key)
                value = self.pos_enc(value)

        attn_out, attn_weights = self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.norm1(query + attn_out)
        ff_out = self.feed_forward(x)
        out = self.norm2(x + ff_out)
        return out, attn_weights
