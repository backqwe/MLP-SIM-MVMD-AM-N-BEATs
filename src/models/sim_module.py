"""SIM (Similarity Module) for feature alignment between decomposed modes.

The SIM module computes cross-mode similarity attention so that the downstream
predictor can leverage inter-mode dependencies learned from the MVMD output.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import MultiHeadAttention


class SIMModule(nn.Module):
    """Similarity (SIM) module.

    Projects each MVMD mode into a shared embedding space, computes
    cross-mode attention, and returns a fused representation together
    with per-mode attention weights.

    Args:
        input_size: Feature dimension of each mode (``channels * time`` or
            a pre-flattened size).
        embed_dim: Dimension of the shared embedding space.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim

        # Project each mode to the embedding space
        self.projection = nn.Linear(input_size, embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=False,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, modes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity-weighted fusion of MVMD modes.

        Args:
            modes: Tensor of shape ``(batch, num_modes, input_size)``.

        Returns:
            Tuple of:
                - ``fused``: Fused representation ``(batch, num_modes, embed_dim)``.
                - ``attn_weights``: Attention weights ``(batch, num_modes, num_modes)``.
        """
        # Project modes: (batch, num_modes, embed_dim)
        projected = self.projection(modes)

        # Self-attention across modes
        fused, attn_weights = self.attn(projected)
        fused = self.out_proj(fused)
        return fused, attn_weights
