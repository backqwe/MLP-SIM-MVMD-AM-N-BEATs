"""Ensemble model: full MLP-SIM-MVMD-AM-N-BEATs pipeline.

This module integrates all components into a unified, end-to-end trainable
model for multivariate time series forecasting:

    1. **MVMD** decomposes the input into ``K`` intrinsic mode functions.
    2. **MLP** encodes each IMF independently.
    3. **SIM** computes cross-mode similarity attention for a fused summary.
    4. **AM (MultiHeadAttention)** re-weights the fused representation over
       the time axis.
    5. **N-BEATs** generates the final forecast from the attended features.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.mvmd import MVMD
from src.models.mlp import MLP
from src.models.sim_module import SIMModule
from src.models.attention import MultiHeadAttention
from src.models.nbeats import NBeats


class EnsembleModel(nn.Module):
    """MLP-SIM-MVMD-AM-N-BEATs ensemble model.

    Args:
        seq_len: Input sequence length (number of past time steps).
        num_channels: Number of input channels / features.
        pred_len: Forecast horizon length.
        num_modes: Number of MVMD modes.
        mlp_hidden_sizes: Hidden layer sizes for the per-mode MLP encoder.
        mlp_embed_dim: Output dimension of the MLP encoder (= SIM input_size).
        sim_embed_dim: Embedding dimension inside the SIM module.
        sim_num_heads: Number of attention heads in SIM.
        am_embed_dim: Embedding dimension of the AM attention layer.
        am_num_heads: Number of attention heads in AM.
        nbeats_stack_types: Stack types for N-BEATs.
        nbeats_num_blocks: Blocks per N-BEATs stack.
        nbeats_num_layers: FC layers per N-BEATs block.
        nbeats_layer_width: Width of N-BEATs FC layers.
        dropout: Global dropout probability.
        mvmd_alpha: MVMD bandwidth constraint.
        mvmd_tau: MVMD noise tolerance.
        mvmd_tol: MVMD convergence tolerance.
        mvmd_max_iter: MVMD maximum iterations.
        use_residual: Whether to add a direct residual connection from the
            last time step of the input to the final forecast.
    """

    def __init__(
        self,
        seq_len: int = 24,
        num_channels: int = 1,
        pred_len: int = 1,
        num_modes: int = 5,
        mlp_hidden_sizes: Optional[List[int]] = None,
        mlp_embed_dim: int = 64,
        sim_embed_dim: int = 64,
        sim_num_heads: int = 4,
        am_embed_dim: int = 64,
        am_num_heads: int = 4,
        nbeats_stack_types: Optional[List[str]] = None,
        nbeats_num_blocks: int = 3,
        nbeats_num_layers: int = 4,
        nbeats_layer_width: int = 256,
        dropout: float = 0.1,
        mvmd_alpha: float = 2000.0,
        mvmd_tau: float = 0.0,
        mvmd_tol: float = 1e-7,
        mvmd_max_iter: int = 500,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if mlp_hidden_sizes is None:
            mlp_hidden_sizes = [128, 64]
        if nbeats_stack_types is None:
            nbeats_stack_types = ["trend", "seasonality", "generic"]

        self.seq_len = seq_len
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.num_modes = num_modes
        self.use_residual = use_residual

        # ---------- Stage 1: MVMD ----------
        self.mvmd = MVMD(
            num_modes=num_modes,
            alpha=mvmd_alpha,
            tau=mvmd_tau,
            tol=mvmd_tol,
            max_iter=mvmd_max_iter,
        )

        # ---------- Stage 2: Per-mode MLP encoder ----------
        # Each mode has shape (batch, channels, seq_len) -> flatten -> MLP
        mode_flat_size = num_channels * seq_len
        self.mlp_encoder = MLP(
            input_size=mode_flat_size,
            hidden_sizes=mlp_hidden_sizes,
            output_size=mlp_embed_dim,
            dropout=dropout,
        )

        # ---------- Stage 3: SIM (cross-mode attention) ----------
        self.sim = SIMModule(
            input_size=mlp_embed_dim,
            embed_dim=sim_embed_dim,
            num_heads=sim_num_heads,
            dropout=dropout,
        )

        # ---------- Stage 4: AM (temporal attention) ----------
        self.am = MultiHeadAttention(
            embed_dim=am_embed_dim,
            num_heads=am_num_heads,
            dropout=dropout,
            use_positional_encoding=True,
        )

        # Project SIM output to AM embed_dim if necessary
        if sim_embed_dim != am_embed_dim:
            self.sim_to_am = nn.Linear(sim_embed_dim, am_embed_dim)
        else:
            self.sim_to_am = nn.Identity()

        # ---------- Stage 5: N-BEATs forecast head ----------
        nbeats_input = num_modes * am_embed_dim
        self.nbeats = NBeats(
            input_size=nbeats_input,
            output_size=pred_len,
            stack_types=nbeats_stack_types,
            num_blocks_per_stack=nbeats_num_blocks,
            num_layers=nbeats_num_layers,
            layer_width=nbeats_layer_width,
            dropout=dropout,
        )

        # Optional residual projection
        if use_residual:
            self.residual_proj = nn.Linear(num_channels, pred_len)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, num_channels)``
               or ``(batch, num_channels, seq_len)``.
               If ``x.shape[1] == seq_len`` the tensor is assumed to be
               ``(batch, seq_len, channels)`` and will be transposed.

        Returns:
            Tuple of:
                - ``forecast``: Predicted values ``(batch, pred_len)``.
                - ``aux``: Auxiliary outputs dict with keys
                  ``'imfs'``, ``'omega'``, ``'attn_weights'``.
        """
        # Normalise to (batch, channels, seq_len)
        if x.shape[1] == self.seq_len:
            x = x.transpose(1, 2)  # -> (batch, channels, seq_len)

        batch = x.shape[0]

        # 1. MVMD decomposition
        imfs, omega = self.mvmd(x)  # (batch, K, C, T), (batch, K)

        # 2. Flatten + MLP encode each mode
        modes_flat = imfs.view(batch, self.num_modes, -1)  # (batch, K, C*T)
        mode_embeds = self.mlp_encoder(modes_flat)          # (batch, K, embed)

        # 3. SIM cross-mode attention
        fused, attn_weights = self.sim(mode_embeds)         # (batch, K, sim_dim)
        fused = self.sim_to_am(fused)                       # (batch, K, am_dim)

        # 4. AM temporal attention (treat modes as the sequence dimension)
        attended, _ = self.am(fused)                        # (batch, K, am_dim)

        # 5. N-BEATs forecast
        nbeats_input = attended.reshape(batch, -1)          # (batch, K * am_dim)
        forecast = self.nbeats(nbeats_input)                # (batch, pred_len)

        # Optional residual from last observed time step
        if self.use_residual:
            last = x[:, :, -1]                              # (batch, channels)
            forecast = forecast + self.residual_proj(last)  # (batch, pred_len)

        aux = {"imfs": imfs, "omega": omega, "attn_weights": attn_weights}
        return forecast, aux
