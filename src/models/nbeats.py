"""N-BEATs: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.

Reference:
    Oreshkin, B. N., et al. (2020). N-BEATS: Neural basis expansion analysis
    for interpretable time series forecasting. ICLR 2020.
    arXiv:1905.10437
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NBeatsBlock(nn.Module):
    """Single N-BEATs block producing backcast and forecast basis expansions.

    Args:
        input_size: Length of the lookback window fed to this block.
        theta_size: Dimension of the basis expansion coefficients.
        forecast_size: Length of the forecast horizon.
        num_layers: Number of fully-connected layers in the block MLP.
        layer_width: Width of each FC layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        forecast_size: int,
        num_layers: int = 4,
        layer_width: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.forecast_size = forecast_size

        layers: List[nn.Module] = []
        prev = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(prev, layer_width), nn.ReLU(), nn.Dropout(dropout)]
            prev = layer_width

        self.fc_stack = nn.Sequential(*layers)
        self.theta_b = nn.Linear(layer_width, theta_size, bias=False)
        self.theta_f = nn.Linear(layer_width, theta_size, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute backcast and forecast theta coefficients.

        Args:
            x: Residual input ``(batch, input_size)``.

        Returns:
            Tuple ``(theta_backcast, theta_forecast)``, both
            ``(batch, theta_size)``.
        """
        h = self.fc_stack(x)
        return self.theta_b(h), self.theta_f(h)


class GenericBlock(NBeatsBlock):
    """Generic N-BEATs block with learned basis functions.

    Args:
        input_size: Lookback window length.
        forecast_size: Forecast horizon length.
        num_layers: Number of FC layers.
        layer_width: FC layer width.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        num_layers: int = 4,
        layer_width: int = 256,
        dropout: float = 0.0,
    ) -> None:
        theta_size = input_size + forecast_size
        super().__init__(input_size, theta_size, forecast_size, num_layers, layer_width, dropout)

        self.backcast_basis = nn.Linear(input_size, input_size, bias=False)
        self.forecast_basis = nn.Linear(forecast_size, forecast_size, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_b, theta_f_coeff = super().forward(x)
        # Split theta into backcast and forecast parts
        backcast = self.backcast_basis(theta_b[:, : self.input_size])
        forecast = self.forecast_basis(theta_f_coeff[:, : self.forecast_size])
        return backcast, forecast


class TrendBlock(NBeatsBlock):
    """Trend block with polynomial basis functions.

    Args:
        input_size: Lookback window length.
        forecast_size: Forecast horizon length.
        degree: Polynomial degree for the trend basis.
        num_layers: Number of FC layers.
        layer_width: FC layer width.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        degree: int = 3,
        num_layers: int = 4,
        layer_width: int = 256,
        dropout: float = 0.0,
    ) -> None:
        theta_size = degree + 1
        super().__init__(input_size, theta_size, forecast_size, num_layers, layer_width, dropout)

        # Precompute polynomial basis matrices (not trainable)
        t_back = torch.linspace(0, 1, input_size).unsqueeze(0)    # (1, T_back)
        t_fore = torch.linspace(0, 1, forecast_size).unsqueeze(0)  # (1, T_fore)

        powers = torch.arange(theta_size).float().unsqueeze(1)     # (degree+1, 1)
        basis_b = t_back.pow(powers).T   # (T_back, degree+1)
        basis_f = t_fore.pow(powers).T   # (T_fore, degree+1)

        self.register_buffer("basis_b", basis_b)
        self.register_buffer("basis_f", basis_f)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_b, theta_f = super().forward(x)
        backcast = (theta_b.unsqueeze(1) * self.basis_b.unsqueeze(0)).sum(-1)
        forecast = (theta_f.unsqueeze(1) * self.basis_f.unsqueeze(0)).sum(-1)
        return backcast, forecast


class SeasonalityBlock(NBeatsBlock):
    """Seasonality block with Fourier series basis functions.

    Args:
        input_size: Lookback window length.
        forecast_size: Forecast horizon length.
        harmonics: Number of Fourier harmonics.
        num_layers: Number of FC layers.
        layer_width: FC layer width.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        harmonics: Optional[int] = None,
        num_layers: int = 4,
        layer_width: int = 256,
        dropout: float = 0.0,
    ) -> None:
        if harmonics is None:
            harmonics = forecast_size // 2
        theta_size = 2 * harmonics
        super().__init__(input_size, theta_size, forecast_size, num_layers, layer_width, dropout)

        t_back = torch.linspace(0, 1, input_size)
        t_fore = torch.linspace(0, 1, forecast_size)
        ks = torch.arange(1, harmonics + 1).float()  # (harmonics,)

        # Basis: [cos, sin] concatenated
        cos_b = torch.cos(2 * torch.pi * ks.unsqueeze(1) * t_back.unsqueeze(0)).T  # (T, H)
        sin_b = torch.sin(2 * torch.pi * ks.unsqueeze(1) * t_back.unsqueeze(0)).T
        cos_f = torch.cos(2 * torch.pi * ks.unsqueeze(1) * t_fore.unsqueeze(0)).T
        sin_f = torch.sin(2 * torch.pi * ks.unsqueeze(1) * t_fore.unsqueeze(0)).T

        self.register_buffer("basis_b", torch.cat([cos_b, sin_b], dim=-1))  # (T, 2H)
        self.register_buffer("basis_f", torch.cat([cos_f, sin_f], dim=-1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_b, theta_f = super().forward(x)
        backcast = (theta_b.unsqueeze(1) * self.basis_b.unsqueeze(0)).sum(-1)
        forecast = (theta_f.unsqueeze(1) * self.basis_f.unsqueeze(0)).sum(-1)
        return backcast, forecast


class NBeatsStack(nn.Module):
    """Stack of N-BEATs blocks of the same type.

    Args:
        block_type: One of ``'generic'``, ``'trend'``, ``'seasonality'``.
        input_size: Lookback window length.
        forecast_size: Forecast horizon length.
        num_blocks: Number of blocks in this stack.
        num_layers: FC layers per block.
        layer_width: FC layer width.
        dropout: Dropout probability.
        share_weights: Whether all blocks in the stack share weights.
    """

    _BLOCK_CLASSES = {
        "generic": GenericBlock,
        "trend": TrendBlock,
        "seasonality": SeasonalityBlock,
    }

    def __init__(
        self,
        block_type: str,
        input_size: int,
        forecast_size: int,
        num_blocks: int = 3,
        num_layers: int = 4,
        layer_width: int = 256,
        dropout: float = 0.0,
        share_weights: bool = False,
    ) -> None:
        super().__init__()

        if block_type not in self._BLOCK_CLASSES:
            raise ValueError(
                f"Unknown block_type '{block_type}'. "
                f"Choose from {list(self._BLOCK_CLASSES.keys())}."
            )

        block_cls = self._BLOCK_CLASSES[block_type]
        block_kwargs = dict(
            input_size=input_size,
            forecast_size=forecast_size,
            num_layers=num_layers,
            layer_width=layer_width,
            dropout=dropout,
        )
        if share_weights:
            shared = block_cls(**block_kwargs)
            self.blocks = nn.ModuleList([shared] * num_blocks)
        else:
            self.blocks = nn.ModuleList(
                [block_cls(**block_kwargs) for _ in range(num_blocks)]
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process residual through all blocks in the stack.

        Args:
            x: Residual signal ``(batch, input_size)``.

        Returns:
            Tuple of:
                - Updated residual ``(batch, input_size)``.
                - Cumulative forecast ``(batch, forecast_size)``.
        """
        forecast = torch.zeros(x.size(0), self.blocks[0].forecast_size, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(x)
            x = x - backcast
            forecast = forecast + block_forecast
        return x, forecast


class NBeats(nn.Module):
    """N-BEATs model composed of multiple stacks.

    Args:
        input_size: Lookback window length (number of past time steps).
        output_size: Forecast horizon length.
        stack_types: List of stack type names
            (``'generic'``, ``'trend'``, ``'seasonality'``).
        num_blocks_per_stack: Number of blocks in each stack.
        num_layers: FC layers per block.
        layer_width: FC layer width.
        dropout: Dropout probability.
        share_weights_in_stack: Whether blocks within a stack share weights.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        stack_types: Optional[List[str]] = None,
        num_blocks_per_stack: int = 3,
        num_layers: int = 4,
        layer_width: int = 256,
        dropout: float = 0.0,
        share_weights_in_stack: bool = False,
    ) -> None:
        super().__init__()

        if stack_types is None:
            stack_types = ["trend", "seasonality", "generic"]

        self.stacks = nn.ModuleList(
            [
                NBeatsStack(
                    block_type=stype,
                    input_size=input_size,
                    forecast_size=output_size,
                    num_blocks=num_blocks_per_stack,
                    num_layers=num_layers,
                    layer_width=layer_width,
                    dropout=dropout,
                    share_weights=share_weights_in_stack,
                )
                for stype in stack_types
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a multi-step forecast.

        Args:
            x: Lookback signal ``(batch, input_size)``.

        Returns:
            Forecast tensor ``(batch, output_size)``.
        """
        forecast = torch.zeros(
            x.size(0),
            self.stacks[0].blocks[0].forecast_size,
            device=x.device,
        )
        residual = x
        for stack in self.stacks:
            residual, stack_forecast = stack(residual)
            forecast = forecast + stack_forecast
        return forecast
