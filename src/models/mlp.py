"""Multi-Layer Perceptron (MLP) module."""

from typing import List, Optional
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable hidden layers.

    This MLP serves as the primary feature extractor in the MLP-SIM stage,
    learning a compact representation of the decomposed signal modes.

    Args:
        input_size: Number of input features.
        hidden_sizes: List of hidden layer sizes.
        output_size: Number of output features.
        dropout: Dropout probability applied after each hidden layer.
        activation: Activation function name ('relu', 'gelu', 'tanh', 'sigmoid').
        use_batch_norm: Whether to apply batch normalization after each hidden layer.
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(self._ACTIVATIONS.keys())}."
            )

        act_cls = self._ACTIVATIONS[activation]
        layers: List[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, input_size)`` or
               ``(batch, seq_len, input_size)``.

        Returns:
            Output tensor of shape ``(batch, output_size)`` or
            ``(batch, seq_len, output_size)``.
        """
        return self.network(x)
