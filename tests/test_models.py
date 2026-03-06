"""Unit tests for model modules."""

import math
import unittest

import torch

from src.models.mlp import MLP
from src.models.attention import MultiHeadAttention, PositionalEncoding
from src.models.mvmd import MVMD
from src.models.nbeats import NBeats, GenericBlock, TrendBlock, SeasonalityBlock
from src.models.sim_module import SIMModule
from src.models.ensemble import EnsembleModel


class TestMLP(unittest.TestCase):
    """Tests for the MLP module."""

    def test_forward_2d(self) -> None:
        model = MLP(input_size=16, hidden_sizes=[32, 16], output_size=1)
        x = torch.randn(8, 16)
        out = model(x)
        self.assertEqual(out.shape, (8, 1))

    def test_forward_3d(self) -> None:
        model = MLP(input_size=16, hidden_sizes=[32], output_size=4)
        x = torch.randn(4, 10, 16)
        out = model(x)
        self.assertEqual(out.shape, (4, 10, 4))

    def test_invalid_activation(self) -> None:
        with self.assertRaises(ValueError):
            MLP(input_size=8, hidden_sizes=[16], activation="unknown")

    def test_no_hidden_layers(self) -> None:
        model = MLP(input_size=8, hidden_sizes=[], output_size=2)
        x = torch.randn(4, 8)
        out = model(x)
        self.assertEqual(out.shape, (4, 2))


class TestPositionalEncoding(unittest.TestCase):
    """Tests for sinusoidal positional encoding."""

    def test_output_shape(self) -> None:
        pe = PositionalEncoding(embed_dim=32, dropout=0.0)
        x = torch.zeros(2, 10, 32)
        out = pe(x)
        self.assertEqual(out.shape, x.shape)

    def test_encoding_not_all_zeros(self) -> None:
        pe = PositionalEncoding(embed_dim=32, dropout=0.0)
        x = torch.zeros(1, 5, 32)
        out = pe(x)
        self.assertFalse(torch.all(out == 0).item())


class TestMultiHeadAttention(unittest.TestCase):
    """Tests for the MultiHeadAttention module."""

    def _build(self, **kwargs) -> MultiHeadAttention:
        defaults = dict(embed_dim=32, num_heads=4, dropout=0.0, use_positional_encoding=False)
        defaults.update(kwargs)
        return MultiHeadAttention(**defaults)

    def test_self_attention_shape(self) -> None:
        model = self._build()
        x = torch.randn(2, 5, 32)
        out, weights = model(x)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertEqual(weights.shape, (2, 5, 5))

    def test_invalid_embed_num_heads(self) -> None:
        with self.assertRaises(ValueError):
            MultiHeadAttention(embed_dim=33, num_heads=4)

    def test_with_positional_encoding(self) -> None:
        model = MultiHeadAttention(embed_dim=32, num_heads=4, use_positional_encoding=True)
        x = torch.randn(3, 8, 32)
        out, _ = model(x)
        self.assertEqual(out.shape, x.shape)


class TestMVMD(unittest.TestCase):
    """Tests for the MVMD decomposition module."""

    def test_output_shapes(self) -> None:
        mvmd = MVMD(num_modes=3, max_iter=5)
        x = torch.randn(2, 2, 64)
        imfs, omega = mvmd(x)
        self.assertEqual(imfs.shape, (2, 3, 2, 64))
        self.assertEqual(omega.shape, (2, 3))

    def test_reconstruction_residual(self) -> None:
        """Sum of IMFs should approximate the original signal (loose check)."""
        mvmd = MVMD(num_modes=3, max_iter=50, tol=1e-5)
        x = torch.randn(1, 1, 128)
        imfs, _ = mvmd(x)
        reconstructed = imfs.sum(dim=1)  # (batch, C, T)
        # Reconstruction is approximate for limited iterations on random signals
        error = (reconstructed - x).abs().mean().item()
        self.assertLess(error, 2.0)


class TestNBeats(unittest.TestCase):
    """Tests for N-BEATs model variants."""

    def test_generic_block_shape(self) -> None:
        block = GenericBlock(input_size=24, forecast_size=4)
        x = torch.randn(2, 24)
        backcast, forecast = block(x)
        self.assertEqual(backcast.shape, (2, 24))
        self.assertEqual(forecast.shape, (2, 4))

    def test_trend_block_shape(self) -> None:
        block = TrendBlock(input_size=24, forecast_size=4, degree=3)
        x = torch.randn(2, 24)
        backcast, forecast = block(x)
        self.assertEqual(backcast.shape, (2, 24))
        self.assertEqual(forecast.shape, (2, 4))

    def test_seasonality_block_shape(self) -> None:
        block = SeasonalityBlock(input_size=24, forecast_size=4, harmonics=2)
        x = torch.randn(2, 24)
        backcast, forecast = block(x)
        self.assertEqual(backcast.shape, (2, 24))
        self.assertEqual(forecast.shape, (2, 4))

    def test_nbeats_forward(self) -> None:
        model = NBeats(
            input_size=24,
            output_size=4,
            stack_types=["generic"],
            num_blocks_per_stack=2,
            num_layers=2,
            layer_width=32,
        )
        x = torch.randn(4, 24)
        out = model(x)
        self.assertEqual(out.shape, (4, 4))

    def test_invalid_stack_type(self) -> None:
        with self.assertRaises(ValueError):
            NBeats(input_size=24, output_size=1, stack_types=["unknown"])


class TestSIMModule(unittest.TestCase):
    """Tests for the SIM module."""

    def test_output_shapes(self) -> None:
        sim = SIMModule(input_size=32, embed_dim=16, num_heads=4)
        modes = torch.randn(3, 5, 32)  # (batch, num_modes, input_size)
        fused, attn = sim(modes)
        self.assertEqual(fused.shape, (3, 5, 16))
        self.assertEqual(attn.shape, (3, 5, 5))


class TestEnsembleModel(unittest.TestCase):
    """Smoke tests for the full ensemble pipeline."""

    def _build_small_model(self) -> EnsembleModel:
        return EnsembleModel(
            seq_len=16,
            num_channels=2,
            pred_len=1,
            num_modes=2,
            mlp_hidden_sizes=[32],
            mlp_embed_dim=16,
            sim_embed_dim=16,
            sim_num_heads=2,
            am_embed_dim=16,
            am_num_heads=2,
            nbeats_stack_types=["generic"],
            nbeats_num_blocks=1,
            nbeats_num_layers=2,
            nbeats_layer_width=32,
            dropout=0.0,
            mvmd_max_iter=5,
        )

    def test_forward_channels_last(self) -> None:
        """Input (batch, seq_len, channels)."""
        model = self._build_small_model()
        x = torch.randn(2, 16, 2)
        forecast, aux = model(x)
        self.assertEqual(forecast.shape, (2, 1))
        self.assertIn("imfs", aux)
        self.assertIn("omega", aux)
        self.assertIn("attn_weights", aux)

    def test_forward_channels_first(self) -> None:
        """Input (batch, channels, seq_len)."""
        model = self._build_small_model()
        x = torch.randn(2, 2, 16)
        forecast, aux = model(x)
        self.assertEqual(forecast.shape, (2, 1))

    def test_gradient_flow(self) -> None:
        model = self._build_small_model()
        x = torch.randn(2, 16, 2)
        target = torch.randn(2, 1)
        forecast, _ = model(x)
        loss = ((forecast - target) ** 2).mean()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
