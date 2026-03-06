"""Unit tests for data loading and preprocessing utilities."""

import unittest

import numpy as np

from src.data.preprocessing import (
    normalize,
    denormalize,
    create_sequences,
    train_val_test_split,
    remove_outliers,
    fill_missing,
)
from src.data.dataset import TimeSeriesDataset


class TestNormalize(unittest.TestCase):
    """Tests for Z-score normalization helpers."""

    def test_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(loc=5.0, scale=2.0, size=(100, 3)).astype(np.float32)
        normed, mean, std = normalize(data)
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(normed.std(axis=0), 1.0, atol=1e-2)

    def test_roundtrip(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(size=(50, 4)).astype(np.float32)
        normed, mean, std = normalize(data)
        recovered = denormalize(normed, mean, std)
        np.testing.assert_allclose(recovered, data, atol=1e-5)

    def test_provided_stats(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        normed, _, _ = normalize(data, mean=np.array([2.0, 3.0]), std=np.array([1.0, 1.0]))
        np.testing.assert_allclose(normed, [[-1.0, -1.0], [1.0, 1.0]], atol=1e-5)


class TestCreateSequences(unittest.TestCase):
    """Tests for the sliding-window sequence creator."""

    def setUp(self) -> None:
        self.T, self.F = 100, 3
        self.data = np.arange(self.T * self.F, dtype=np.float32).reshape(self.T, self.F)

    def test_number_of_samples(self) -> None:
        seqs, tgts = create_sequences(self.data, seq_len=10, pred_len=1)
        self.assertEqual(len(seqs), self.T - 10 - 1 + 1)
        self.assertEqual(len(tgts), len(seqs))

    def test_sequence_shape(self) -> None:
        seqs, tgts = create_sequences(self.data, seq_len=10, pred_len=2)
        self.assertEqual(seqs.shape, (len(seqs), 10, self.F))
        self.assertEqual(tgts.shape, (len(tgts), 2))

    def test_target_column_is_last(self) -> None:
        """Targets should come from the last column of data."""
        seqs, tgts = create_sequences(self.data, seq_len=5, pred_len=1)
        expected_first_target = self.data[5, -1]
        self.assertAlmostEqual(tgts[0, 0], expected_first_target)


class TestTrainValTestSplit(unittest.TestCase):
    """Tests for the chronological train / val / test splitter."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.seqs = rng.random((100, 10, 3)).astype(np.float32)
        self.tgts = rng.random((100, 1)).astype(np.float32)

    def test_sizes(self) -> None:
        (X_tr, _), (X_val, _), (X_te, _) = train_val_test_split(
            self.seqs, self.tgts, train_ratio=0.7, val_ratio=0.1
        )
        self.assertEqual(len(X_tr), 70)
        self.assertEqual(len(X_val), 10)
        self.assertEqual(len(X_te), 20)

    def test_no_data_leakage(self) -> None:
        """Splits must be contiguous and non-overlapping."""
        (X_tr, _), (X_val, _), (X_te, _) = train_val_test_split(
            self.seqs, self.tgts, train_ratio=0.7, val_ratio=0.1
        )
        # Last element of train != first element of val (they are contiguous slices)
        self.assertFalse(np.array_equal(X_tr[-1], X_val[0]))


class TestRemoveOutliers(unittest.TestCase):
    """Tests for the outlier removal utility."""

    def test_replaces_extreme_values(self) -> None:
        data = np.ones((50, 2), dtype=np.float32)
        data[10, 0] = 1000.0  # extreme outlier
        cleaned = remove_outliers(data, z_threshold=3.0)
        self.assertLess(abs(cleaned[10, 0]), 10.0)


class TestFillMissing(unittest.TestCase):
    """Tests for the missing-value interpolation utility."""

    def test_no_nans_after_fill(self) -> None:
        data = np.array(
            [[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan], [4.0, 8.0]], dtype=np.float32
        )
        filled = fill_missing(data)
        self.assertFalse(np.isnan(filled).any())


class TestTimeSeriesDataset(unittest.TestCase):
    """Tests for the TimeSeriesDataset class."""

    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.seqs = rng.random((20, 10, 3)).astype(np.float32)
        self.tgts = rng.random((20, 1)).astype(np.float32)

    def test_len(self) -> None:
        ds = TimeSeriesDataset(self.seqs, self.tgts)
        self.assertEqual(len(ds), 20)

    def test_getitem_shapes(self) -> None:
        import torch

        ds = TimeSeriesDataset(self.seqs, self.tgts)
        x, y = ds[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(x.shape, (10, 3))
        self.assertEqual(y.shape, (1,))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            TimeSeriesDataset(self.seqs, self.tgts[:10])


if __name__ == "__main__":
    unittest.main()
