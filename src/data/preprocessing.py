"""Data preprocessing utility functions."""

from typing import Optional, Tuple

import numpy as np


def normalize(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Z-score normalization column-wise.

    Args:
        data: Array of shape ``(T, F)``.
        mean: Pre-computed column means. Computed from ``data`` if ``None``.
        std: Pre-computed column standard deviations. Computed if ``None``.
        eps: Small constant added to std to avoid division by zero.

    Returns:
        Tuple of ``(normalized, mean, std)``.
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    return (data - mean) / (std + eps), mean, std


def denormalize(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Reverse Z-score normalization.

    Args:
        data: Normalized array.
        mean: Column means used during normalization.
        std: Column standard deviations used during normalization.
        eps: Same eps used during normalization.

    Returns:
        Denormalised array of the same shape.
    """
    return data * (std + eps) + mean


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    pred_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping input/target sequences with a sliding window.

    The target is taken from the **last column** of ``data``.

    Args:
        data: Array of shape ``(T, F)`` where the last column is the target.
        seq_len: Input window length.
        pred_len: Number of future steps to predict.

    Returns:
        Tuple of:
            - ``sequences``: Shape ``(N, seq_len, F)``.
            - ``targets``: Shape ``(N, pred_len)``.
    """
    sequences, targets = [], []
    total = len(data)
    for i in range(total - seq_len - pred_len + 1):
        sequences.append(data[i : i + seq_len])
        targets.append(data[i + seq_len : i + seq_len + pred_len, -1])
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_val_test_split(
    sequences: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split sequences into train / validation / test sets (chronological).

    Args:
        sequences: Array of shape ``(N, seq_len, F)``.
        targets: Array of shape ``(N, pred_len)``.
        train_ratio: Fraction of samples for training.
        val_ratio: Fraction of samples for validation.

    Returns:
        Tuple of ``((X_train, y_train), (X_val, y_val), (X_test, y_test))``.
    """
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = sequences[:train_end], targets[:train_end]
    X_val, y_val = sequences[train_end:val_end], targets[train_end:val_end]
    X_test, y_test = sequences[val_end:], targets[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def remove_outliers(
    data: np.ndarray,
    z_threshold: float = 3.0,
) -> np.ndarray:
    """Replace outliers (beyond ``z_threshold`` std) with column medians.

    Args:
        data: Array of shape ``(T, F)``.
        z_threshold: Z-score threshold above which a value is considered an outlier.

    Returns:
        Array with outliers replaced by column medians.
    """
    data = data.copy()
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    median = np.median(data, axis=0)
    z_scores = np.abs((data - mean) / (std + 1e-8))
    mask = z_scores > z_threshold
    data[mask] = np.broadcast_to(median, data.shape)[mask]
    return data


def fill_missing(
    data: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate missing (NaN) values in each column.

    Args:
        data: Array of shape ``(T, F)`` possibly containing NaN values.
        method: Interpolation method passed to ``pd.Series.interpolate``.

    Returns:
        Array with NaNs filled.
    """
    import pandas as pd

    df = pd.DataFrame(data)
    df = df.interpolate(method=method, axis=0).ffill().bfill()
    return df.values.astype(np.float32)
