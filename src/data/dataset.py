"""Dataset loading and preprocessing utilities."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocessing import normalize, create_sequences, train_val_test_split


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for sliding-window time series sequences.

    Args:
        sequences: Input sequences array of shape ``(N, seq_len, features)``.
        targets: Target values array of shape ``(N, pred_len)``.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        if len(sequences) != len(targets):
            raise ValueError(
                f"sequences length ({len(sequences)}) must match "
                f"targets length ({len(targets)})."
            )
        self.sequences = torch.from_numpy(sequences).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def load_csv(
    path: str,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    datetime_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CSV file and return feature and target arrays.

    Args:
        path: Path to the CSV file.
        target_col: Name of the target column.
        feature_cols: List of feature column names. If ``None``, all columns
            except ``target_col`` (and ``datetime_col``) are used.
        datetime_col: Optional datetime index column to drop.

    Returns:
        Tuple of:
            - ``features``: Array of shape ``(T, num_features)``.
            - ``targets``: Array of shape ``(T,)``.
    """
    df = pd.read_csv(path)
    if datetime_col and datetime_col in df.columns:
        df = df.drop(columns=[datetime_col])

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    return features, targets


def create_dataloaders(
    data: Union[str, np.ndarray],
    target: Optional[np.ndarray] = None,
    target_col: str = "target",
    feature_cols: Optional[List[str]] = None,
    seq_len: int = 24,
    pred_len: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize_data: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[Tuple]]:
    """Build train / val / test DataLoaders from a CSV path or numpy arrays.

    Args:
        data: Path to a CSV file *or* a pre-loaded feature array ``(T, F)``.
        target: Target array ``(T,)`` when ``data`` is a numpy array.
        target_col: Target column name (used only when ``data`` is a CSV path).
        feature_cols: Feature column names (used only when ``data`` is a CSV path).
        seq_len: Input sequence length.
        pred_len: Forecast horizon length.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        batch_size: Mini-batch size.
        num_workers: DataLoader worker count.
        normalize_data: Whether to apply Z-score normalisation.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader, scaler_params)``
        where ``scaler_params`` is ``(mean, std)`` or ``None`` if normalisation
        was not applied.
    """
    if isinstance(data, str):
        features, target_arr = load_csv(data, target_col, feature_cols)
    else:
        features = data
        target_arr = target if target is not None else data[:, -1]

    scaler_params = None
    if normalize_data:
        features, feat_mean, feat_std = normalize(features)
        scaler_params = (feat_mean, feat_std)

    # Combine into a single array: features + target as last column
    combined = np.concatenate([features, target_arr.reshape(-1, 1)], axis=1)
    seqs, tgts = create_sequences(combined, seq_len, pred_len)

    # Split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        seqs, tgts, train_ratio, val_ratio
    )

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, scaler_params
