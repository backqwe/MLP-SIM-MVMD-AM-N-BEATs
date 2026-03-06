"""Evaluation metrics for time series prediction."""

from typing import Dict

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground-truth values, shape ``(N,)`` or ``(N, H)``.
        y_pred: Predicted values, same shape.

    Returns:
        Scalar MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Scalar RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error (%).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar MAPE value in percent.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Scalar R² value (1.0 is perfect prediction).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Scalar MSE value.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all standard evaluation metrics.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys ``'mae'``, ``'rmse'``, ``'mape'``,
        ``'r2'``, ``'mse'``.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mse": mse(y_true, y_pred),
    }
