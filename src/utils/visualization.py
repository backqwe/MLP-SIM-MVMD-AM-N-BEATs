"""Visualisation utilities for predictions and training curves."""

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction vs Ground Truth",
    xlabel: str = "Time Step",
    ylabel: str = "Value",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot predicted versus actual time series values.

    Args:
        y_true: Ground-truth values, shape ``(N,)``.
        y_pred: Predicted values, shape ``(N,)``.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure` object.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Ground Truth", linewidth=1.5, color="steelblue")
    ax.plot(y_pred, label="Prediction", linewidth=1.5, linestyle="--", color="tomato")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    title: str = "Training / Validation Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training (and optional validation) loss curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure` object.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train", linewidth=1.5, color="steelblue")
    if val_losses is not None:
        ax.plot(epochs, val_losses, label="Validation", linewidth=1.5, color="tomato")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_imfs(
    imfs: np.ndarray,
    channel: int = 0,
    title: str = "MVMD Intrinsic Mode Functions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the decomposed intrinsic mode functions for one channel.

    Args:
        imfs: IMF array of shape ``(num_modes, channels, time)``.
        channel: Which channel to visualise.
        title: Figure title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure` object.
    """
    num_modes = imfs.shape[0]
    fig, axes = plt.subplots(num_modes, 1, figsize=(12, 2 * num_modes), sharex=True)
    if num_modes == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        ax.plot(imfs[k, channel], linewidth=1.0)
        ax.set_ylabel(f"IMF {k + 1}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time Step")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
