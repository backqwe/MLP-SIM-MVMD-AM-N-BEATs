"""Prediction / inference script for MLP-SIM-MVMD-AM-N-BEATs."""

import argparse
import os

import numpy as np
import torch
import yaml

from src.data.dataset import create_dataloaders
from src.models.ensemble import EnsembleModel
from src.utils.logger import get_logger
from src.utils.metrics import compute_all_metrics
from src.utils.visualization import plot_predictions


def get_device(device_str: str) -> torch.device:
    """Resolve device string to a :class:`torch.device`."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


@torch.no_grad()
def predict(
    model: EnsembleModel,
    loader,
    device: torch.device,
) -> tuple:
    """Run inference on a data loader.

    Args:
        model: Trained :class:`~src.models.ensemble.EnsembleModel`.
        loader: DataLoader providing ``(x, y)`` batches.
        device: Inference device.

    Returns:
        Tuple of ``(predictions, targets)`` as numpy arrays of shape ``(N,)``.
    """
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        out, _ = model(x)
        preds.append(out.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds).squeeze(), np.concatenate(targets).squeeze()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with MLP-SIM-MVMD-AM-N-BEATs")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory to save prediction outputs",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger(
        name="predict",
        log_level=cfg["logging"]["log_level"],
        log_dir=cfg["logging"]["log_dir"],
    )

    device = get_device(cfg["device"])
    logger.info("Using device: %s", device)

    # Build data loaders (we only use the test split)
    logger.info("Loading data...")
    _, _, test_loader, _ = create_dataloaders(
        data=cfg["data"]["train_path"],
        target_col=cfg["data"]["target_col"],
        feature_cols=cfg["data"]["feature_cols"],
        seq_len=cfg["data"]["seq_len"],
        pred_len=cfg["data"]["pred_len"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # Build and load model
    model = EnsembleModel(
        seq_len=cfg["data"]["seq_len"],
        pred_len=cfg["data"]["pred_len"],
        num_modes=cfg["mvmd"]["num_modes"],
        mlp_embed_dim=cfg["sim"]["embed_dim"],
        sim_embed_dim=cfg["sim"]["embed_dim"],
        sim_num_heads=cfg["sim"]["num_heads"],
        am_embed_dim=cfg["attention"]["embed_dim"],
        am_num_heads=cfg["attention"]["num_heads"],
        nbeats_stack_types=cfg["nbeats"]["stack_types"],
        nbeats_num_blocks=cfg["nbeats"]["num_blocks_per_stack"],
        nbeats_num_layers=cfg["nbeats"]["num_layers"],
        nbeats_layer_width=cfg["nbeats"]["layer_width"],
        dropout=cfg["nbeats"]["dropout"],
        mvmd_alpha=cfg["mvmd"]["alpha"],
        mvmd_tau=cfg["mvmd"]["tau"],
        mvmd_tol=cfg["mvmd"]["tol"],
        mvmd_max_iter=cfg["mvmd"]["max_iter"],
        use_residual=cfg["ensemble"]["use_residual"],
    ).to(device)

    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    # Inference
    y_pred, y_true = predict(model, test_loader, device)

    # Metrics
    metrics = compute_all_metrics(y_true, y_pred)
    logger.info("Test metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %.6f", k.upper(), v)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "predictions.npy"), y_pred)
    np.save(os.path.join(args.output_dir, "targets.npy"), y_true)

    plot_predictions(
        y_true, y_pred,
        save_path=os.path.join(args.output_dir, "predictions.png"),
    )
    logger.info("Results saved to '%s'", args.output_dir)


if __name__ == "__main__":
    main()
