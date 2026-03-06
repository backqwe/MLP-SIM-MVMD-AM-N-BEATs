"""Training entry-point script for MLP-SIM-MVMD-AM-N-BEATs."""

import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from src.data.dataset import create_dataloaders
from src.models.ensemble import EnsembleModel
from src.utils.logger import get_logger
from src.utils.metrics import compute_all_metrics
from src.utils.visualization import plot_loss_curves


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Resolve device string to a :class:`torch.device`."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Construct optimizer from config."""
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    name = cfg["training"]["optimizer"].lower()
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    """Construct LR scheduler from config."""
    name = cfg["training"].get("scheduler", "none").lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    if name == "step":
        return StepLR(
            optimizer,
            step_size=cfg["training"]["scheduler_step_size"],
            gamma=cfg["training"]["scheduler_gamma"],
        )
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    return None


def train_epoch(
    model: EnsembleModel,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred, _ = model(x)
        loss = criterion(pred, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: EnsembleModel,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate on a data loader and return average loss."""
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred, _ = model(x)
        total_loss += criterion(pred, y).item() * len(x)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP-SIM-MVMD-AM-N-BEATs")
    parser.add_argument("--config", default="config/default_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger(
        name="train",
        log_level=cfg["logging"]["log_level"],
        log_dir=cfg["logging"]["log_dir"],
    )
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    logger.info("Using device: %s", device)

    # Build dataloaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(
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

    # Build model
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

    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    loss_name = cfg["training"].get("loss", "mse").lower()
    criterion: nn.Module = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.HuberLoss(),
    }[loss_name]

    save_dir = cfg["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            cfg["training"]["grad_clip"],
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        logger.info(
            "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f",
            epoch, cfg["training"]["epochs"], train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            logger.info("  Saved best model (val_loss=%.6f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    plot_loss_curves(
        train_losses, val_losses,
        save_path=os.path.join(save_dir, "loss_curves.png"),
    )

    logger.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
