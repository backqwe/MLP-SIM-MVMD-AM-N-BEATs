"""Logging utility for MLP-SIM-MVMD-AM-N-BEATs."""

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str = "mlp_sim_mvmd",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and return a configured logger.

    Outputs to *stdout* and optionally to a rotating file handler.

    Args:
        name: Logger name (also used as the filename prefix).
        log_level: Logging level string (``'DEBUG'``, ``'INFO'``, etc.).
        log_dir: Directory for the log file. Used only when ``log_file`` is
            also provided or auto-derived.
        log_file: Log file name. If ``None`` and ``log_dir`` is set, defaults
            to ``<name>.log`` inside ``log_dir``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler (optional)
    if log_dir is not None or log_file is not None:
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        file_path = log_file or os.path.join(log_dir or ".", f"{name}.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
