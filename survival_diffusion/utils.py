"""Serialization and run logging helpers."""

import hashlib
from datetime import datetime

import numpy as np
import torch


def to_serializable(obj):
    """Convert tensors and NumPy values into JSON-compatible Python values."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def log_message(msg, log_file=None):
    """Print a timestamped message and optionally append it to a log file."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_msg = f"{timestamp} {msg}"
    print(full_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(full_msg + "\n")


def safe_wandb_name(name: str, max_len: int = 120) -> str:
    """Shorten a run name while retaining a deterministic hash suffix."""
    name = str(name)
    if len(name) <= max_len:
        return name
    suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    head = name[: max_len - 9]
    return f"{head}_{suffix}"
