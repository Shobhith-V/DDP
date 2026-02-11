"""Checkpoint save/load for training resumption."""

import os
import torch
from typing import Any


def save_checkpoint(state: dict[str, Any], path: str) -> None:
    """Save checkpoint dict to path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load checkpoint from path."""
    return torch.load(path, map_location=device, weights_only=False)
