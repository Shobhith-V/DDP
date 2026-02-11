"""Evaluation metrics."""

import numpy as np


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def compute_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.std() < 1e-12 or target.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred.flatten(), target.flatten())[0, 1])
