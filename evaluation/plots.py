"""Plotting for results."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_full_results(
    brain_losses: list,
    mlp_losses: list,
    feedback_losses: list,
    target_ecg: np.ndarray,
    predicted_ecg_baseline: np.ndarray,
    predicted_ecg_feedback: np.ndarray,
    target_eeg: np.ndarray,
    P_out_baseline: np.ndarray,
    t: np.ndarray,
    t_duration: float,
    target_idx: int,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(15, 20))

    axes[0].plot(brain_losses)
    axes[0].set_title("Stage 1: Brain Pre-training Loss")
    axes[0].grid(True)

    axes[1].plot(mlp_losses)
    axes[1].set_title("Stage 2: MLP Training Loss")
    axes[1].grid(True)

    axes[2].plot(feedback_losses)
    axes[2].set_title("Stage 3: Brain->Feedback->Heart Loss")
    axes[2].grid(True)

    timesteps = np.linspace(0, t_duration, len(target_ecg))
    axes[3].plot(timesteps, target_ecg, label="Target ECG", linewidth=2)
    axes[3].plot(timesteps, predicted_ecg_baseline, label="Baseline ECG", linestyle="--")
    axes[3].plot(timesteps, predicted_ecg_feedback, label="Feedback ECG", linestyle=":")
    axes[3].set_title("ECG Prediction: Baseline vs Brain Feedback")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(t, target_eeg, label="Target EEG", linewidth=2)
    axes[4].plot(t, P_out_baseline, label="P_out baseline", alpha=0.7)
    axes[4].set_title("Brain Output: P_out vs Target")
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout()
    out_path = output_dir / f"full_feedback_result_idx{target_idx}.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()
