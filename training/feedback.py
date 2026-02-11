"""Stage 3: Differentiable feedback — HeartOscillatorTorch, time-varying modulation, NO reset_weights."""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dynamics.heart_oscillators_torch import HeartOscillatorTorch
from models.heart import HeartModel
from models.feedback import FeedbackMLP

logger = logging.getLogger(__name__)


def train_feedback_loop(
    trained_heart_model: HeartModel,
    rcos_phi: torch.Tensor,
    ecg_target_signal: np.ndarray,
    device: torch.device,
    T: float = 2.0,
    dt: float = 0.01,
    omega1_hz: float = 1.0,
    omega2_hz: float = 1.2,
    num_epochs: int = 5000,
    lr: float = 1e-3,
    log_interval: int = 500,
) -> tuple[HeartModel, FeedbackMLP, list]:
    """Train heart + feedback jointly. NO reset_weights. Fully differentiable."""
    logger.info("Stage 3: Brain->Feedback->Heart (differentiable)")

    trained_heart_model.train()
    feedback_mlp = FeedbackMLP().to(device)

    heart_osc_torch = HeartOscillatorTorch(
        omega1_hz=omega1_hz,
        omega2_hz=omega2_hz,
    ).to(device)

    optimizer = optim.Adam(
        list(trained_heart_model.parameters()) + list(feedback_mlp.parameters()),
        lr=lr,
    )
    criterion = nn.MSELoss()

    ecg_target = torch.tensor(
        ecg_target_signal[::10], dtype=torch.float32, device=device
    ).unsqueeze(1)

    rcos_phi_device = rcos_phi.to(device)
    if rcos_phi_device.requires_grad:
        rcos_phi_device = rcos_phi_device.detach()

    losses = []
    for epoch in range(num_epochs):
        modulation = feedback_mlp(rcos_phi_device.unsqueeze(-1))
        heart_traj = heart_osc_torch(T=T, dt=dt, modulation=modulation)
        predicted_ecg = trained_heart_model(heart_traj)
        loss = criterion(predicted_ecg, ecg_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Feedback Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    logger.info("Feedback training finished")
    return trained_heart_model, feedback_mlp, losses
