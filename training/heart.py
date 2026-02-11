"""Stage 0: Heart model pre-training."""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dynamics.heart_oscillators import simulate_coupled_oscillators_numpy
from models.heart import HeartModel

logger = logging.getLogger(__name__)


def train_heart_model(
    ecg_processed: np.ndarray,
    device: torch.device,
    omega1_hz: float = 1.0,
    omega2_hz: float = 1.2,
    dt: float = 0.01,
    T: float = 2.0,
    num_epochs: int = 25000,
    lr: float = 1e-3,
    log_interval: int = 2500,
) -> HeartModel:
    logger.info("Stage 0: Heart pre-training")
    heart_model = HeartModel().to(device)
    optimizer = optim.Adam(heart_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    sim_osc = simulate_coupled_oscillators_numpy(
        T=T, dt=dt, omega1_hz=omega1_hz, omega2_hz=omega2_hz
    )
    sim_osc_tensor = torch.tensor(sim_osc, dtype=torch.float32, device=device)
    ecg_target = torch.tensor(
        ecg_processed[::10], dtype=torch.float32, device=device
    ).unsqueeze(1)

    for epoch in range(num_epochs):
        heart_model.train()
        pred = heart_model(sim_osc_tensor)
        loss = criterion(pred, ecg_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Heart Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    logger.info("Heart pre-training finished")
    return heart_model
