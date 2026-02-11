"""Stage 2: ECG -> OscillatorLayer -> Brain (MLP training, brain frozen)."""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dynamics.brain_ode import TorchRevHopfNetwork
from dynamics.heart_oscillators import simulate_coupled_oscillators_numpy
from models.ecg_to_brain import ECGToOscillatorMLP

logger = logging.getLogger(__name__)


def train_mlp_on_frozen_brain(
    trained_heart_model,
    initial_brain_params: dict,
    Sc_reduced_osc: np.ndarray,
    N: int,
    target_signal: np.ndarray,
    t: np.ndarray,
    device: torch.device,
    ecg_dim: int = 50,
    n_vns: int = 64,
    hidden_dim: int = 64,
    T: float = 2.0,
    dt: float = 0.01,
    omega1_hz: float = 1.0,
    omega2_hz: float = 1.2,
    num_epochs: int = 100,
    lr: float = 1e-2,
    log_interval: int = 20,
) -> tuple[ECGToOscillatorMLP, list]:
    logger.info("Stage 2: ECG -> OscillatorLayer -> Brain")

    ecg_to_osc_mlp = ECGToOscillatorMLP(
        ecg_dim=ecg_dim,
        N_VNS=n_vns,
        hidden_dim=hidden_dim,
        output_dim=N,
        T=T,
        device=device,
    ).to(device)

    optimizer = optim.Adam(ecg_to_osc_mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    with torch.no_grad():
        sim_osc = simulate_coupled_oscillators_numpy(
            T=T, dt=dt, omega1_hz=omega1_hz, omega2_hz=omega2_hz
        )
        sim_osc_t = torch.tensor(sim_osc, dtype=torch.float32, device=device)
        ecg_features = trained_heart_model.get_features(sim_osc_t)

    D_tensor = torch.tensor(target_signal, dtype=torch.float32, device=device)
    t_eval = torch.tensor(t, dtype=torch.float32, device=device)

    losses = []
    for epoch in range(num_epochs):
        model = TorchRevHopfNetwork(
            mu=1.0,
            eta_omega=0.0,
            eta_alpha=0.0,
            eta_theta=0.0,
            D_tensor=D_tensor,
            t_eval=t_eval,
            N=N,
            Sc=Sc_reduced_osc,
            mlp_model=ecg_to_osc_mlp,
            hidden_repr=ecg_features,
            device=device,
        )

        r, phi, theta, omega, alpha, _ = model.solve(
            initial_brain_params["r"],
            initial_brain_params["phi"],
            initial_brain_params["theta"],
            initial_brain_params["omega"],
            initial_brain_params["alpha"],
        )

        P_out = torch.sum(alpha * r * torch.cos(phi), dim=1)
        loss = criterion(P_out, D_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % log_interval == 0:
            logger.info(f"ECG->Brain Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    return ecg_to_osc_mlp, losses
