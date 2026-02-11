"""Stage 1: Brain pre-training (Hebbian plasticity, no gradients)."""

import logging
import numpy as np
import torch
import torch.nn as nn

from dynamics.brain_ode import TorchRevHopfNetwork
from utils.connectivity import expand_structural_connectivity, get_random_frequencies

logger = logging.getLogger(__name__)


def pre_train_brain_model(
    eeg_processed: np.ndarray,
    Sw_all: np.ndarray,
    non_zero_indices_per_row: list,
    target_idx: int,
    t: np.ndarray,
    target_signal: np.ndarray,
    device: torch.device,
    osc_per_region: int = 3,
    eta_omega: float = 0.05,
    eta_alpha: float = 0.005,
    eta_theta: float = 0.05,
    num_epochs: int = 30,
    seed: int = 42,
) -> tuple[dict, np.ndarray, int, list]:
    logger.info("Stage 1: Brain pre-training")

    connected = np.unique(np.append(non_zero_indices_per_row[target_idx], target_idx))
    N_regions = len(connected)
    N = N_regions * osc_per_region

    Sc_reduced = Sw_all[np.ix_(connected, connected)]
    Sc_osc = expand_structural_connectivity(Sc_reduced, osc_per_region, seed=seed)

    omega_full = get_random_frequencies(68, osc_per_region, 1, 20, seed)
    alpha_full = np.random.uniform(0.1, 0.7, 68 * osc_per_region)
    omega0 = np.concatenate([omega_full[i * osc_per_region : (i + 1) * osc_per_region] for i in connected])
    alpha0 = np.clip(
        np.concatenate([alpha_full[i * osc_per_region : (i + 1) * osc_per_region] for i in connected]),
        0.05,
        0.5,
    )
    r0 = 0.1 * np.ones(N)
    phi0 = np.zeros(N)
    theta_random = np.pi * (2 * np.random.rand(N, N) - 1)
    theta0 = theta_random - theta_random.T

    D_tensor = torch.tensor(target_signal, dtype=torch.float32, device=device)
    t_eval = torch.tensor(t, dtype=torch.float32, device=device)

    model = TorchRevHopfNetwork(
        mu=1.0,
        eta_omega=eta_omega,
        eta_alpha=eta_alpha,
        eta_theta=eta_theta,
        D_tensor=D_tensor,
        t_eval=t_eval,
        N=N,
        Sc=Sc_osc,
        mlp_model=None,
        hidden_repr=None,
        device=device,
    )

    criterion = nn.MSELoss()
    losses = []

    for epoch in range(num_epochs):
        with torch.no_grad():
            r, phi, theta, omega, alpha, _ = model.solve(r0, phi0, theta0, omega0, alpha0)
            P_out = torch.sum(alpha * r * torch.cos(phi), dim=1)
            loss = criterion(P_out, D_tensor)
            losses.append(loss.item())

            theta0 = theta[-1].cpu().numpy()
            omega0 = omega[-1].cpu().numpy()
            alpha0 = alpha[-1].cpu().numpy()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Brain Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    final_params = {
        "r": r0,
        "phi": phi0,
        "theta": theta0,
        "omega": omega0,
        "alpha": alpha0,
    }
    return final_params, Sc_osc, N, losses
