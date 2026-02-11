"""
Training routines for the heart model, brain model and feedback loop.
"""

from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.interpolate import interp1d

from config import debug_interval
from models_heart import HeartModel, ECGToOscillatorMLP, FeedbackMLP
from models_brain_ode import TorchRevHopfNetwork
from oscillator_utils import (
    simulate_coupled_oscillators,
    get_random_frequencies,
    expand_structural_connectivity,
    reset_weights,
)


def train_heart_model(ecg_target_signal: np.ndarray, device: str) -> HeartModel:
    """
    Stage 0: pre‑train the heart model to map oscillator trajectories → ECG.
    """
    print("--- Starting Heart Model Pre-training ---")
    heart_model = HeartModel().to(device)
    optimizer = optim.Adam(heart_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    sim_osc_input = torch.tensor(
        simulate_coupled_oscillators(T=2, dt=0.01),
        dtype=torch.float32,
    ).to(device)
    ecg_target = torch.tensor(
        ecg_target_signal[::10], dtype=torch.float32
    ).to(device).unsqueeze(1)

    for epoch in range(25000):
        predicted_ecg = heart_model(sim_osc_input)
        loss = criterion(predicted_ecg, ecg_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2500 == 0:
            print(f"Heart Epoch {epoch+1}, Loss: {loss.item():.6f}")

    print("--- Heart Pre-training Finished ---")
    return heart_model


def pre_train_brain_model(
    eeg_processed: np.ndarray,
    Sw_all: np.ndarray,
    target_idx: int,
    non_zero_indices_per_row: List[np.ndarray],
    t: np.ndarray,
    D_function,
    device: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int, List[float]]:
    """
    Stage 1: pre‑train the brain reversed‑Hopf model without ECG feedback.
    """
    print("\n--- Stage 1: Brain Pre-training ---")
    connected_indices = np.unique(
        np.append(non_zero_indices_per_row[target_idx], target_idx)
    )
    N_reduced_regions = len(connected_indices)
    osc_per_region = 2
    N = N_reduced_regions * osc_per_region

    Sc_reduced_regional = Sw_all[np.ix_(connected_indices, connected_indices)]
    Sc_reduced_osc = expand_structural_connectivity(
        Sc_reduced_regional, osc_per_region, seed=42
    )

    omega_full = get_random_frequencies(
        68, osc_per_region, low=1, high=20, seed=42
    )
    alpha_full = np.random.uniform(0.1, 0.7, 68 * osc_per_region)
    omega0 = np.concatenate(
        [
            omega_full[i * osc_per_region : (i + 1) * osc_per_region]
            for i in connected_indices
        ]
    )
    alpha0 = np.clip(
        np.concatenate(
            [
                alpha_full[i * osc_per_region : (i + 1) * osc_per_region]
                for i in connected_indices
            ]
        ),
        0.05,
        0.5,
    )
    r0 = 0.1 * np.ones(N)
    phi0 = np.zeros(N)
    theta_random = np.pi * (2 * np.random.rand(N, N) - 1)
    theta0 = theta_random - theta_random.T

    model = TorchRevHopfNetwork(
        mu=1.0,
        eta_omega=0.05,
        eta_alpha=0.005,
        eta_theta=0.05,
        D_function=D_function,
        N=N,
        Sc=Sc_reduced_osc,
        mlp_model=None,
        hidden_repr=None,
        device=device,
    )

    criterion = nn.MSELoss()
    D_true = torch.tensor(D_function(t), device=device, dtype=torch.float32)
    losses: List[float] = []

    for epoch in range(250):
        with torch.no_grad():
            r, phi, theta, omega, alpha, _ = model.solve(
                r0, phi0, theta0, omega0, alpha0, t
            )
            P_out = torch.sum(alpha * r * torch.cos(phi), axis=1)
            loss = criterion(P_out, D_true)
            losses.append(loss.item())

            theta0, omega0 = (
                theta[-1].cpu().numpy(),
                omega[-1].cpu().numpy(),
            )
            alpha0 = alpha[-1].cpu().numpy()

        if (epoch + 1) % 10 == 0:
            print(f"Brain Epoch {epoch+1}/100, Loss: {loss.item():.6f}")

    final_params = {
        "r": r0,
        "phi": phi0,
        "theta": theta0,
        "omega": omega0,
        "alpha": alpha0,
    }
    return final_params, Sc_reduced_osc, N, losses


def train_mlp_on_frozen_brain(
    trained_heart_model: HeartModel,
    initial_brain_params: Dict[str, np.ndarray],
    Sc_reduced_osc: np.ndarray,
    N: int,
    D_function,
    t: np.ndarray,
    device: str,
) -> Tuple[ECGToOscillatorMLP, List[float]]:
    """
    Stage 2: Train an MLP that maps heart features to brain driving signal.
    """
    print("\n--- Stage 2: MLP Training ---")
    r0, phi0, theta0, omega0, alpha0 = (
        initial_brain_params["r"],
        initial_brain_params["phi"],
        initial_brain_params["theta"],
        initial_brain_params["omega"],
        initial_brain_params["alpha"],
    )

    trained_heart_model.eval()
    mlp_model = ECGToOscillatorMLP(output_dim=N).to(device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    fs = 1 / (t[1] - t[0])
    simulated_ecg_input = torch.tensor(
        simulate_coupled_oscillators(T=t[-1] + 1 / fs, dt=1 / fs),
        dtype=torch.float32,
    ).to(device)
    losses: List[float] = []

    for epoch in range(400):
        mlp_model.train()
        with torch.no_grad():
            hidden_repr = trained_heart_model.get_features(simulated_ecg_input)

        model = TorchRevHopfNetwork(
            mu=1.0,
            eta_omega=0.0,
            eta_alpha=0.0,
            eta_theta=0.0,
            D_function=D_function,
            N=N,
            Sc=Sc_reduced_osc,
            mlp_model=mlp_model,
            hidden_repr=hidden_repr,
            device=device,
        )

        r, phi, theta, omega, alpha, _ = model.solve(
            r0, phi0, theta0, omega0, alpha0, t
        )
        P_out = torch.sum(alpha * r * torch.cos(phi), axis=1)
        D_true = torch.tensor(
            D_function(t), device=device, dtype=torch.float32
        )
        loss = criterion(P_out, D_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"MLP Epoch {epoch+1}/400, Loss: {loss.item():.6f}")

    return mlp_model, losses


def train_feedback_loop(
    trained_heart_model: HeartModel,
    rcos_phi_final: np.ndarray,
    ecg_target_signal: np.ndarray,
    T: float = 2.0,
    dt: float = 0.01,
    device: str = "cuda",
    num_epochs: int = 10000,
):
    """
    Stage 3: Train heart with brain rcos_phi → FeedbackMLP → heart modulation.
    """
    print("--- Starting Brain→Feedback→Heart Training ---")

    trained_heart_model.apply(reset_weights)
    trained_heart_model.train()

    # FeedbackMLP: rcos_phi[time] → modulation[2]
    feedback_mlp = FeedbackMLP(input_dim=1, hidden_dim=64, output_dim=2).to(
        device
    )
    optimizer = optim.Adam(
        list(trained_heart_model.parameters())
        + list(feedback_mlp.parameters()),
        lr=1e-3,
    )
    criterion = nn.MSELoss()

    ecg_target = torch.tensor(
        ecg_target_signal[::10], dtype=torch.float32
    ).to(device).unsqueeze(1)
    rcos_phi_tensor = torch.tensor(
        rcos_phi_final, dtype=torch.float32
    ).to(device)

    losses: List[float] = []
    for epoch in range(num_epochs):
        # Brain rcos_phi → FeedbackMLP → modulation
        feedback_output = feedback_mlp(rcos_phi_tensor)  # [time, 2]
        modulation = (
            feedback_output.mean(dim=0).detach().cpu().numpy()
        )  # [2] mean modulation

        # Modulated heart simulation
        sim_osc_mod = simulate_coupled_oscillators(
            T=T, dt=dt, modulation=np.tile(modulation, (int(T / dt), 1))
        )
        sim_osc_tensor = torch.tensor(
            sim_osc_mod, dtype=torch.float32
        ).to(device)

        # Predict ECG
        predicted_ecg = trained_heart_model(sim_osc_tensor)
        loss = criterion(predicted_ecg, ecg_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % debug_interval == 0:
            print(
                f"Feedback Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}"
            )

    print("--- Feedback Training Finished ---")
    return trained_heart_model, feedback_mlp, losses


def make_target_function_from_eeg(
    eeg_processed: np.ndarray,
    target_idx: int,
    t: np.ndarray,
):
    """
    Helper to build the target driving function D(t) from a processed EEG trace.
    """
    target_signal = eeg_processed[target_idx, ::10]
    D_function = interp1d(
        t,
        target_signal,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return target_signal, D_function

