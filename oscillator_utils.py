"""
Utility functions for coupled oscillator simulations and connectivity.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn


def simulate_coupled_oscillators(
    T: float = 10.0,
    dt: float = 1 / 1000,
    alpha: float = 1.0,
    omega1: float = 5.01,
    omega2: float = 5.1,
    A_init: float = 0.0001,
    theta_init: float = 3.14,
    n: float = 1.0,
    modulation: np.ndarray | None = None,
) -> np.ndarray:
    """
    Simulate two coupled Hopf‑like oscillators.

    This is a direct translation of `simulate_coupled_oscillators` from
    the notebook.
    """
    N = int(T / dt)
    r1, r2, phi1, phi2 = 1.0, 1.0, 0.0, 0.0
    A12, A21 = A_init, A_init
    theta12, theta21 = theta_init, theta_init

    R1, R2, Phi1, Phi2 = (
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
    )

    for i in range(N):
        R1[i], R2[i], Phi1[i], Phi2[i] = r1, r2, phi1, phi2

        coupling12 = A12 * r2 * np.cos(theta12 + n * (phi2 - phi1))
        coupling21 = A21 * r1 * np.cos(theta21 + n * (phi1 - phi2))

        m0 = (
            0.1 * modulation[i, 0]
            if modulation is not None and i < len(modulation)
            else 0.0
        )
        m1 = (
            0.1 * modulation[i, 1]
            if modulation is not None and i < len(modulation)
            else 0.0
        )

        dr1 = alpha * r1 - r1**3 + coupling12 + m0
        dr2 = alpha * r2 - r2**3 + coupling21 + m1

        dphi1 = omega1 + A12 * r2 / r1 * np.sin(theta12 + n * (phi2 - phi1))
        dphi2 = omega2 + A21 * r1 / r2 * np.sin(theta21 + n * (phi1 - phi2))

        r1 += dr1 * dt
        r2 += dr2 * dt
        phi1 += dphi1 * dt
        phi2 += dphi2 * dt

    return np.stack(
        (R1 * np.cos(Phi1), R1 * np.sin(Phi1), R2 * np.cos(Phi2), R2 * np.sin(Phi2)),
        axis=1,
    )


def get_random_frequencies(
    num_regions: int,
    osc_per_region: int,
    low: float = 1.0,
    high: float = 20.0,
    seed: int | None = None,
) -> np.ndarray:
    """Sample random oscillator frequencies in Hz and return rad/s."""
    if seed is not None:
        np.random.seed(seed)
    total_oscillators = num_regions * osc_per_region
    freqs_hz = np.random.uniform(low, high, total_oscillators)
    return 2 * np.pi * freqs_hz


def expand_structural_connectivity(
    Sc_region: np.ndarray,
    osc_per_region: int,
    intra_value: float = 0.0001,
    seed: int | None = None,
) -> np.ndarray:
    """
    Expand a region‑level structural connectivity matrix to oscillator level.

    Follows the original notebook implementation: each region is replaced
    by a block of oscillators and intra‑/inter‑regional weights are
    distributed accordingly.
    """
    if seed is not None:
        np.random.seed(seed)
    num_regions = Sc_region.shape[0]
    N = num_regions * osc_per_region
    Sc_full = np.zeros((N, N))
    for i in range(num_regions):
        for j in range(num_regions):
            start_i, end_i = i * osc_per_region, (i + 1) * osc_per_region
            start_j, end_j = j * osc_per_region, (j + 1) * osc_per_region
            if i == j:
                Sc_full[start_i:end_i, start_j:end_j] = intra_value
            else:
                rand_block = np.random.rand(osc_per_region, osc_per_region)
                rand_block *= Sc_region[i, j] / (rand_block.sum() + 1e-9)
                Sc_full[start_i:end_i, start_j:end_j] = rand_block
    np.fill_diagonal(Sc_full, 0.0)
    return Sc_full


def reset_weights(m: nn.Module) -> None:
    """
    If a module has `reset_parameters`, call it.

    This is used before re‑training the heart model inside the feedback loop.
    """
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

