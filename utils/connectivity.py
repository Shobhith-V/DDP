"""Structural connectivity expansion and frequency sampling."""

import numpy as np


def get_random_frequencies(
    num_regions: int,
    osc_per_region: int,
    low_hz: float = 1.0,
    high_hz: float = 20.0,
    seed: int | None = None,
) -> np.ndarray:
    """Return omega in rad/s from uniform Hz range."""
    if seed is not None:
        np.random.seed(seed)
    total = num_regions * osc_per_region
    freqs_hz = np.random.uniform(low_hz, high_hz, total)
    return 2 * np.pi * freqs_hz


def expand_structural_connectivity(
    Sc_region: np.ndarray,
    osc_per_region: int,
    intra_value: float = 0.0001,
    seed: int | None = None,
) -> np.ndarray:
    """Expand region-level SC to oscillator-level."""
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
