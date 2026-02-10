"""
Data loading utilities for the brain–heart feedback project.

This module encapsulates all file I/O and data preparation that was
originally done inside the notebook.
"""

from typing import Tuple, List

import numpy as np
from scipy.io import loadmat
import mne
import sys

from config import (
    DEFAULT_ECG_FIF_PATH,
    DEFAULT_EEG_SCOUT_MAT_PATH,
    DEFAULT_SC_MAT_PATH,
)


def load_ecg_eeg_and_connectivity(
    ecg_fif_path: str = DEFAULT_ECG_FIF_PATH,
    eeg_mat_path: str = DEFAULT_EEG_SCOUT_MAT_PATH,
    sc_mat_path: str = DEFAULT_SC_MAT_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Load ECG, EEG and structural connectivity data.

    This follows the logic in the original notebook:
    - read a raw FIF file with MNE and extract one ECG-like channel
    - read a MATLAB file containing the regional EEG data
    - read a structural connectivity matrix and normalise it

    Returns
    -------
    ecg_data : np.ndarray
        1D array with ECG samples.
    eeg_data : np.ndarray
        2D array (regions x time) with EEG samples.
    Sw_all : np.ndarray
        Normalised structural connectivity matrix.
    non_zero_indices_per_row : list[np.ndarray]
        For each row in Sw_all, indices of non‑zero connections.
    """
    try:
        # ECG from raw FIF (channel index and sample range follow the notebook)
        raw = mne.io.read_raw_fif(ecg_fif_path, preload=False)
        data, times = raw[322, 2000:4000]
        ecg_data = -data[0]

        # EEG scout data
        mat = loadmat(eeg_mat_path)
        eeg_data = mat["Value"][:, 2000:4000]

        # Structural connectivity
        sc_data = loadmat(sc_mat_path)
        sc_matrix = sc_data["sc"]

        max_val = np.max(sc_matrix)
        Sw_all = (sc_matrix / max_val) * 0.01 if max_val > 0 else sc_matrix

    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        sys.exit(1)

    non_zero_indices_per_row = [
        np.nonzero(Sw_all[i, :])[0] for i in range(Sw_all.shape[0])
    ]

    return ecg_data, eeg_data, Sw_all, non_zero_indices_per_row

