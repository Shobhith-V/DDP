"""Data loading: ECG, EEG, structural connectivity."""

import os
from pathlib import Path

import numpy as np
import mne
from scipy.io import loadmat


def _resolve_path(base_dir: Path, path: str) -> Path:
    """Resolve path relative to project root or absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return p


def load_ecg(
    fif_path: str,
    channel: int = 322,
    t_start: int = 2000,
    t_end: int = 4000,
    negate: bool = True,
    base_dir: Path | None = None,
) -> np.ndarray:
    """Load ECG from MNE FIF file."""
    base_dir = base_dir or Path.cwd()
    path = _resolve_path(base_dir, fif_path)
    raw = mne.io.read_raw_fif(str(path), preload=False)
    data, _ = raw[channel, t_start:t_end]
    ecg = -data[0] if negate else data[0]
    return ecg.astype(np.float64)


def load_eeg(
    mat_path: str,
    t_start: int = 2000,
    t_end: int = 4000,
    base_dir: Path | None = None,
) -> np.ndarray:
    """Load EEG from scout MAT file."""
    base_dir = base_dir or Path.cwd()
    path = _resolve_path(base_dir, mat_path)
    mat = loadmat(str(path))
    return mat["Value"][:, t_start:t_end].astype(np.float64)


def load_sc(
    mat_path: str,
    scale: float = 0.01,
    base_dir: Path | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load structural connectivity, normalize, return (Sc, non_zero_indices_per_row)."""
    base_dir = base_dir or Path.cwd()
    path = _resolve_path(base_dir, mat_path)
    sc_data = loadmat(str(path))
    sc_matrix = sc_data["sc"].astype(np.float64)
    max_val = np.max(sc_matrix)
    Sw = (sc_matrix / max_val) * scale if max_val > 0 else sc_matrix
    non_zero = [np.nonzero(Sw[i, :])[0] for i in range(Sw.shape[0])]
    return Sw, non_zero


def load_all_data(
    ecg_path: str,
    eeg_path: str,
    sc_path: str,
    ecg_channel: int = 322,
    t_start: int = 2000,
    t_end: int = 4000,
    ecg_negate: bool = True,
    scale: float = 0.01,
    base_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Load ECG, EEG, SC, and non-zero indices."""
    ecg = load_ecg(ecg_path, ecg_channel, t_start, t_end, ecg_negate, base_dir)
    eeg = load_eeg(eeg_path, t_start, t_end, base_dir)
    sc, non_zero = load_sc(sc_path, scale, base_dir)
    return ecg, eeg, sc, non_zero
