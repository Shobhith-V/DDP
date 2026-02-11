"""Signal preprocessing: bandpass, detrend, normalize."""

import numpy as np
from scipy.signal import butter, filtfilt, detrend


def preprocess_signal(
    signal: np.ndarray,
    fs: float = 1000,
    lowcut: float = 1.5,
    highcut: float = 20,
) -> np.ndarray:
    """Detrend, bandpass filter, z-score normalize."""
    detrended = detrend(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, detrended)
    normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-12)
    return normalized.astype(np.float64)
