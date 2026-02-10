"""
Signal preprocessing utilities.

These functions were originally defined in the notebook and are now
collectd here so that both ECG and EEG preprocessing are reusable.
"""

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, detrend


def preprocess_signal(
    signal: np.ndarray,
    fs: float = 1000.0,
    lowcut: float = 1.5,
    highcut: float = 20.0,
) -> np.ndarray:
    """
    Detrend, band‑pass filter and z‑score normalise a 1D signal.

    Parameters follow the original notebook:
    - detrend
    - 4th‑order Butterworth band‑pass between lowcut and highcut
    - standardisation to zero mean, unit variance
    """
    detrended = detrend(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, detrended)
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    return normalized


def preprocess_ecg_eeg(
    ecg_data: np.ndarray,
    eeg_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper that applies `preprocess_signal` to the ECG
    and to every EEG region, using the exact parameters from the notebook.
    """
    ecg_processed = preprocess_signal(
        ecg_data, fs=1000.0, lowcut=1.5, highcut=20.0
    )
    eeg_processed = np.array(
        [
            preprocess_signal(row, fs=1000.0, lowcut=0.5, highcut=30.0)
            for row in eeg_data
        ]
    )
    return ecg_processed, eeg_processed

