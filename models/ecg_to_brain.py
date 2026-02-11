"""ECG features -> OscillatorLayer -> Brain drive."""

import torch
import torch.nn as nn

from .oscillator_layer import OscillatorLayer


class ECGToOscillatorMLP(nn.Module):
    """ECG features -> pre_osc -> OscillatorLayer -> post_osc -> brain_drive [N]"""

    def __init__(
        self,
        ecg_dim: int = 50,
        N_VNS: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 16,
        T: float = 2.0,
        fs: float = 100,
        coupling_sparsity: float = 0.3,
        coupling_strength: float = 0.05,
        freq_hz_min: float = 2.0,
        freq_hz_max: float = 10.0,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.pre_osc = nn.Sequential(
            nn.Linear(ecg_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, N_VNS * 2),
        )
        self.osc_layer = OscillatorLayer(
            N_osc=N_VNS,
            T=T,
            fs=fs,
            coupling_sparsity=coupling_sparsity,
            coupling_strength=coupling_strength,
            freq_hz_min=freq_hz_min,
            freq_hz_max=freq_hz_max,
            seed=seed,
            device=device,
        )
        self.post_osc = nn.Sequential(
            nn.Linear(N_VNS * 2, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, ecg_features: torch.Tensor) -> torch.Tensor:
        """ecg_features: (batch, ecg_dim) or (T, ecg_dim). Returns (batch, output_dim)."""
        if ecg_features.dim() == 1:
            ecg_features = ecg_features.unsqueeze(0)
        pre = self.pre_osc(ecg_features)
        amp_input = pre[:, : pre.shape[-1] // 2]
        phase_input = pre[:, pre.shape[-1] // 2 :]
        osc_out = self.osc_layer(amp_input, phase_input)
        brain_drive = self.post_osc(osc_out)
        if brain_drive.shape[0] == 1:
            return brain_drive.squeeze(0)
        return brain_drive
