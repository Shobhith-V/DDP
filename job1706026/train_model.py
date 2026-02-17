"""
Minimal mechanistic heart–brain model with stateful oscillators.

Goal:
- Prioritise a coherent dynamical system over architectural complexity.
- No BL blocks, no top oscillator layer, no fake feedback using ground‑truth EEG.
- Discrete‑time, stateful oscillators (no ODE solver).

Pipeline (per sequence):
1) Heart Hopf oscillators (2 units) generate latent cardiac state.
2) Linear/MLP readout from heart state → model ECG (ECG_hat).
3) ECG_hat drives NTS Hopf network (N units, coupled, stateful).
4) Linear readout from NTS state → MEG/EEG_hat (subset of channels).

Loss:
    L = MSE(ECG_hat, ECG_true) + λ * MSE(MEG_hat, MEG_true)

Oscillators are inspired by your earlier Hopf code and donn.py, but:
- States (r, phi) persist across time within a sequence.
- No ground‑truth brain state is used as an input.
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, detrend
import mne

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def load_ecg_eeg(
    ecg_path: str,
    eeg_path: str,
    ecg_chan: int = 322,
    t_start: int = 2000,
    t_end: int = 4000,
):
    """
    Load ECG from MEG file and EEG from .mat.

    Returns:
        ecg_raw: 1D numpy array
        eeg_raw: 2D numpy array (n_channels, T)
    """
    raw = mne.io.read_raw_fif(ecg_path, preload=True)
    data, times = raw[ecg_chan, t_start:t_end]
    ecg_raw = -data[0]

    mat = loadmat(eeg_path)
    eeg_raw = mat["Value"][:, t_start:t_end]

    return ecg_raw, eeg_raw


def bandpass_normalize(signal, fs, lowcut, highcut):
    """Detrend, band‑pass filter and z‑score a 1D numpy signal."""
    x = detrend(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype="band")
    xf = filtfilt(b, a, x)
    xf = (xf - np.mean(xf)) / (np.std(xf) + 1e-8)
    return xf


# =============================================================================
# Stateful discrete‑time Hopf oscillators
# =============================================================================

class HeartHopf(nn.Module):
    """
    Two coupled Hopf oscillators modelling cardiac dynamics.

    Dynamics (for each oscillator i):
        dr_i/dt   = α r_i - r_i^3 + coupling + ...
        dφ_i/dt   = ω_i + coupling_phase

    Implemented as simple Euler steps with persistent state across time.
    """

    def __init__(
        self,
        dt: float = 0.01,
        alpha: float = 1.0,
        omega1: float = 5.0,
        omega2: float = 5.1,
        A_init: float = 1e-4,
        theta_init: float = 3.14,
        n: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.dt = dt
        self.alpha = alpha
        self.register_buffer("omega", torch.tensor([omega1, omega2], dtype=torch.float32))
        self.register_buffer("A", torch.tensor([[0.0, A_init], [A_init, 0.0]], dtype=torch.float32))
        self.register_buffer("theta", torch.tensor([[0.0, theta_init], [theta_init, 0.0]], dtype=torch.float32))
        self.n = n

        # Persistent state (initialised via reset_state)
        self.r_state = None  # (batch, 2)
        self.phi_state = None  # (batch, 2)
        self.to(device)

    def reset_state(self, batch_size: int, r0: float = 1.0, phi0: float = 0.0, device=None):
        """Initialise oscillator state for a new sequence."""
        if device is None:
            device = next(self.parameters()).device
        self.r_state = torch.full((batch_size, 2), r0, dtype=torch.float32, device=device)
        self.phi_state = torch.full((batch_size, 2), phi0, dtype=torch.float32, device=device)

    def step(self):
        """
        Advance heart dynamics by one dt.

        Returns:
            state_t: (batch, 4) = [r1 cos φ1, r1 sin φ1, r2 cos φ2, r2 sin φ2]
        """
        r = self.r_state  # (B, 2)
        phi = self.phi_state  # (B, 2)

        # Broadcasting over batch
        # Pairwise phase differences (2x2) shared across batch
        phi_diff = phi.unsqueeze(2) - phi.unsqueeze(1)  # (B, 2, 2)

        # Coupling terms
        coupling = (self.A * torch.cos(self.theta + self.n * phi_diff)).sum(dim=2)  # (B, 2)

        dr = self.alpha * r - r ** 3 + coupling
        dphi = self.omega.unsqueeze(0) + (self.A * torch.sin(self.theta + self.n * phi_diff)).sum(
            dim=2
        ) / (r + 1e-6)

        r = r + dr * self.dt
        phi = phi + dphi * self.dt

        self.r_state = r
        self.phi_state = phi

        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        state = torch.cat([x, y], dim=-1)  # (B, 4)
        return state


class NTShopfNetwork(nn.Module):
    """
    N‑unit coupled Hopf network driven by ECG.

    Each oscillator i:
        dr_i/dt = (μ - r_i^2) r_i + drive + coupling_r
        dφ_i/dt = ω_i + coupling_φ

    Input drive is a scalar ECG(t) broadcast to all units.
    States persist across time for each sequence.
    """

    def __init__(
        self,
        N: int = 32,
        dt: float = 0.01,
        mu: float = 1.0,
        min_freq: float = 1.0,
        max_freq: float = 20.0,
        coupling_strength: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__()
        self.N = N
        self.dt = dt
        self.mu = mu
        self.coupling_strength = coupling_strength

        # Learnable frequencies (Hz, then converted to rad/s)
        self.omega_raw = nn.Parameter(torch.randn(1, N))

        # Learnable symmetric coupling matrix with zero diagonal
        C = torch.randn(N, N) * 0.01
        C = 0.5 * (C + C.T)
        C.fill_diagonal_(0.0)
        self.C = nn.Parameter(C)

        self.min_freq = min_freq
        self.max_freq = max_freq

        # Persistent state
        self.r_state = None  # (batch, N)
        self.phi_state = None  # (batch, N)

        self.to(device)

    def reset_state(self, batch_size: int, r0: float = 0.1, phi0: float = 0.0, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.r_state = torch.full((batch_size, self.N), r0, dtype=torch.float32, device=device)
        self.phi_state = torch.full((batch_size, self.N), phi0, dtype=torch.float32, device=device)

    def _frequencies(self, batch_size: int):
        """Map omega_raw → [min_freq, max_freq] Hz → rad/s, broadcast over batch."""
        freq_hz = torch.sigmoid(self.omega_raw) * (self.max_freq - self.min_freq) + self.min_freq
        omega = 2 * np.pi * freq_hz  # (1, N)
        return omega.repeat(batch_size, 1)  # (B, N)

    def step(self, ecg_t: torch.Tensor):
        """
        Advance network by one dt.

        Args:
            ecg_t: (batch, 1) scalar ECG at time t

        Returns:
            state_t: (batch, 2N) = [r cos φ, r sin φ]
        """
        r = self.r_state  # (B, N)
        phi = self.phi_state  # (B, N)
        B, N = r.shape

        omega = self._frequencies(B)  # (B, N)

        # Coupling via C
        # Use r * exp(i phi) representation
        x = r * torch.cos(phi)  # (B, N)
        y = r * torch.sin(phi)  # (B, N)

        # Linear mixing across units
        x_c = torch.matmul(x, self.C.T)  # (B, N)
        y_c = torch.matmul(y, self.C.T)  # (B, N)

        # Convert coupling back to radial/phase terms approximately
        r_c = (x_c * torch.cos(phi) + y_c * torch.sin(phi))  # projection on radial direction
        phi_c = (-x_c * torch.sin(phi) + y_c * torch.cos(phi)) / (r + 1e-6)  # tangential component

        # Drive from ECG (broadcast)
        drive = ecg_t  # (B, 1)
        drive = drive.expand(-1, N)  # (B, N)

        dr = (self.mu - r ** 2) * r + self.coupling_strength * r_c + 0.1 * drive
        dphi = omega + self.coupling_strength * phi_c

        r = r + dr * self.dt
        phi = phi + dphi * self.dt

        self.r_state = r
        self.phi_state = phi

        x_new = r * torch.cos(phi)
        y_new = r * torch.sin(phi)
        state = torch.cat([x_new, y_new], dim=-1)  # (B, 2N)
        return state


# =============================================================================
# Readout heads and full model
# =============================================================================

class HeartToECG(nn.Module):
    """Small MLP from heart state (4D) to scalar ECG."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, h):
        return self.net(h)


class NTSToMEG(nn.Module):
    """Linear readout from NTS oscillator state (2N) to K MEG/EEG channels."""

    def __init__(self, nts_dim: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nts_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_out),
        )

    def forward(self, z):
        return self.net(z)


class HeartBrainModel(nn.Module):
    """
    Full stateful heart → ECG → NTS → MEG model.

    Usage per sequence:
        model.reset_states(batch_size)
        for t in range(T):
            ecg_hat_t, meg_hat_t = model.step()
    """

    def __init__(self, nts_units: int = 32, meg_channels: int = 5, dt: float = 0.01, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.dt = dt

        self.heart = HeartHopf(dt=dt, device=device)
        self.heart_head = HeartToECG()

        self.nts = NTShopfNetwork(N=nts_units, dt=dt, device=device)
        self.nts_head = NTSToMEG(nts_dim=2 * nts_units, n_out=meg_channels)

        self.to(device)

    def reset_states(self, batch_size: int = 1):
        dev = self.device
        self.heart.reset_state(batch_size=batch_size, device=dev)
        self.nts.reset_state(batch_size=batch_size, device=dev)

    def step(self):
        """
        Advance entire system by one dt.

        Returns:
            ecg_hat_t: (batch, 1)
            meg_hat_t: (batch, K)
        """
        # Heart dynamics and ECG readout
        heart_state = self.heart.step()  # (B, 4)
        ecg_hat = self.heart_head(heart_state)  # (B, 1)

        # NTS dynamics driven by model ECG
        nts_state = self.nts.step(ecg_hat.detach())  # detach to keep heart/nts gradients separable if desired
        meg_hat = self.nts_head(nts_state)  # (B, K)

        return ecg_hat, meg_hat


# =============================================================================
# Training loop
# =============================================================================

def train_sequence_model(
    model: HeartBrainModel,
    ecg_target: torch.Tensor,
    meg_target: torch.Tensor,
    dt: float,
    num_epochs: int = 500,
    lr: float = 1e-3,
    lambda_meg: float = 1.0,
    print_every: int = 50,
):
    """
    Train on a single sequence (batch_size = 1) for now.

    Args:
        ecg_target: (T,) numpy or (T,) torch → converted to (1, T, 1)
        meg_target: (K, T) numpy or (K, T) torch → converted to (1, T, K)
    """
    device = model.device
    if not torch.is_tensor(ecg_target):
        ecg_target = torch.tensor(ecg_target, dtype=torch.float32, device=device)
    if not torch.is_tensor(meg_target):
        meg_target = torch.tensor(meg_target, dtype=torch.float32, device=device)

    # Shapes: (T, 1), (T, K)
    if ecg_target.dim() == 1:
        ecg_target = ecg_target.unsqueeze(-1)
    if meg_target.dim() == 2:
        meg_target = meg_target.transpose(0, 1)  # (T, K) already, just ensure

    T = ecg_target.shape[0]
    K = meg_target.shape[1]

    # Add batch dimension
    ecg_target = ecg_target.unsqueeze(0)  # (1, T, 1)
    meg_target = meg_target.unsqueeze(0)  # (1, T, K)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        model.reset_states(batch_size=1)

        ecg_preds = []
        meg_preds = []

        for t in range(T):
            ecg_hat_t, meg_hat_t = model.step()
            ecg_preds.append(ecg_hat_t)  # (1, 1)
            meg_preds.append(meg_hat_t)  # (1, K)

        ecg_preds = torch.stack(ecg_preds, dim=1)  # (1, T, 1)
        meg_preds = torch.stack(meg_preds, dim=1)  # (1, T, K)

        ecg_loss = mse(ecg_preds, ecg_target)
        meg_loss = mse(meg_preds, meg_target)
        loss = ecg_loss + lambda_meg * meg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Total: {loss.item():.6f} | "
                f"ECG: {ecg_loss.item():.6f} | "
                f"MEG: {meg_loss.item():.6f}"
            )

    return model


# =============================================================================
# Main script
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths – adjust if needed
    ecg_file_path = "/home/shobs/Desktop/DDP/transdef_mf2pt2_rest_raw.fif"
    eeg_file_path = "/home/shobs/Desktop/DDP/scout_id_309.mat"

    # -------------------------------------------------------------------------
    # Load and preprocess
    # -------------------------------------------------------------------------
    ecg_raw, eeg_raw = load_ecg_eeg(ecg_file_path, eeg_file_path)

    fs_raw = 1000  # from your notebook
    ecg_proc = bandpass_normalize(ecg_raw, fs=fs_raw, lowcut=1.5, highcut=20.0)

    # Preprocess EEG similarly (0.5–30 Hz for all channels)
    eeg_proc = np.array(
        [bandpass_normalize(ch, fs=fs_raw, lowcut=0.5, highcut=30.0) for ch in eeg_raw]
    )  # (n_channels, T_raw)

    # -------------------------------------------------------------------------
    # Downsample to 100 Hz by simple decimation (1:10)
    # -------------------------------------------------------------------------
    ds_factor = 10
    ecg_ds = ecg_proc[::ds_factor]  # (T,)
    eeg_ds = eeg_proc[:, ::ds_factor]  # (n_channels, T)

    fs_ds = fs_raw // ds_factor  # 100 Hz
    dt = 1.0 / fs_ds

    # Use a manageable window (e.g. first 2 seconds)
    T_window = int(2.0 * fs_ds)
    ecg_target = ecg_ds[:T_window]  # (T,)

    # Choose first K channels as MEG proxy
    K = 5
    meg_target = eeg_ds[:K, :T_window]  # (K, T)

    print(f"T_window = {T_window}, fs_ds = {fs_ds}")
    print(f"ecg_target shape: {ecg_target.shape}")
    print(f"meg_target shape: {meg_target.shape}")

    # -------------------------------------------------------------------------
    # Build and train model
    # -------------------------------------------------------------------------
    model = HeartBrainModel(nts_units=32, meg_channels=K, dt=dt, device=device)
    model.to(device)

    train_sequence_model(
        model,
        ecg_target=ecg_target,
        meg_target=meg_target,
        dt=dt,
        num_epochs=500,
        lr=1e-3,
        lambda_meg=1.0,
        print_every=50,
    )

    print("Training complete.")


if __name__ == "__main__":
    main()

"""
ECG-to-Brain Neural Network with Oscillator Layers
Architecture inspired by the New Architecture diagram with:
- Feedback Intermediate Layers (BL1-BL4) with oscillators in BL4
- Heart Model with fundamental oscillators
- ECG-to-Brain pathway (HL1-HL2) with NTS oscillator layer
- Top oscillator layer
- Outputs: ECG and MEG

Oscillator layers inspired by donn.py (ResHopf style)
Optimized for computational efficiency.

COMPUTATIONAL OPTIMIZATIONS:
1. Single-step oscillator updates instead of full ODE integration
2. Efficient batch processing throughout
3. Reduced number of internal integration steps (num_steps parameter)
4. Simplified coupling calculations
5. No ODE solver calls - direct Euler integration

FREQUENCY RANGES:
- VN oscillators (NTS): 2-10 Hz (vagal nerve typical range)
- BL4 oscillators: 2-10 Hz
- Top oscillators: 2-10 Hz
- Fundamental heart oscillators: 5.0-5.2 Hz (heart rate range)

To reduce computational expense further:
- Reduce num_steps in CoupledOscillatorLayer (default: 10)
- Reduce batch size
- Use fewer oscillator units
- Consider using half precision (float16)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
import mne
import os
import time
from typing import Tuple, Optional


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data(ecg_path, eeg_path, sc_path):
    """Load ECG, EEG, and structural connectivity data."""
    raw = mne.io.read_raw_fif(ecg_path, preload=True)
    data, times = raw[322, 2000:4000]
    ecg_data = -data[0]
    
    mat = loadmat(eeg_path)
    eeg_data = mat['Value'][:, 2000:4000]
    
    sc_data = loadmat(sc_path)
    sc_matrix = sc_data['sc']
    max_val = np.max(sc_matrix)
    Sw_all = (sc_matrix / max_val) * 0.01 if max_val > 0 else sc_matrix
    
    return ecg_data, eeg_data, Sw_all


def preprocess_signal(signal, fs=1000, lowcut=1.5, highcut=20):
    """Preprocess signal with detrending, filtering, and normalization."""
    detrended = detrend(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, detrended)
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    return normalized


def heart_osc(T, dt, alpha=1, omega1=5.01, omega2=5.1, A_init=0.0001, 
              theta_init=3.14, n=1, modulation=None):
    """Simulate coupled heart oscillators."""
    N = int(T / dt)
    r1, r2, phi1, phi2 = 1.0, 1.0, 0.0, 0.0
    A12, A21 = A_init, A_init
    theta12, theta21 = theta_init, theta_init
    
    R1, R2, Phi1, Phi2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    
    for i in range(N):
        R1[i], R2[i], Phi1[i], Phi2[i] = r1, r2, phi1, phi2
        
        coupling12 = A12 * r2 * np.cos(theta12 + n * (phi2 - phi1))
        coupling21 = A21 * r1 * np.cos(theta21 + n * (phi1 - phi2))
        
        mod1 = 0.1 * modulation[i, 0] if modulation is not None and i < len(modulation) else 0
        mod2 = 0.1 * modulation[i, 1] if modulation is not None and i < len(modulation) else 0
        
        dr1 = alpha * r1 - r1**3 + coupling12 + mod1
        dr2 = alpha * r2 - r2**3 + coupling21 + mod2
        
        dphi1 = omega1 + A12 * r2 / r1 * np.sin(theta12 + n * (phi2 - phi1))
        dphi2 = omega2 + A21 * r1 / r2 * np.sin(theta21 + n * (phi1 - phi2))
        
        r1 += dr1 * dt
        r2 += dr2 * dt
        phi1 += dphi1 * dt
        phi2 += dphi2 * dt
    
    return np.stack((R1*np.cos(Phi1), R1*np.sin(Phi1), R2*np.cos(Phi2), R2*np.sin(Phi2)), axis=1)


# ============================================================================
# OSCILLATOR LAYERS (Inspired by donn.py)
# ============================================================================

class EfficientResHopf(nn.Module):
    """
    Efficient Residual Hopf Oscillator Layer
    Inspired by ResHopf from donn.py, optimized for batch processing.
    Uses single-step updates for efficiency.
    """
    
    def __init__(self, units, min_omega=2.0, max_omega=10.0, dt=0.01,
                 train_omegas=True, input_scaler=2.0, device="cpu", num_steps=1):
        super().__init__()
        
        self.units = units
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.dt = dt
        self.num_steps = num_steps  # Number of internal integration steps
        self.mu0 = 1.0
        self.beta1 = 1.0
        self.input_scaler = input_scaler
        self.train_omegas = train_omegas
        
        # Learnable frequencies
        if train_omegas:
            self.omegas = nn.Parameter(torch.randn(1, units))
        else:
            self.register_buffer('omegas', torch.rand(1, units))
        
        # Initial states (buffers, not parameters)
        self.register_buffer('initial_r', torch.ones(units) * 0.1)
        self.register_buffer('initial_phi', torch.zeros(units))
        
        self.to(device)
    
    def forward(self, X_real, X_imag=None, r=None, phi=None):
        """
        Forward pass with efficient single-step update.
        
        Args:
            X_real: Real part of input (batch_size, units)
            X_imag: Imaginary part (optional, defaults to zeros)
            r: Initial radius (optional)
            phi: Initial phase (optional)
        
        Returns:
            z_complex: Complex output (batch_size, units)
            r_final: Final radius
            phi_final: Final phase
        """
        batch_size = X_real.shape[0]
        
        if X_imag is None:
            X_imag = torch.zeros_like(X_real)
        
        # Initialize states
        if r is None:
            r = self.initial_r.unsqueeze(0).repeat(batch_size, 1)
        if phi is None:
            phi = self.initial_phi.unsqueeze(0).repeat(batch_size, 1)
        
        # Compute frequencies
        omega_intl = self.max_omega - self.min_omega
        omegas = torch.sigmoid(self.omegas) * omega_intl + self.min_omega
        omegas = omegas * (2 * np.pi)  # Convert to rad/s
        omegas = omegas.repeat(batch_size, 1)  # (B, units)
        
        # Run num_steps integration steps
        for _ in range(self.num_steps):
            # Input forcing (inspired by ResHopf from donn.py)
            input_r = self.input_scaler * X_real * torch.cos(phi)
            input_phi = self.input_scaler * X_imag * torch.sin(phi)
            
            # Hopf dynamics (matching donn.py ResHopf)
            dr_dt = (self.mu0 - self.beta1 * (r**2)) * r + input_r
            dphi_dt = omegas - input_phi
            
            r = r + dr_dt * self.dt
            phi = phi + dphi_dt * self.dt
        
        # Convert to complex
        z_real = r * torch.cos(phi)
        z_imag = r * torch.sin(phi)
        z_complex = torch.complex(z_real, z_imag)
        
        return z_complex, r, phi


class CoupledOscillatorLayer(nn.Module):
    """
    Coupled oscillator layer for BL4 and NTS layers.
    Uses efficient batch processing with coupling matrix.
    Inspired by donn.py but with coupling between oscillators.
    """
    
    def __init__(self, units, min_omega=2.0, max_omega=10.0, dt=0.01,
                 coupling_strength=0.05, train_omegas=True, device="cpu", num_steps=10):
        super().__init__()
        
        self.units = units
        self.dt = dt
        self.num_steps = num_steps
        self.coupling_strength = coupling_strength
        
        # Base oscillator parameters
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.mu0 = 1.0
        self.beta1 = 1.0
        self.input_scaler = 2.0
        self.train_omegas = train_omegas
        
        # Learnable frequencies
        if train_omegas:
            self.omegas = nn.Parameter(torch.randn(1, units))
        else:
            self.register_buffer('omegas', torch.rand(1, units))
        
        # Initial states
        self.register_buffer('initial_r', torch.ones(units) * 0.1)
        self.register_buffer('initial_phi', torch.zeros(units))
        
        # Coupling matrix (learnable)
        self.coupling_matrix = nn.Parameter(
            torch.randn(units, units) * 0.01
        )
        # Make it symmetric and zero diagonal
        with torch.no_grad():
            self.coupling_matrix.data = (self.coupling_matrix.data + self.coupling_matrix.data.T) / 2
            self.coupling_matrix.data.fill_diagonal_(0.0)
        
        self.to(device)
    
    def forward(self, X_real, X_imag=None):
        """
        Forward pass with coupling.
        
        Args:
            X_real: Input real part (batch_size, units)
            X_imag: Input imaginary part (optional)
        
        Returns:
            z_complex: Complex output (batch_size, units)
            r: Final radius
            phi: Final phase
        """
        batch_size = X_real.shape[0]
        
        if X_imag is None:
            X_imag = torch.zeros_like(X_real)
        
        # Initialize states
        r = self.initial_r.unsqueeze(0).repeat(batch_size, 1)
        phi = self.initial_phi.unsqueeze(0).repeat(batch_size, 1)
        
        # Compute frequencies
        omega_intl = self.max_omega - self.min_omega
        omegas = torch.sigmoid(self.omegas) * omega_intl + self.min_omega
        omegas = omegas * (2 * np.pi)
        omegas = omegas.repeat(batch_size, 1)
        
        # Run integration steps with coupling
        for _ in range(self.num_steps):
            # Input forcing
            input_r = self.input_scaler * X_real * torch.cos(phi)
            input_phi = self.input_scaler * X_imag * torch.sin(phi)
            
            # Coupling contribution
            coupling_r = self.coupling_strength * torch.matmul(
                self.coupling_matrix, (r * torch.cos(phi)).T
            ).T
            coupling_phi = self.coupling_strength * torch.matmul(
                self.coupling_matrix, (r * torch.sin(phi)).T
            ).T
            
            # Hopf dynamics with coupling
            dr_dt = (self.mu0 - self.beta1 * (r**2)) * r + input_r + coupling_r
            dphi_dt = omegas - input_phi + coupling_phi
            
            r = r + dr_dt * self.dt
            phi = phi + dphi_dt * self.dt
        
        # Convert to complex
        z_real = r * torch.cos(phi)
        z_imag = r * torch.sin(phi)
        z_complex = torch.complex(z_real, z_imag)
        
        return z_complex, r, phi


# ============================================================================
# MAIN ARCHITECTURE COMPONENTS
# ============================================================================

class FeedbackIntermediateLayer(nn.Module):
    """
    Feedback Intermediate Layer (BL1-BL4)
    BL4 contains oscillators.
    """
    
    def __init__(self, N_s=256, device="cpu"):
        super().__init__()
        
        self.device = device
        
        # BL1: 256 neurons
        self.BL1 = nn.Sequential(
            nn.Linear(N_s, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU()
        )
        
        # BL2: 256 -> 128
        self.BL2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU()
        )
        
        # BL3: 128 -> 64
        self.BL3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU()
        )
        
        # BL4: 64 -> 2 with oscillators (6 oscillators)
        self.BL4_pre = nn.Sequential(
            nn.Linear(64, 6),
            nn.SiLU()
        )
        self.BL4_osc = CoupledOscillatorLayer(
            units=6, min_omega=2.0, max_omega=10.0, dt=0.01,
            coupling_strength=0.05, train_omegas=True, device=device
        )
        self.BL4_post = nn.Linear(6, 2)
        
        self.to(device)
    
    def forward(self, x):
        """
        x: (batch_size, N_s)
        Returns: (batch_size, 2) for heart oscillators
        """
        bl1_out = self.BL1(x)
        bl2_out = self.BL2(bl1_out)
        bl3_out = self.BL3(bl2_out)
        
        # BL4 with oscillators
        bl4_pre = self.BL4_pre(bl3_out)
        z_osc, r, phi = self.BL4_osc(bl4_pre)
        bl4_out = self.BL4_post(torch.real(z_osc))
        
        return bl1_out, bl4_out  # Return BL1 for top oscillator layer


class HeartModel(nn.Module):
    """
    Heart Model with fundamental oscillators.
    L1 -> L2 -> L3, with fundamental oscillators below L1.
    """
    
    def __init__(self, input_dim=4, hidden_dim=100, feature_dim=50, output_dim=1, device="cpu"):
        super().__init__()
        
        self.device = device
        
        # Fundamental oscillators (2 oscillators)
        self.fundamental_osc = CoupledOscillatorLayer(
            units=2, min_omega=5.0, max_omega=5.2, dt=0.01,
            coupling_strength=0.0001, train_omegas=False, device=device
        )
        
        # Heart layers
        self.L1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.L2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.L3 = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.SiLU()
        )
        
        self.output_layer = nn.Linear(feature_dim, output_dim)
        
        self.to(device)
    
    def forward(self, x, feedback_input=None):
        """
        x: (batch_size, 4) or (batch_size, T, 4) - fundamental oscillator input
        feedback_input: (batch_size, 2) or (batch_size, T, 2) - from BL4
        Returns: ECG output and features
        """
        # Handle time-series vs single-step
        is_time_series = x.dim() == 3
        if is_time_series:
            batch_size, T, _ = x.shape
            x = x.view(-1, x.shape[-1])  # (B*T, 4)
            if feedback_input is not None:
                feedback_input = feedback_input.view(-1, feedback_input.shape[-1])  # (B*T, 2)
        
        # Fundamental oscillators
        if feedback_input is not None:
            z_fund, r_fund, phi_fund = self.fundamental_osc(feedback_input)
            fund_real = torch.real(z_fund)  # (B*T, 2)
        else:
            fund_real = x[:, :2]  # (B*T, 2)
        
        # Combine with input (use first 2 dims of x)
        combined = torch.cat([fund_real, x[:, :2]], dim=-1)  # (B*T, 4)
        
        l1_out = self.L1(combined)
        l2_out = self.L2(l1_out)
        l3_out = self.L3(l2_out)
        
        ecg_output = self.output_layer(l3_out)
        
        # Reshape back if time-series
        if is_time_series:
            ecg_output = ecg_output.view(batch_size, T, -1)
            l3_out = l3_out.view(batch_size, T, -1)
        
        return ecg_output, l3_out  # Return features for ECG-to-brain pathway
    
    def get_features(self, x, feedback_input=None):
        """Get features without ECG output."""
        _, features = self.forward(x, feedback_input)
        return features


class ECGToBrainPathway(nn.Module):
    """
    ECG-to-Brain pathway (HL1 -> NTS Oscillators -> HL2)
    Optimized for computational efficiency.
    """
    
    def __init__(self, ecg_feature_dim=50, N_VNS=128, hidden_dim=64, 
                 output_dim=16, device="cpu"):
        super().__init__()
        
        self.device = device
        
        # HL1: ECG features -> hidden representation
        self.HL1 = nn.Sequential(
            nn.Linear(ecg_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # 3 neurons
        )
        
        # NTS Oscillator Layer (6 oscillators)
        self.NTS_osc = CoupledOscillatorLayer(
            units=6, min_omega=2.0, max_omega=10.0, dt=0.01,
            coupling_strength=0.05, train_omegas=True, device=device
        )
        self.NTS_proj = nn.Linear(3, 6)  # Project HL1 output to NTS
        
        # HL2: NTS -> output
        self.HL2 = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)  # 4 neurons
        )
        
        self.to(device)
    
    def forward(self, ecg_features):
        """
        ecg_features: (batch_size, ecg_feature_dim) or (T, ecg_feature_dim)
        Returns: (batch_size, output_dim)
        """
        if ecg_features.dim() == 1:
            ecg_features = ecg_features.unsqueeze(0)
        
        # HL1
        hl1_out = self.HL1(ecg_features)  # (B, 3)
        
        # Project to NTS oscillator input
        nts_input = self.NTS_proj(hl1_out)  # (B, 6)
        
        # NTS oscillators
        z_nts, r_nts, phi_nts = self.NTS_osc(nts_input)
        nts_real = torch.real(z_nts)  # (B, 6)
        
        # HL2
        hl2_out = self.HL2(nts_real)  # (B, output_dim)
        
        return hl2_out


class TopOscillatorLayer(nn.Module):
    """
    Top oscillator layer receiving inputs from HL2 and BL1.
    """
    
    def __init__(self, hl2_dim=4, bl1_dim=256, output_dim=5, device="cpu"):
        super().__init__()
        
        self.device = device
        
        # Project inputs
        self.hl2_proj = nn.Linear(hl2_dim, 5)
        self.bl1_proj = nn.Linear(bl1_dim, 5)
        
        # Top oscillators (5 oscillators)
        self.top_osc = CoupledOscillatorLayer(
            units=5, min_omega=2.0, max_omega=10.0, dt=0.01,
            coupling_strength=0.05, train_omegas=True, device=device
        )
        
        self.to(device)
    
    def forward(self, hl2_input, bl1_input):
        """
        hl2_input: (batch_size, hl2_dim)
        bl1_input: (batch_size, bl1_dim)
        Returns: (batch_size, output_dim) - MEG output
        """
        # Project inputs
        hl2_proj = self.hl2_proj(hl2_input)  # (B, 5)
        bl1_proj = self.bl1_proj(bl1_input)  # (B, 5)
        
        # Combine
        combined = hl2_proj + bl1_proj
        
        # Top oscillators
        z_top, r_top, phi_top = self.top_osc(combined)
        meg_output = torch.real(z_top)  # (B, 5)
        
        return meg_output


class FullModel(nn.Module):
    """
    Complete model integrating all components.
    """
    
    def __init__(self, N_s=256, ecg_feature_dim=50, device="cpu"):
        super().__init__()
        
        self.device = device
        
        # Components
        self.feedback_layer = FeedbackIntermediateLayer(N_s=N_s, device=device)
        self.heart_model = HeartModel(device=device)
        self.ecg_to_brain = ECGToBrainPathway(
            ecg_feature_dim=ecg_feature_dim, device=device
        )
        self.top_osc_layer = TopOscillatorLayer(device=device)
        
        self.to(device)
    
    def forward(self, heart_input, brain_input):
        """
        heart_input: (batch_size, T, 4) or (batch_size, 4) - fundamental oscillator input
        brain_input: (batch_size, T, N_s) or (batch_size, N_s) - brain state input
        
        Returns:
            ecg_output: (batch_size, T, 1) or (batch_size, 1)
            meg_output: (batch_size, T, 5) or (batch_size, 5)
        """
        # Handle time-series vs single-step
        is_time_series = heart_input.dim() == 3
        
        if is_time_series:
            batch_size, T, _ = heart_input.shape
            # Process each time step
            ecg_outputs = []
            meg_outputs = []
            
            for t in range(T):
                heart_t = heart_input[:, t, :]  # (B, 4)
                brain_t = brain_input[:, t, :]  # (B, N_s)
                
                # Feedback layer
                bl1_out, bl4_out = self.feedback_layer(brain_t)
                
                # Heart model with feedback
                ecg_output, ecg_features = self.heart_model(heart_t, bl4_out)
                
                # ECG-to-brain pathway
                hl2_out = self.ecg_to_brain(ecg_features)
                
                # Top oscillator layer
                meg_output = self.top_osc_layer(hl2_out, bl1_out)
                
                ecg_outputs.append(ecg_output)
                meg_outputs.append(meg_output)
            
            ecg_output = torch.stack(ecg_outputs, dim=1)  # (B, T, 1)
            meg_output = torch.stack(meg_outputs, dim=1)  # (B, T, 5)
        else:
            # Single-step processing
            bl1_out, bl4_out = self.feedback_layer(brain_input)
            ecg_output, ecg_features = self.heart_model(heart_input, bl4_out)
            hl2_out = self.ecg_to_brain(ecg_features)
            meg_output = self.top_osc_layer(hl2_out, bl1_out)
        
        return ecg_output, meg_output


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_heart_model(ecg_target_signal, device, num_epochs=10000):
    """Pre-train heart model."""
    print("--- Training Heart Model ---")
    heart_model = HeartModel(device=device).to(device)
    optimizer = optim.Adam(heart_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    sim_osc_input = torch.tensor(
        heart_osc(T=2, dt=0.01), dtype=torch.float32
    ).to(device)
    ecg_target = torch.tensor(
        ecg_target_signal[::10], dtype=torch.float32
    ).to(device).unsqueeze(1)
    
    for epoch in range(num_epochs):
        predicted_ecg, _ = heart_model(sim_osc_input)
        loss = criterion(predicted_ecg, ecg_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    print("--- Heart Model Training Complete ---\n")
    return heart_model


def train_full_model(model, heart_input, brain_input, ecg_target, meg_target,
                     device, num_epochs=500, lr=1e-3):
    """Train the full model end-to-end."""
    print("--- Training Full Model ---")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(num_epochs):
        ecg_pred, meg_pred = model(heart_input, brain_input)
        
        # Reshape predictions and targets for loss calculation
        ecg_pred_flat = ecg_pred.view(-1, ecg_pred.shape[-1])
        ecg_target_flat = ecg_target.view(-1, ecg_target.shape[-1])
        meg_pred_flat = meg_pred.view(-1, meg_pred.shape[-1])
        meg_target_flat = meg_target.view(-1, meg_target.shape[-1])
        
        ecg_loss = criterion(ecg_pred_flat, ecg_target_flat)
        meg_loss = criterion(meg_pred_flat, meg_target_flat)
        total_loss = ecg_loss + meg_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, "
                  f"Total Loss: {total_loss.item():.6f}, "
                  f"ECG Loss: {ecg_loss.item():.6f}, "
                  f"MEG Loss: {meg_loss.item():.6f}")
    
    print("--- Full Model Training Complete ---\n")
    return model, losses


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training script."""
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Data paths
    ecg_file_path = '/home/shobs/Desktop/DDP/transdef_mf2pt2_rest_raw.fif'
    eeg_file_path = "/home/shobs/Desktop/DDP/scout_id_309.mat"
    sc_file_path = '/home/shobs/Desktop/DDP/SC_CC120309-27.mat'
    
    # Load and preprocess data
    print("Loading data...")
    ecg_data, eeg_data, Sw_all = load_data(ecg_file_path, eeg_file_path, sc_file_path)
    ecg_processed = preprocess_signal(ecg_data, fs=1000, lowcut=1.5, highcut=20)
    eeg_processed = np.array([
        preprocess_signal(row, fs=1000, lowcut=0.5, highcut=30) 
        for row in eeg_data
    ])
    print("Data loaded and preprocessed.\n")
    
    # Prepare training data
    t_duration = 2.0
    fs = 100
    t = np.arange(0, t_duration, 1/fs)
    T = len(t)
    
    # Heart input (fundamental oscillators)
    heart_osc_input = heart_osc(T=t_duration, dt=1/fs)
    heart_input = torch.tensor(heart_osc_input, dtype=torch.float32).to(device)  # (T, 4)
    
    # Brain input (use first N_s=256 dimensions of EEG, pad if needed)
    N_s = 256
    eeg_channels = min(N_s, eeg_processed.shape[0])
    brain_data = eeg_processed[:eeg_channels, ::10].T  # (T, eeg_channels)
    
    # Pad to N_s if needed
    if eeg_channels < N_s:
        padding = np.zeros((T, N_s - eeg_channels))
        brain_data = np.concatenate([brain_data, padding], axis=1)
    
    brain_input = torch.tensor(brain_data, dtype=torch.float32).to(device)  # (T, N_s)
    
    # Targets
    ecg_target = torch.tensor(
        ecg_processed[::10], dtype=torch.float32
    ).to(device).unsqueeze(-1)  # (T, 1)
    
    # MEG target (use first 5 EEG channels)
    meg_target = torch.tensor(
        eeg_processed[:5, ::10].T, dtype=torch.float32
    ).to(device)  # (T, 5)
    
    # Add batch dimension (batch_size=1 for now)
    heart_input = heart_input.unsqueeze(0)  # (1, T, 4)
    brain_input = brain_input.unsqueeze(0)  # (1, T, N_s)
    ecg_target = ecg_target.unsqueeze(0)  # (1, T, 1)
    meg_target = meg_target.unsqueeze(0)  # (1, T, 5)
    
    print(f"Heart input shape: {heart_input.shape}")
    print(f"Brain input shape: {brain_input.shape}")
    print(f"ECG target shape: {ecg_target.shape}")
    print(f"MEG target shape: {meg_target.shape}\n")
    
    # Step 1: Pre-train heart model
    heart_model = train_heart_model(ecg_processed, device, num_epochs=5000)
    
    # Step 2: Create and train full model
    model = FullModel(N_s=N_s, ecg_feature_dim=50, device=device)
    
    # Use pre-trained heart model weights
    model.heart_model.load_state_dict(heart_model.state_dict(), strict=False)
    
    # Train full model
    model, losses = train_full_model(
        model, heart_input, brain_input, ecg_target, meg_target,
        device, num_epochs=200, lr=1e-3
    )
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    model.eval()
    with torch.no_grad():
        ecg_pred, meg_pred = model(heart_input, brain_input)
        
        # Reshape for loss calculation
        ecg_pred_flat = ecg_pred.view(-1, ecg_pred.shape[-1])
        ecg_target_flat = ecg_target.view(-1, ecg_target.shape[-1])
        meg_pred_flat = meg_pred.view(-1, meg_pred.shape[-1])
        meg_target_flat = meg_target.view(-1, meg_target.shape[-1])
        
        ecg_mse = nn.MSELoss()(ecg_pred_flat, ecg_target_flat).item()
        meg_mse = nn.MSELoss()(meg_pred_flat, meg_target_flat).item()
        
        print(f"ECG MSE: {ecg_mse:.6f}")
        print(f"MEG MSE: {meg_mse:.6f}")
    
    print("\nTraining complete!")
    
    return model, losses


if __name__ == "__main__":
    model, losses = main()
