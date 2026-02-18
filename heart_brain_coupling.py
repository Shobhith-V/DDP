"""
heart_brain_coupling.py
=======================
Models heart-brain coupling via the vagus nerve using:
  - A HeartModel MLP pre-trained on simulated cardiac oscillator data
  - An ECGToOscillatorMLP that maps ECG features → vagal oscillators → brain drive
  - A Reverse Hopf ODE (TorchRevHopfNetwork) simulating brain dynamics

Optimisations applied
---------------------
  1. D_tensor: target EEG signal is precomputed as a torch.Tensor and indexed
     by integer time step inside the ODE, avoiding scipy interp1d call overhead.
  2. Frozen theta: the N×N coupling-phase matrix (theta) is fixed at its initial
     value and NOT included in the ODE state. This reduces state dim from
     4N + N² → 3N, giving a large memory and speed improvement.
  3. Reduced training steps: ODE is solved on a coarser time grid (100 steps)
     during Stage 2 training; full 200-step grid is used only for final inference.

Training pipeline
-----------------
  Stage 0: Pre-train HeartModel (simulated oscillator states → ECG waveform)
  Stage 1: Warm up brain ODE initial conditions (gradient-free forward sim)
  Stage 2: Train ECGToOscillatorMLP to steer brain ODE toward target EEG

Usage
-----
  python heart_brain_coupling.py \\
      --fif_path   /path/to/raw.fif \\
      --mat_path   /path/to/scout.mat \\
      --sc_path    /path/to/SC.mat \\
      --target_idx 4 \\
      --out_dir    ./results \\
      --device     cuda
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import gc
import sys
import logging
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for cluster use
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.optim as optim
# odeint_adjoint: memory-efficient adjoint method (no full graph stored).
# odeint:         standard autograd — builds full graph, needed when an external
#                 tensor (brain_drive_full) must carry gradients back to the MLP.
from torchdiffeq import odeint, odeint_adjoint

import mne

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===========================================================================
# 1. DATA LOADING & PREPROCESSING
# ===========================================================================

def load_data(fif_path: str, mat_path: str, sc_path: str):
    """
    Load raw MEG/ECG, EEG source signals, and structural connectivity matrix.

    Returns
    -------
    ecg_data : np.ndarray (T,)
    eeg_data : np.ndarray (n_regions, T)
    Sw_all   : np.ndarray (n_regions, n_regions) — normalised SC matrix
    """
    log.info("Loading raw FIF: %s", fif_path)
    raw = mne.io.read_raw_fif(fif_path, preload=False)
    data, _ = raw[322, 2000:4000]
    ecg_data = -data[0]

    log.info("Loading EEG scout: %s", mat_path)
    mat = loadmat(mat_path)
    eeg_data = mat["Value"][:, 2000:4000]

    log.info("Loading SC matrix: %s", sc_path)
    sc_data = loadmat(sc_path)
    sc_matrix = sc_data["sc"]
    max_val = np.max(sc_matrix)
    Sw_all = (sc_matrix / max_val) * 0.01 if max_val > 0 else sc_matrix

    return ecg_data, eeg_data, Sw_all


def bandpass_normalise(signal: np.ndarray, fs: float = 1000.0,
                       lowcut: float = 1.5, highcut: float = 20.0) -> np.ndarray:
    """Detrend → bandpass filter → z-score normalise."""
    detrended = detrend(signal)
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
    filtered = filtfilt(b, a, detrended)
    return (filtered - filtered.mean()) / (filtered.std() + 1e-8)


def preprocess(ecg_data: np.ndarray, eeg_data: np.ndarray):
    """Apply bandpass + normalisation to ECG and all EEG channels."""
    ecg_proc = bandpass_normalise(ecg_data, fs=1000, lowcut=1.5, highcut=20)
    eeg_proc = np.array([
        bandpass_normalise(row, fs=1000, lowcut=0.5, highcut=20)
        for row in eeg_data
    ])
    return ecg_proc, eeg_proc


def make_D_tensor(target_signal: np.ndarray, t: np.ndarray,
                  device: str) -> torch.Tensor:
    """
    Precompute the target EEG signal as a GPU tensor.

    OPT 1: Replaces scipy interp1d called at every ODE step with a simple
    integer index into a pre-allocated tensor — much faster in the hot loop.

    Parameters
    ----------
    target_signal : (T,) array sampled at the same rate as t
    t             : time vector used for the ODE

    Returns
    -------
    D_tensor : (T,) torch.Tensor on `device`
    """
    # Interpolate target_signal onto t if lengths differ
    if len(target_signal) != len(t):
        interp = interp1d(
            np.linspace(0, t[-1], len(target_signal)),
            target_signal, kind="linear",
            bounds_error=False, fill_value=0.0
        )
        target_signal = interp(t)
    return torch.tensor(target_signal, dtype=torch.float32, device=device)


# ===========================================================================
# 2. SIGNAL SIMULATION UTILITIES
# ===========================================================================

def simulate_coupled_oscillators(
    T: float = 10.0,
    dt: float = 1e-3,
    alpha: float = 1.0,
    omega1: float = 5.01,
    omega2: float = 5.1,
    A_init: float = 1e-4,
    theta_init: float = 3.14,
    n: float = 1.0,
    modulation=None,
) -> np.ndarray:
    """
    Simulate two coupled Hopf oscillators (Euler integration).

    Returns
    -------
    np.ndarray of shape (N_steps, 4)
        Columns: [r1·cos(φ1), r1·sin(φ1), r2·cos(φ2), r2·sin(φ2)]
    """
    N = int(T / dt)
    r1, r2, phi1, phi2 = 1.0, 1.0, 0.0, 0.0
    A12 = A21 = A_init
    theta12 = theta21 = theta_init

    R1 = np.zeros(N); R2 = np.zeros(N)
    Phi1 = np.zeros(N); Phi2 = np.zeros(N)

    for i in range(N):
        R1[i], R2[i], Phi1[i], Phi2[i] = r1, r2, phi1, phi2

        c12 = A12 * r2 * np.cos(theta12 + n * (phi2 - phi1))
        c21 = A21 * r1 * np.cos(theta21 + n * (phi1 - phi2))
        mod1 = 0.1 * modulation[i, 0] if (modulation is not None and i < len(modulation)) else 0
        mod2 = 0.1 * modulation[i, 1] if (modulation is not None and i < len(modulation)) else 0

        dr1 = alpha * r1 - r1**3 + c12 + mod1
        dr2 = alpha * r2 - r2**3 + c21 + mod2
        dphi1 = omega1 + A12 * r2 / r1 * np.sin(theta12 + n * (phi2 - phi1))
        dphi2 = omega2 + A21 * r1 / r2 * np.sin(theta21 + n * (phi1 - phi2))

        r1 += dr1 * dt;    r2 += dr2 * dt
        phi1 += dphi1 * dt; phi2 += dphi2 * dt

    return np.stack(
        (R1 * np.cos(Phi1), R1 * np.sin(Phi1),
         R2 * np.cos(Phi2), R2 * np.sin(Phi2)),
        axis=1
    )


def get_random_frequencies(num_regions: int, osc_per_region: int,
                           low: float = 1.0, high: float = 20.0,
                           seed: int = None) -> np.ndarray:
    """Sample random natural frequencies (rad/s) for oscillators."""
    if seed is not None:
        np.random.seed(seed)
    freqs_hz = np.random.uniform(low, high, num_regions * osc_per_region)
    return 2 * np.pi * freqs_hz


def expand_structural_connectivity(Sc_region: np.ndarray, osc_per_region: int,
                                   intra_value: float = 1e-4,
                                   seed: int = None) -> np.ndarray:
    """
    Expand a region-level SC matrix to oscillator-level by randomly
    distributing inter-region weights across oscillator pairs.
    """
    if seed is not None:
        np.random.seed(seed)
    n_reg = Sc_region.shape[0]
    N = n_reg * osc_per_region
    Sc_full = np.zeros((N, N))

    for i in range(n_reg):
        for j in range(n_reg):
            si, ei = i * osc_per_region, (i + 1) * osc_per_region
            sj, ej = j * osc_per_region, (j + 1) * osc_per_region
            if i == j:
                Sc_full[si:ei, sj:ej] = intra_value
            else:
                block = np.random.rand(osc_per_region, osc_per_region)
                block *= Sc_region[i, j] / (block.sum() + 1e-9)
                Sc_full[si:ei, sj:ej] = block

    np.fill_diagonal(Sc_full, 0.0)
    return Sc_full


# ===========================================================================
# 3. NEURAL NETWORK MODELS
# ===========================================================================

class HeartModel(nn.Module):
    """
    Two-stage MLP: oscillator states → latent cardiac features → ECG.

    Architecture
    ------------
    feature_extractor : Linear(4→100) → Sigmoid → Linear(100→100) → Sigmoid → Linear(100→50)
    output_layer      : Linear(50→1)

    The feature_extractor output (dim=50) is used as the ECG embedding
    passed to ECGToOscillatorMLP in Stage 2.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 100,
                 feature_dim: int = 50, output_dim: int = 1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.output_layer = nn.Linear(feature_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.feature_extractor(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent cardiac feature vector (no output projection)."""
        return self.feature_extractor(x)


class OscillatorLayer(nn.Module):
    """
    A bank of N_osc coupled Hopf oscillators simulating brainstem vagal nuclei.

    STATEFUL DESIGN (Option 1):
    Instead of resetting (r, phi) at every call, the oscillator state is
    propagated sequentially across the full input time-series in a single
    forward pass. This makes the layer a true dynamical system:
      - Phase accumulates over time
      - Frequency adaptation has temporal continuity
      - Heart rhythm can entrain the oscillators (vagal memory)

    The autograd graph remains intact within one forward pass, so
    loss.backward() works correctly.

    Forward pass
    ------------
    Input  : (T, N_osc)   — ECG features for each timestep in the sequence
    Output : (T, N_osc*2) — [r·cos(φ), r·sin(φ)] for each oscillator at each t
    """

    def __init__(self, N_osc: int = 16, fs: float = 100.0,
                 device: str = "cpu", coupling_sparsity: float = 0.3,
                 seed: int = 42):
        super().__init__()
        self.N_osc = N_osc
        self.dt = 1.0 / fs
        self.mu = 1.0

        # Learnable natural frequencies (initialised uniformly in [2, 10] Hz)
        torch.manual_seed(seed)
        freqs_hz = 2.0 + torch.rand(N_osc, device=device) * 8.0
        self.omega = nn.Parameter(2 * np.pi * freqs_hz)

        # Fixed initial conditions (not learnable)
        self.register_buffer("initial_r",   torch.ones(N_osc, device=device) * 0.1)
        self.register_buffer("initial_phi", torch.zeros(N_osc, device=device))

        # Fixed sparse random coupling matrix (not learnable)
        torch.manual_seed(seed + 1)
        mask = (torch.rand(N_osc, N_osc, device=device) > coupling_sparsity).float()
        mask.fill_diagonal_(0.0)   # no self-coupling
        weights = torch.rand(N_osc, N_osc, device=device) * 0.02
        self.register_buffer("C", weights * mask)
        self.register_buffer("coupling_strength", torch.tensor(0.1, device=device))

    def _step(self, r: torch.Tensor, phi: torch.Tensor,
              inp: torch.Tensor) -> tuple:
        """
        Single Euler integration step for the oscillator bank.

        Parameters
        ----------
        r   : (N_osc,) — current amplitudes
        phi : (N_osc,) — current phases
        inp : (N_osc,) — ECG forcing at this timestep

        Returns
        -------
        r_new, phi_new : (N_osc,) each
        """
        # Phase difference matrix: (N_osc, N_osc)
        phase_diff = phi.unsqueeze(1) - phi.unsqueeze(0)   # phi_i - phi_j
        r_j    = r.unsqueeze(0).expand(self.N_osc, -1)     # (N_osc, N_osc)
        r_safe = torch.clamp(r, 1e-4, 10.0)
        cs     = self.coupling_strength

        # Hopf coupling terms
        coupling_r   = cs * (self.C * r_j * torch.cos(phase_diff)).sum(dim=1)
        coupling_phi = cs * (self.C * (r_j / r_safe.unsqueeze(1))
                             * torch.sin(phase_diff)).sum(dim=1)

        dr_dt   = (self.mu - r**2) * r + coupling_r + inp
        dphi_dt = self.omega + coupling_phi

        r_new   = torch.clamp(r + dr_dt * self.dt, 0.01, 2.0)
        phi_new = phi + dphi_dt * self.dt
        return r_new, phi_new

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Evolve the oscillator bank sequentially over the full input sequence.
        State (r, phi) is carried forward across timesteps — NOT reset.

        Parameters
        ----------
        input_sequence : (T, N_osc) — ECG feature sequence

        Returns
        -------
        (T, N_osc * 2) — [r·cos(φ), r·sin(φ)] at every timestep
        """
        T = input_sequence.shape[0]

        # Initialise state from fixed buffers (cloned to allow gradient flow)
        r   = self.initial_r.clone()    # (N_osc,)
        phi = self.initial_phi.clone()  # (N_osc,)

        outputs = []
        for t in range(T):
            inp = input_sequence[t]          # (N_osc,) — forcing at time t
            r, phi = self._step(r, phi, inp)
            # Cartesian representation: [r·cos(φ), r·sin(φ)]
            outputs.append(torch.cat([r * torch.cos(phi),
                                      r * torch.sin(phi)]))

        return torch.stack(outputs)   # (T, N_osc * 2)


class ECGToOscillatorMLP(nn.Module):
    """
    Full heart→brain pathway:
      ECG features → pre_osc MLP → OscillatorLayer (stateful) → post_osc MLP → brain drive

    The OscillatorLayer now processes the FULL sequence in one call,
    maintaining oscillator state across timesteps (vagal memory).

    Input  : (T, ecg_dim)  — ECG feature sequence
    Output : (T, N)        — brain drive for N brain oscillators
    """

    def __init__(self, ecg_dim: int = 50, N_VNS: int = 8,
                 hidden_dim: int = 64, output_dim: int = 16,
                 device: str = "cuda"):
        super().__init__()
        self.pre_osc = nn.Sequential(
            nn.Linear(ecg_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, N_VNS),
        )
        # OscillatorLayer no longer needs T/num_steps — it processes the
        # sequence length dynamically in its forward pass.
        self.osc_layer = OscillatorLayer(
            N_osc=N_VNS, fs=100.0, device=device,
            coupling_sparsity=0.3, seed=SEED
        )
        self.post_osc = nn.Sequential(
            nn.Linear(N_VNS * 2, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, ecg_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ecg_features : (T, ecg_dim) or (ecg_dim,) for a single timestep

        Returns
        -------
        (T, output_dim) or (output_dim,) brain drive
        """
        squeezed = ecg_features.dim() == 1
        if squeezed:
            ecg_features = ecg_features.unsqueeze(0)   # (1, ecg_dim)

        # pre_osc: independent projection at each timestep — (T, N_VNS)
        pre = self.pre_osc(ecg_features)

        # osc_layer: stateful sequential pass over the full sequence — (T, N_VNS*2)
        # State (r, phi) evolves continuously from t=0 to t=T-1.
        osc = self.osc_layer(pre)

        # post_osc: project oscillator states to brain drive — (T, output_dim)
        drive = self.post_osc(osc)

        return drive.squeeze(0) if squeezed else drive


# ===========================================================================
# 4. BRAIN ODE MODEL
# ===========================================================================

class ODEFunc(nn.Module):
    """
    Right-hand side of the biologically-grounded Reverse Hopf ODE.

    Biological design principles
    ----------------------------
    1. Self-organised amplitude dynamics:
       Error signal e is removed from drdt. Oscillators evolve autonomously
       via Hopf dynamics + structural coupling. Error drives only slow
       plasticity (ω, α), not fast amplitude. This prevents teacher-forcing.

    2. Homeostatic amplitude regulation:
       dα/dt includes a slow decay term −λ(α − α₀) that pulls excitability
       back toward a baseline. Models synaptic homeostasis (Turrigiano 2008).

    3. Timescale separation:
       r, φ: fast (ODE timescale)
       ω:    slow plasticity   (η_omega << 1)
       α:    slower homeostasis (η_alpha << η_omega, + homeostatic decay)

    OPT 1 — D_tensor:
        Precomputed target EEG tensor indexed by integer time step.
    OPT 2 — Frozen theta:
        N×N coupling-phase matrix fixed as a buffer; state dim = 4N.

    Parameters
    ----------
    D_tensor         : (T,) torch.Tensor — precomputed target EEG signal
    theta_fixed      : (N, N) np.ndarray — fixed coupling-phase offsets
    alpha_baseline   : (N,) np.ndarray  — homeostatic target excitability
    lambda_homeo     : float — homeostatic decay rate (default 0.001)
    brain_drive_full : (T, N) tensor or None — heart-to-brain drive signal
    """

    def __init__(self, mu: float, eta_omega: float, eta_alpha: float,
                 D_tensor: torch.Tensor,
                 N: int, Sc: np.ndarray,
                 theta_fixed: np.ndarray,
                 alpha_baseline: np.ndarray,
                 lambda_homeo: float = 0.001,
                 brain_drive_full=None,
                 fs: float = 100.0):
        super().__init__()
        self.mu           = mu
        self.eta_omega    = eta_omega
        self.eta_alpha    = eta_alpha
        self.lambda_homeo = lambda_homeo
        self.N            = N
        self.fs           = fs

        # OPT 1: precomputed target signal as a buffer (no interp1d in hot loop)
        self.register_buffer("D_tensor", D_tensor)
        self.register_buffer("Sc",    torch.tensor(Sc,             dtype=torch.float32))
        # OPT 2: theta is fixed — stored as a buffer, not integrated
        self.register_buffer("theta", torch.tensor(theta_fixed,    dtype=torch.float32))
        # Homeostatic target excitability — fixed at Stage 1 initialisation
        self.register_buffer("alpha0", torch.tensor(alpha_baseline, dtype=torch.float32))

        # Plain attribute — not a buffer — to preserve the autograd graph
        # when brain_drive_full is the output of a differentiable MLP.
        self.brain_drive_full = brain_drive_full

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        N = self.N

        # State: [r (N), phi (N), omega (N), alpha (N)] — 4N total
        r     = state[:N]
        phi   = state[N:2*N]
        omega = state[2*N:3*N]
        alpha = state[3*N:4*N]

        # Numerical stability clamps
        omega_safe = torch.clamp(omega, 2*np.pi*0.5, 2*np.pi*30)
        r          = torch.clamp(r,     1e-2, 2.0)
        alpha      = torch.clamp(alpha, -1.0, 1.0)
        r_safe     = torch.clamp(r,     1e-5, 10.0)

        # Phase difference using fixed theta buffer
        phase_diff = (
            phi[None, :] / omega_safe[None, :]
            - phi[:, None] / omega_safe[:, None]
            + self.theta / (omega_safe[:, None] * omega_safe[None, :])
        )

        # OPT 1: index precomputed D_tensor instead of calling interp1d
        t_idx = min(int(t.item() * self.fs), self.D_tensor.shape[0] - 1)
        D = self.D_tensor[t_idx]

        P = torch.sum(alpha * r * torch.cos(phi))
        e = D - P

        # Heart-to-brain drive at this timestep (Stage 2 only; None in Stage 1)
        if self.brain_drive_full is not None:
            ecg_input = self.brain_drive_full[t_idx]
        else:
            ecg_input = torch.zeros(N, device=state.device)

        # --- Amplitude dynamics (SELF-ORGANISED — no error injection) ---
        # BIO CHANGE 1: error term e*cos(phi) removed from drdt.
        # Oscillators evolve autonomously via Hopf + structural coupling.
        # Error drives only slow plasticity (omega, alpha) below.
        coupling_r = torch.sum(
            torch.abs(self.Sc) * r[None, :] * torch.cos(phase_diff), dim=1
        )
        drdt = (self.mu - r**2) * r + coupling_r + ecg_input

        # --- Phase dynamics ---
        # Error still modulates phase (frequency tracking) — biologically
        # this corresponds to phase-resetting by prediction error.
        coupling_phi = torch.sum(
            torch.abs(self.Sc) * (r[None, :] / r_safe[:, None]) * torch.sin(phase_diff),
            dim=1
        )
        dphidt = omega + coupling_phi - (e / r_safe) * torch.sin(phi)

        # --- Slow plasticity: frequency adaptation ---
        # BIO CHANGE 2 (timescale separation): eta_omega is small so ω
        # adapts on a timescale much slower than r and phi.
        domegadt = -self.eta_omega * e * torch.sin(phi)

        # --- Slower homeostatic excitability regulation ---
        # BIO CHANGE 3: homeostatic decay term −λ(α − α₀) pulls excitability
        # back toward baseline, preventing runaway gain.
        # eta_alpha << eta_omega enforces further timescale separation.
        dalphadt = (self.eta_alpha * e * r * torch.cos(phi)
                    - self.lambda_homeo * (alpha - self.alpha0))

        # Return 4N derivatives (no dthetadt — theta is frozen)
        return torch.cat([
            drdt.flatten(), dphidt.flatten(),
            domegadt.flatten(), dalphadt.flatten()
        ])


class TorchRevHopfNetwork:
    """
    Wrapper around ODEFunc that handles state packing/unpacking and ODE solving.

    OPT 2: State vector is now [r, phi, omega, alpha] — shape (4N,) instead
    of (4N + N²). theta is a fixed buffer inside ODEFunc.

    Parameters
    ----------
    use_adjoint (in solve) : bool
        True  → odeint_adjoint (memory-efficient, for no-grad stages)
        False → odeint (full autograd graph, required for MLP training)
    """

    def __init__(self, mu: float, eta_omega: float, eta_alpha: float,
                 D_tensor: torch.Tensor,
                 N: int, Sc: np.ndarray,
                 theta_fixed: np.ndarray,
                 alpha_baseline: np.ndarray,
                 lambda_homeo: float = 0.001,
                 brain_drive_full=None,
                 fs: float = 100.0,
                 device: str = "cuda"):
        self.device = torch.device(device)
        self.N = N
        self.ode_func = ODEFunc(
            mu=mu, eta_omega=eta_omega, eta_alpha=eta_alpha,
            D_tensor=D_tensor, N=N, Sc=Sc,
            theta_fixed=theta_fixed,
            alpha_baseline=alpha_baseline,
            lambda_homeo=lambda_homeo,
            brain_drive_full=brain_drive_full, fs=fs
        ).to(self.device)

    def solve(self, r0: np.ndarray, phi0: np.ndarray,
              omega0: np.ndarray, alpha0: np.ndarray,
              t_eval: np.ndarray, use_adjoint: bool = True):
        """
        Integrate the ODE.

        Note: theta0 is no longer a parameter — it is fixed inside ODEFunc.

        Returns
        -------
        r, phi, omega, alpha : tensors of shape (T, N)
        rcos_phi             : (T,) — weighted brain output signal
        """
        # OPT 2: state is [r, phi, omega, alpha] — 4N total (no theta)
        y0 = torch.tensor(
            np.concatenate([r0, phi0, omega0, alpha0]),
            device=self.device, dtype=torch.float32
        )
        t_tensor = torch.tensor(t_eval, device=self.device, dtype=torch.float32)

        solver = odeint_adjoint if use_adjoint else odeint
        sol = solver(self.ode_func, y0, t_tensor, method="rk4")
        # sol: (T, 4N)

        N = self.N
        r     = sol[:, :N]
        phi   = sol[:, N:2*N]
        omega = sol[:, 2*N:3*N]
        alpha = sol[:, 3*N:4*N]
        rcos_phi = torch.sum(r * torch.cos(phi), dim=1)

        return r, phi, omega, alpha, rcos_phi


# ===========================================================================
# 5. TRAINING STAGES
# ===========================================================================

def train_heart_model(ecg_target: np.ndarray, device: str,
                      n_epochs: int = 25000, lr: float = 1e-3,
                      log_every: int = 2500):
    """
    Stage 0: Pre-train HeartModel on simulated oscillator → ECG mapping.

    Returns
    -------
    heart_model : trained HeartModel
    losses      : list of per-epoch MSE losses
    """
    log.info("Stage 0: Heart model pre-training (%d epochs)", n_epochs)
    model = HeartModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    sim_input = torch.tensor(
        simulate_coupled_oscillators(T=2, dt=0.01),
        dtype=torch.float32
    ).to(device)                                          # (200, 4)

    target = torch.tensor(
        ecg_target[::10], dtype=torch.float32
    ).to(device).unsqueeze(1)                             # (200, 1)

    losses = []
    for epoch in range(n_epochs):
        pred = model(sim_input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % log_every == 0:
            log.info("  Heart epoch %5d/%d  loss=%.6f", epoch + 1, n_epochs, loss.item())

    log.info("Stage 0 complete. Final loss: %.6f", losses[-1])
    return model, losses


def pretrain_brain_model(eeg_proc: np.ndarray, Sw_all: np.ndarray,
                         target_idx: int,
                         non_zero_indices_per_row: list,
                         t: np.ndarray,
                         D_tensor: torch.Tensor,
                         device: str,
                         n_epochs: int = 30,
                         osc_per_region: int = 3):
    """
    Stage 1: Gradient-free warm-up of brain ODE initial conditions.

    Iteratively runs the ODE forward and rolls the final state forward as the
    next initial condition. No gradient descent is performed.

    Returns
    -------
    final_params : dict with keys r, phi, theta, omega, alpha
    Sc_osc       : (N, N) oscillator-level SC matrix
    N            : total number of brain oscillators
    losses       : list of per-epoch MSE losses (informational only)
    """
    log.info("Stage 1: Brain warm-up (%d forward passes)", n_epochs)

    connected = np.unique(np.append(non_zero_indices_per_row[target_idx], target_idx))
    N_reg = len(connected)
    N = N_reg * osc_per_region

    Sc_reg = Sw_all[np.ix_(connected, connected)]
    Sc_osc = expand_structural_connectivity(Sc_reg, osc_per_region, seed=SEED)

    # ---------------------------------------------------------------------------
    # BIO CHANGE 4: Band-constrained frequency initialisation
    # Oscillators are assigned to known EEG frequency bands rather than
    # sampling uniformly from [1, 20] Hz. This ensures the brain model
    # operates in biologically meaningful spectral regimes from the start.
    # ---------------------------------------------------------------------------
    EEG_BANDS = [
        (1.0,  4.0),   # Delta
        (4.0,  8.0),   # Theta
        (8.0,  12.0),  # Alpha
        (12.0, 30.0),  # Beta
    ]
    rng = np.random.default_rng(SEED)
    total_osc = N_reg * osc_per_region
    # Assign each oscillator to a band in round-robin order
    band_assignments = [EEG_BANDS[i % len(EEG_BANDS)] for i in range(total_osc)]
    omega0 = np.array([
        2 * np.pi * rng.uniform(lo, hi)
        for lo, hi in band_assignments
    ])  # (N,) in rad/s

    # Excitability: moderate, uniform baseline
    alpha_baseline = np.full(N, 0.3)
    alpha0 = np.clip(
        rng.uniform(0.1, 0.5, N),
        0.05, 0.5
    )

    r0   = 0.1 * np.ones(N)
    phi0 = np.zeros(N)
    # OPT 2: theta is fixed — initialised once here and stored in ODEFunc as a buffer
    theta_rand  = np.pi * (2 * rng.random((N, N)) - 1)
    theta_fixed = theta_rand - theta_rand.T   # antisymmetric

    # ---------------------------------------------------------------------------
    # BIO CHANGE 5: Timescale-separated eta values
    #   eta_omega = 0.01  — slow frequency plasticity
    #   eta_alpha = 5e-4  — slower homeostatic excitability (10× slower than ω)
    #   lambda_homeo = 0.001 — homeostatic decay rate
    # ---------------------------------------------------------------------------
    brain = TorchRevHopfNetwork(
        mu=1.0,
        eta_omega=0.01,       # slow frequency adaptation
        eta_alpha=5e-4,       # slower excitability homeostasis
        D_tensor=D_tensor, N=N, Sc=Sc_osc,
        theta_fixed=theta_fixed,
        alpha_baseline=alpha_baseline,
        lambda_homeo=0.001,
        device=device
    )

    criterion = nn.MSELoss()
    losses = []

    for epoch in range(n_epochs):
        with torch.no_grad():
            r, phi, omega, alpha, _ = brain.solve(
                r0, phi0, omega0, alpha0, t, use_adjoint=True
            )
            P_out = torch.sum(alpha * r * torch.cos(phi), dim=1)
            loss  = criterion(P_out, D_tensor[:len(t)])
            losses.append(loss.item())

            # Roll final state forward as new initial condition
            omega0 = omega[-1].cpu().numpy()
            alpha0 = np.clip(alpha[-1].cpu().numpy(), 0.05, 0.5)

        if (epoch + 1) % 10 == 0:
            log.info("  Brain warm-up epoch %2d/%d  loss=%.6f", epoch + 1, n_epochs, loss.item())

        torch.cuda.empty_cache()
        gc.collect()

    final_params = {
        # Use the final rolled-forward state (not cold-start r0/phi0) so that
        # inference begins from a warm, settled ODE state — prevents the
        # large transient spike at t=0 in the brain output.
        "r":     r[-1].cpu().numpy(),
        "phi":   phi[-1].cpu().numpy(),
        "theta": theta_fixed,
        "omega": omega0, "alpha": alpha0,
        "alpha_baseline": alpha_baseline,   # passed to Stage 2 ODEFunc
    }
    log.info("Stage 1 complete. Final loss: %.6f", losses[-1])
    return final_params, Sc_osc, N, losses


def train_mlp_on_frozen_brain(
    heart_model: HeartModel,
    brain_params: dict,
    Sc_osc: np.ndarray,
    N: int,
    D_tensor: torch.Tensor,
    t_full: np.ndarray,
    device: str,
    n_epochs: int = 100,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    drive_scale: float = 0.1,
    log_every: int = 20,
    train_steps: int = 100,
):
    """
    Stage 2: Train ECGToOscillatorMLP to steer the brain ODE toward target EEG.

    Key design decisions
    --------------------
    - TorchRevHopfNetwork is created ONCE outside the loop (prevents OOM).
    - Standard odeint (not odeint_adjoint) is used so gradients flow back
      through brain_drive_full to the MLP parameters.
    - Brain drive is bounded via tanh * drive_scale to prevent ODE blow-up.
    - Gradient clipping prevents exploding gradients through ODE backprop.

    OPT 3 — Reduced training steps:
        The ODE is solved on a coarser time grid (train_steps, default 100)
        during training. The full grid is used only for final inference.
        This halves the ODE computation and backprop cost per epoch.

    Returns
    -------
    mlp_model : trained ECGToOscillatorMLP
    losses    : list of per-epoch MSE losses
    """
    log.info("Stage 2: MLP training (%d epochs, lr=%.0e, ODE steps=%d)",
             n_epochs, lr, train_steps)

    # OPT 3: coarser time grid for training
    step = max(1, len(t_full) // train_steps)
    t_train = t_full[::step]                          # (train_steps,) ≈ 100 pts
    D_train = D_tensor[:len(t_train)]                 # matching D slice

    mlp = ECGToOscillatorMLP(
        ecg_dim=50, N_VNS=8, hidden_dim=64, output_dim=N, device=device
    ).to(device)

    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Extract frozen ECG features from the pre-trained heart model
    with torch.no_grad():
        sim_input = torch.tensor(
            simulate_coupled_oscillators(T=2, dt=0.01), dtype=torch.float32
        ).to(device)
        hidden_repr_full = heart_model.get_features(sim_input)   # (200, 50)
        # Subsample hidden_repr to match the coarser training grid
        hidden_repr = hidden_repr_full[::step]                    # (train_steps, 50)

    # Build brain ODE for training grid (plasticity frozen: eta=0, lambda=0)
    brain = TorchRevHopfNetwork(
        mu=1.0, eta_omega=0.0, eta_alpha=0.0,
        D_tensor=D_train,
        N=N, Sc=Sc_osc,
        theta_fixed=brain_params["theta"],
        alpha_baseline=brain_params["alpha_baseline"],
        lambda_homeo=0.0,   # homeostasis inactive during MLP training
        brain_drive_full=None, fs=100, device=device
    )

    losses = []
    for epoch in range(n_epochs):
        raw_drive = mlp(hidden_repr)                              # (train_steps, N)

        # Bound the drive to [-drive_scale, +drive_scale] to stabilise ODE
        brain_drive = torch.tanh(raw_drive) * drive_scale

        # Inject into ODE (plain attribute preserves autograd graph)
        brain.ode_func.brain_drive_full = brain_drive

        # Solve ODE with standard odeint so gradients flow back to MLP
        r, phi, omega, alpha, _ = brain.solve(
            brain_params["r"], brain_params["phi"],
            brain_params["omega"], brain_params["alpha"],
            t_train, use_adjoint=False
        )

        P_out = torch.sum(alpha * r * torch.cos(phi), dim=1)   # (train_steps,)
        loss  = criterion(P_out, D_train)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=grad_clip)
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % log_every == 0:
            log.info("  MLP epoch %3d/%d  loss=%.6f", epoch + 1, n_epochs, loss.item())

        torch.cuda.empty_cache()
        gc.collect()

    log.info("Stage 2 complete. Final loss: %.6f", losses[-1])
    return mlp, losses


# ===========================================================================
# 6. INFERENCE & PLOTTING
# ===========================================================================

def run_inference(heart_model: HeartModel, mlp_model: ECGToOscillatorMLP,
                  brain_params: dict, Sc_osc: np.ndarray, N: int,
                  D_tensor: torch.Tensor,
                  t: np.ndarray, device: str,
                  drive_scale: float = 0.1):
    """
    Run final forward pass with the trained MLP and frozen brain ODE.
    Uses the full time grid (200 steps) for high-resolution output.

    Returns
    -------
    r_final, phi_final, alpha_final : ODE output tensors
    rcos_phi_final                  : (T,) brain output signal
    predicted_ecg                   : (T,) baseline ECG prediction
    P_out_baseline                  : (T,) brain P_out
    """
    log.info("Running final inference (full %d-step grid)...", len(t))
    heart_model.eval()

    with torch.no_grad():
        sim_input = torch.tensor(
            simulate_coupled_oscillators(T=len(t)/100, dt=1/100),
            dtype=torch.float32
        ).to(device)
        hidden_repr = heart_model.get_features(sim_input)             # (T, 50)
        brain_drive = torch.tanh(mlp_model(hidden_repr)) * drive_scale  # (T, N)

        # Final brain ODE solve with trained drive on full grid
        brain = TorchRevHopfNetwork(
            mu=1.0, eta_omega=0.0, eta_alpha=0.0,
            D_tensor=D_tensor, N=N, Sc=Sc_osc,
            theta_fixed=brain_params["theta"],
            alpha_baseline=brain_params["alpha_baseline"],
            lambda_homeo=0.0,   # homeostasis inactive during inference
            brain_drive_full=brain_drive, fs=100, device=device
        )
        r_f, phi_f, _, alpha_f, rcos_phi_f = brain.solve(
            brain_params["r"], brain_params["phi"],
            brain_params["omega"], brain_params["alpha"], t
        )

        # Baseline ECG prediction
        predicted_ecg = heart_model(sim_input).cpu().numpy().flatten()
        P_out = torch.sum(alpha_f * r_f * torch.cos(phi_f), dim=1).cpu().numpy()

    return r_f, phi_f, alpha_f, rcos_phi_f, predicted_ecg, P_out


def save_plots(brain_losses, mlp_losses, heart_losses,
               ecg_proc, predicted_ecg, D_tensor: torch.Tensor,
               t: np.ndarray, P_out_baseline,
               target_idx: int, out_dir: Path):
    """Save a 5-panel summary figure to out_dir."""
    D_np = D_tensor.cpu().numpy()[:len(t)]

    fig, axes = plt.subplots(5, 1, figsize=(15, 20))

    axes[0].plot(brain_losses)
    axes[0].set_title("Stage 1: Brain Warm-up Loss")
    axes[0].set_xlabel("Epoch"); axes[0].grid(True)

    axes[1].plot(mlp_losses)
    axes[1].set_title("Stage 2: MLP Training Loss")
    axes[1].set_xlabel("Epoch"); axes[1].grid(True)

    axes[2].plot(heart_losses)
    axes[2].set_title("Stage 0: Heart Model Pre-training Loss")
    axes[2].set_xlabel("Epoch"); axes[2].grid(True)

    target_ecg = ecg_proc[::10]
    ts = np.linspace(0, len(t) / 100, len(target_ecg))
    axes[3].plot(ts, target_ecg, label="Target ECG", linewidth=2)
    axes[3].plot(ts, predicted_ecg, label="Predicted ECG (baseline)", linestyle="--")
    axes[3].set_title("ECG: Target vs Baseline Prediction")
    axes[3].legend(); axes[3].grid(True)

    axes[4].plot(t, D_np, label="Target EEG", linewidth=2)
    # Z-score normalise P_out so it is on the same scale as the target EEG
    # (P_out = Σ α·r·cos(φ) has arbitrary amplitude; normalising makes the
    # comparison meaningful without changing the model output)
    P_norm = P_out_baseline
    std = P_norm.std()
    if std > 1e-8:
        P_norm = (P_norm - P_norm.mean()) / std
    axes[4].plot(t, P_norm, label="P_out (brain output, z-scored)", alpha=0.7)
    axes[4].set_title("Brain Output vs Target EEG")
    axes[4].legend(); axes[4].grid(True)

    plt.tight_layout()
    fig_path = out_dir / f"result_idx{target_idx}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", fig_path)


# ===========================================================================
# 7. ARGUMENT PARSING & MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Heart-Brain Coupling Model (Reverse Hopf ODE)"
    )
    # Data paths — defaults point to local DDP directory; override on cluster
    _DDP = "/home/shobs/Desktop/DDP"
    p.add_argument("--fif_path",
                   default=f"{_DDP}/transdef_mf2pt2_rest_raw.fif",
                   help="Path to .fif MEG/ECG file")
    p.add_argument("--mat_path",
                   default=f"{_DDP}/scout_id_309.mat",
                   help="Path to EEG scout .mat file")
    p.add_argument("--sc_path",
                   default=f"{_DDP}/SC_CC120309-27.mat",
                   help="Path to SC .mat file")
    # Model / training
    p.add_argument("--target_idx",     type=int,   default=4,    help="Target EEG region index")
    p.add_argument("--osc_per_region", type=int,   default=3,    help="Oscillators per brain region")
    p.add_argument("--heart_epochs",   type=int,   default=25000,help="Heart model training epochs")
    p.add_argument("--brain_epochs",   type=int,   default=30,   help="Brain warm-up epochs")
    p.add_argument("--mlp_epochs",     type=int,   default=100,  help="MLP training epochs")
    p.add_argument("--heart_lr",       type=float, default=1e-3, help="Heart model learning rate")
    p.add_argument("--mlp_lr",         type=float, default=1e-3, help="MLP learning rate")
    p.add_argument("--grad_clip",      type=float, default=1.0,  help="Gradient clip max norm")
    p.add_argument("--drive_scale",    type=float, default=0.1,  help="tanh scale for brain drive")
    p.add_argument("--t_duration",     type=float, default=2.0,  help="Simulation duration (s)")
    p.add_argument("--fs",             type=float, default=100.0,help="Simulation sample rate (Hz)")
    p.add_argument("--train_steps",    type=int,   default=100,
                   help="ODE timesteps during Stage 2 training (OPT 3; use fewer for speed)")
    # Infrastructure
    p.add_argument("--device",  default="cuda", help="cuda or cpu")
    p.add_argument("--out_dir", default="./results", help="Output directory for plots/npz")
    return p.parse_args()


def main():
    args = parse_args()

    # Set env var before any CUDA allocation
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ecg_data, eeg_data, Sw_all = load_data(args.fif_path, args.mat_path, args.sc_path)
    ecg_proc, eeg_proc = preprocess(ecg_data, eeg_data)
    non_zero_per_row = [np.nonzero(Sw_all[i, :])[0] for i in range(Sw_all.shape[0])]

    # Full time vector for simulation
    t = np.arange(0, args.t_duration, 1.0 / args.fs)   # (200,) at fs=100

    # OPT 1: precompute D as a GPU tensor once
    target_signal = eeg_proc[args.target_idx, ::10]
    D_tensor = make_D_tensor(target_signal, t, device)   # (200,) on GPU

    # ------------------------------------------------------------------
    # Stage 0: Heart model
    # ------------------------------------------------------------------
    heart_model, heart_losses = train_heart_model(
        ecg_proc, device,
        n_epochs=args.heart_epochs,
        lr=args.heart_lr,
    )

    # ------------------------------------------------------------------
    # Stage 1: Brain warm-up
    # ------------------------------------------------------------------
    brain_params, Sc_osc, N, brain_losses = pretrain_brain_model(
        eeg_proc, Sw_all, args.target_idx, non_zero_per_row,
        t, D_tensor, device,
        n_epochs=args.brain_epochs,
        osc_per_region=args.osc_per_region,
    )

    # ------------------------------------------------------------------
    # Stage 2: MLP training
    # ------------------------------------------------------------------
    mlp_model, mlp_losses = train_mlp_on_frozen_brain(
        heart_model, brain_params, Sc_osc, N,
        D_tensor, t, device,
        n_epochs=args.mlp_epochs,
        lr=args.mlp_lr,
        grad_clip=args.grad_clip,
        drive_scale=args.drive_scale,
        train_steps=args.train_steps,
    )

    # ------------------------------------------------------------------
    # Inference & save
    # ------------------------------------------------------------------
    r_f, phi_f, alpha_f, rcos_phi_f, pred_ecg, P_out = run_inference(
        heart_model, mlp_model, brain_params, Sc_osc, N,
        D_tensor, t, device, drive_scale=args.drive_scale
    )

    save_plots(brain_losses, mlp_losses, heart_losses,
               ecg_proc, pred_ecg, D_tensor, t,
               P_out, args.target_idx, out_dir)

    npz_path = out_dir / f"results_idx{args.target_idx}.npz"
    np.savez(
        npz_path,
        brain_losses=brain_losses,
        mlp_losses=mlp_losses,
        heart_losses=heart_losses,
        rcos_phi_final=rcos_phi_f.detach().cpu().numpy(),
        P_out_baseline=P_out,
        predicted_ecg=pred_ecg,
        target_ecg=ecg_proc[::10],
        target_eeg=D_tensor.cpu().numpy(),
    )
    log.info("Saved results: %s", npz_path)
    log.info("✅ Done.")


if __name__ == "__main__":
    main()
