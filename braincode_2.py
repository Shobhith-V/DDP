import h5py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint
import gc
import mne
import time
import sys
import os
import math

# =============================================================================
# DATA LOADING
# =============================================================================
new_raw = "/home/shobs/Desktop/DDP/transdef_mf2pt2_rest_raw.fif"
raw = mne.io.read_raw_fif(new_raw, preload=True)
data, times = raw[322, 2000:4000]
ecg_data = -data[0]   # shape: (2000,)

mat = loadmat("/home/shobs/Desktop/DDP/scout_id_309.mat")
eeg_data = mat['Value'][:, 2000:4000]   # shape: (num_regions, 2000)

sc_data = loadmat("/home/shobs/Desktop/DDP/SC_CC120309-27.mat")
sc_matrix = sc_data['sc']
max_val = np.max(sc_matrix)
Sw_all = (sc_matrix / max_val) * 0.01 if max_val > 0 else sc_matrix
non_zero_indices_per_row = [np.nonzero(Sw_all[i, :])[0] for i in range(Sw_all.shape[0])]

# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_signal(signal, fs=1000, lowcut=1.5, highcut=20):
    detrended = detrend(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, detrended)
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    return normalized

# ecg_data is 1D (2000,)
ecg_processed = preprocess_signal(ecg_data, fs=1000, lowcut=1.5, highcut=20)  # shape: (2000,)

# eeg_data is 2D (num_regions, 2000)
eeg_processed = np.array([preprocess_signal(row, fs=1000, lowcut=0.5, highcut=30) for row in eeg_data])

# =============================================================================
# SIMULATION UTILITY
# =============================================================================
def simulate_coupled_oscillators(T=10, dt=1/1000, alpha=1, omega1=5.01, omega2=5.1,
                                  A_init=0.0001, theta_init=3.14, n=1.0, modulation=None):
    """
    Simulates a pair of coupled Hopf oscillators.
    modulation: if provided, shape (N_steps, >=2) — first two columns modulate oscillator 1 and 2.
    Returns array of shape (N_steps, 4): [r1*cos(phi1), r1*sin(phi1), r2*cos(phi2), r2*sin(phi2)]
    """
    N = int(T / dt)
    r1, r2, phi1, phi2 = 1.0, 1.0, 0.0, 0.0
    A12, A21 = A_init, A_init
    theta12, theta21 = theta_init, theta_init

    R1, R2, Phi1, Phi2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        R1[i], R2[i], Phi1[i], Phi2[i] = r1, r2, phi1, phi2

        coupling12 = A12 * r2 * np.cos(theta12 + n * (phi2 - phi1))
        coupling21 = A21 * r1 * np.cos(theta21 + n * (phi1 - phi2))

        # BUG FIX: modulation may have more than 2 columns (shape: N_steps x N_osc).
        # We safely index column 0 and column 1.
        mod1 = modulation[i, 0] if (modulation is not None and i < len(modulation)) else 0.0
        mod2 = modulation[i, 1] if (modulation is not None and i < len(modulation)) else 0.0

        dr1 = alpha * r1 - r1**3 + coupling12 + 0.1 * mod1
        dr2 = alpha * r2 - r2**3 + coupling21 + 0.1 * mod2

        dphi1 = omega1 + A12 * r2 / r1 * np.sin(theta12 + n * (phi2 - phi1))
        dphi2 = omega2 + A21 * r1 / r2 * np.sin(theta21 + n * (phi1 - phi2))

        r1 += dr1 * dt
        r2 += dr2 * dt
        phi1 += dphi1 * dt
        phi2 += dphi2 * dt

    return np.stack((R1 * np.cos(Phi1), R1 * np.sin(Phi1), R2 * np.cos(Phi2), R2 * np.sin(Phi2)), axis=1)


def get_random_frequencies(num_regions, osc_per_region, low=1, high=20, seed=None):
    if seed is not None:
        np.random.seed(seed)
    total_oscillators = num_regions * osc_per_region
    freqs_hz = np.random.uniform(low, high, total_oscillators)
    return 2 * np.pi * freqs_hz


def expand_structural_connectivity(Sc_region, osc_per_region, intra_value=0.0001, seed=None):
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


def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class OscillatorLayer(nn.Module):
    """
    Residual Hopf Oscillator Layer:
    ECG-derived features → complex forcing → Hopf dynamics → Cartesian state output
    """

    def __init__(self, N_osc=64, T=2.0, fs=100, min_freq=2.0, max_freq=10.0,
                 input_scaler=2.0, train_omegas=True, device="cpu"):
        super().__init__()

        self.N_osc = N_osc
        self.num_steps = int(T * fs)
        self.dt = 1.0 / fs
        self.input_scaler = input_scaler

        # Radial parameters
        self.register_buffer("mu0", torch.tensor(1.0))
        self.beta = 1.0

        # Frequency parameters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.train_omegas = train_omegas

        if train_omegas:
            self.omegas = nn.Parameter(torch.randn(1, N_osc))
        else:
            omega = torch.rand(1, N_osc)
            self.register_buffer("omegas", omega)

        # Initial states (buffers, not parameters)
        self.register_buffer("initial_r", torch.ones(1, N_osc) * 0.1)
        self.register_buffer("initial_phi", torch.zeros(1, N_osc))

        self.to(device)

    def forward(self, input_features):
        """
        input_features: (batch_size, N_osc)
        returns: (batch_size, 2*N_osc)
        """
        batch_size = input_features.shape[0]

        r = self.initial_r.repeat(batch_size, 1)
        phi = self.initial_phi.repeat(batch_size, 1)

        omega_range = self.max_freq - self.min_freq
        omegas = torch.sigmoid(self.omegas) * omega_range + self.min_freq
        omegas = omegas * (2 * math.pi)       # convert Hz → rad/s
        omegas = omegas.repeat(batch_size, 1)

        for _ in range(self.num_steps):
            input_r   = self.input_scaler * input_features * torch.cos(phi)
            input_phi = self.input_scaler * input_features * torch.sin(phi)

            dr_dt   = (self.mu0 - self.beta * r**2) * r + input_r
            dphi_dt = omegas - input_phi

            r   = r   + dr_dt   * self.dt
            phi = phi + dphi_dt * self.dt

        z_r = r * torch.cos(phi)
        z_i = r * torch.sin(phi)
        return torch.cat([z_r, z_i], dim=-1)


class HeartModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=100, feature_dim=50, output_dim=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.output_layer = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.output_layer(features)

    def get_features(self, x):
        return self.feature_extractor(x)


class ECGToOscillatorMLP(nn.Module):
    """
    ECG → MLP → Residual Hopf Oscillator → MLP → Brain drive [N]
    """

    def __init__(self, ecg_dim=50, N_VNS=128, hidden_dim=64, output_dim=16, device="cuda"):
        super().__init__()
        self.device = device

        self.pre_osc = nn.Sequential(
            nn.Linear(ecg_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, N_VNS)
        )

        self.osc_layer = OscillatorLayer(
            N_osc=N_VNS, T=2.0, fs=100, min_freq=2.0, max_freq=10.0,
            input_scaler=2.0, train_omegas=True, device=device
        )

        self.post_osc = nn.Sequential(
            nn.Linear(N_VNS * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.to(device)

    def forward(self, ecg_features):
        """
        ecg_features: (batch_size, ecg_dim) or (ecg_dim,)
        returns: (batch_size, output_dim)
        """
        if ecg_features.dim() == 1:
            ecg_features = ecg_features.unsqueeze(0)

        ecg_features = ecg_features.to(self.device)
        pre_drive  = self.pre_osc(ecg_features)   # (B, N_VNS)
        osc_out    = self.osc_layer(pre_drive)     # (B, 2*N_VNS)
        brain_drive = self.post_osc(osc_out)       # (B, output_dim)
        return brain_drive


class ODEFuc(nn.Module):
    def __init__(self, mu, eta_theta, eta_omega, eta_alpha,
                 D_function, N, Sc, mlp_model=None, hidden_repr=None):
        super().__init__()
        self.mu = mu
        self.eta_theta = eta_theta
        self.eta_omega = eta_omega
        self.eta_alpha = eta_alpha
        self.D_function = D_function
        self.N = N
        self.register_buffer('Sc', Sc)
        self.mlp_model = mlp_model
        self.hidden_repr = hidden_repr

    def forward(self, t, state):
        N = self.N
        r     = state[:N]
        phi   = state[N:2*N]
        theta = state[2*N:2*N + N**2].view(N, N)
        omega = state[2*N + N**2:3*N + N**2]
        alpha = state[3*N + N**2:4*N + N**2]

        omega_safe = torch.clamp(omega, 2 * np.pi * 0.5, 2 * np.pi * 20)
        r          = torch.clamp(r, 1e-1, 2.0)
        alpha      = torch.clamp(alpha, -1.0, 1.0)
        r_safe     = torch.clamp(r, 1e-5, 10.0)

        phase_diff = torch.clamp(
            phi[None, :] / omega_safe[None, :] - phi[:, None] / omega_safe[:, None]
            + theta / (omega_safe[:, None] * omega_safe[None, :]),
            -1e2, 1e2
        )

        D = torch.tensor(self.D_function(t.item()), device=state.device, dtype=state.dtype)
        P = torch.sum(alpha * r * torch.cos(phi))
        e = D - P

        # ECG drive from the MLP
        ecg_input = torch.zeros(N, device=state.device, dtype=state.dtype)
        if (self.mlp_model is not None) and (self.hidden_repr is not None):
            t_idx = min(int(t.item() * 100), self.hidden_repr.shape[0] - 1)
            ecg_features = self.hidden_repr[t_idx].to(device=state.device, dtype=state.dtype)
            # ecg_features is 1D (feature_dim,); ECGToOscillatorMLP handles unsqueezing internally
            ecg_out = self.mlp_model(ecg_features)   # → (1, N) after unsqueeze inside MLP
            ecg_input = torch.clamp(ecg_out.squeeze(), 0.01, 5.0)
            # If N==1, squeeze() returns a scalar; ensure it stays 1-D
            if ecg_input.dim() == 0:
                ecg_input = ecg_input.unsqueeze(0)

        coupling_r   = torch.sum(torch.abs(self.Sc) * r[None, :] * torch.cos(phase_diff), dim=1)
        drdt         = (self.mu - r**2) * r + coupling_r + e * torch.cos(phi) + ecg_input

        coupling_phi = torch.sum(torch.abs(self.Sc) * (r[None, :] / r_safe[:, None]) * torch.sin(phase_diff), dim=1)
        dphidt       = omega + coupling_phi - (e / r_safe) * torch.sin(phi)

        dthetadt = self.eta_theta * torch.sin(phase_diff) * torch.abs(self.Sc)
        domegadt = -self.eta_omega * e * torch.sin(phi)
        dalphadt = self.eta_alpha * e * r * torch.cos(phi)

        drdt     = torch.clamp(drdt,     -1e2, 1e2)
        dphidt   = torch.clamp(dphidt,   -1e2, 1e2)
        dthetadt = torch.clamp(dthetadt, -1e2, 1e2)
        domegadt = torch.clamp(domegadt, -1e2, 1e2)
        dalphadt = torch.clamp(dalphadt, -1e2, 1e2)

        return torch.cat([drdt.flatten(), dphidt.flatten(), dthetadt.flatten(),
                          domegadt.flatten(), dalphadt.flatten()])


class TorchRevHopfNetwork:
    def __init__(self, mu, eta_omega, eta_alpha, eta_theta,
                 D_function, N, Sc, mlp_model, hidden_repr, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.N = N
        self.ode_func = ODEFuc(
            mu=mu, eta_theta=eta_theta, eta_omega=eta_omega, eta_alpha=eta_alpha,
            D_function=D_function, N=N,
            Sc=torch.tensor(Sc, device=self.device, dtype=torch.float32),
            mlp_model=mlp_model,
            hidden_repr=hidden_repr.to(self.device) if hidden_repr is not None else None
        ).to(self.device)

    def solve(self, r0, phi0, theta0, omega0, alpha0, t_eval):
        dtype = torch.float32
        y0 = torch.tensor(
            np.concatenate([r0, phi0, theta0.flatten(), omega0, alpha0]),
            device=self.device, dtype=dtype
        )
        t_eval_tensor = torch.tensor(t_eval, device=self.device, dtype=dtype)

        sol = odeint_adjoint(self.ode_func, y0, t_eval_tensor, method='rk4', rtol=1e-5, atol=1e-7)

        N = self.N
        r     = sol[:, :N]
        phi   = sol[:, N:2*N]
        theta = sol[:, 2*N:2*N + N**2].view(-1, N, N)
        omega = sol[:, 2*N + N**2:3*N + N**2]
        alpha = sol[:, 3*N + N**2:4*N + N**2]

        rcos_phi = torch.sum(r * torch.cos(phi), dim=1)   # brain output signal

        return r, phi, theta, omega, alpha, rcos_phi

# =============================================================================
# TRAINING STAGES
# =============================================================================

def train_heart_model(ecg_target_signal, device):
    """
    Stage 0: Pre-train HeartModel to map coupled-oscillator states → ECG.
    ecg_target_signal: 1D numpy array of length 2000 (at 1000 Hz).
    """
    print("--- Starting Heart Model Pre-training ---")
    heart_model = HeartModel().to(device)
    optimizer = optim.Adam(heart_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Simulate at dt=0.01 → 200 steps for T=2s  (same length as ecg[::10])
    sim_osc_input = torch.tensor(
        simulate_coupled_oscillators(T=2, dt=0.01), dtype=torch.float32
    ).to(device)  # (200, 4)

    # BUG FIX: ecg_target_signal is 1D (2000,); downsample to match sim length
    ecg_target = torch.tensor(
        ecg_target_signal[::10], dtype=torch.float32
    ).to(device).unsqueeze(1)  # (200, 1)

    for epoch in range(25000):
        optimizer.zero_grad()                        # zero grad BEFORE forward pass
        predicted_ecg = heart_model(sim_osc_input)   # (200, 1)
        loss = criterion(predicted_ecg, ecg_target)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2500 == 0:
            print(f"  Heart Epoch {epoch+1:5d}, Loss: {loss.item():.6f}")

    print("--- Heart Pre-training Finished ---")
    return heart_model


def pre_train_brain_model(eeg_processed, Sw_all, target_idx,
                          non_zero_indices_per_row, t, D_function, device):
    """
    Stage 1: Warm-up brain Hopf network (no MLP) using sequential simulation.
    Each epoch refines the initial conditions for the next epoch.
    """
    print("\n--- Stage 1: Brain Pre-training (warm-up) ---")
    connected_indices = np.unique(np.append(non_zero_indices_per_row[target_idx], target_idx))
    N_reduced_regions = len(connected_indices)
    osc_per_region = 3
    N = N_reduced_regions * osc_per_region

    Sc_reduced_regional = Sw_all[np.ix_(connected_indices, connected_indices)]
    Sc_reduced_osc = expand_structural_connectivity(Sc_reduced_regional, osc_per_region, seed=42)

    omega_full = get_random_frequencies(68, osc_per_region, low=1, high=20, seed=42)
    alpha_full = np.random.uniform(0.1, 0.7, 68 * osc_per_region)

    omega0 = np.concatenate([
        omega_full[i * osc_per_region:(i + 1) * osc_per_region] for i in connected_indices
    ])
    alpha0 = np.clip(np.concatenate([
        alpha_full[i * osc_per_region:(i + 1) * osc_per_region] for i in connected_indices
    ]), 0.05, 0.5)
    r0 = 0.1 * np.ones(N)
    phi0 = np.zeros(N)
    theta_random = np.pi * (2 * np.random.rand(N, N) - 1)
    theta0 = theta_random - theta_random.T

    criterion = nn.MSELoss()
    D_true = torch.tensor(D_function(t), device=device, dtype=torch.float32)
    losses = []

    for epoch in range(30):
        model = TorchRevHopfNetwork(
            mu=1.0, eta_omega=0.05, eta_alpha=0.005, eta_theta=0.05,
            D_function=D_function, N=N, Sc=Sc_reduced_osc,
            mlp_model=None, hidden_repr=None, device=device
        )
        # Intentional: no gradient update; we roll forward and carry over final state
        with torch.no_grad():
            r, phi, theta, omega, alpha, _ = model.solve(r0, phi0, theta0, omega0, alpha0, t)
            P_out = torch.sum(alpha * r * torch.cos(phi), dim=1)
            loss = criterion(P_out, D_true)
            losses.append(loss.item())

            # Carry final state forward as next initial condition
            theta0 = theta[-1].cpu().numpy()
            omega0 = omega[-1].cpu().numpy()
            alpha0 = alpha[-1].cpu().numpy()

        if (epoch + 1) % 10 == 0:
            print(f"  Brain Epoch {epoch+1:3d}/30, Loss: {loss.item():.6f}")

    final_params = {
        'r': r0, 'phi': phi0,
        'theta': theta0, 'omega': omega0, 'alpha': alpha0
    }
    return final_params, Sc_reduced_osc, N, losses


def train_mlp_on_frozen_brain(trained_heart_model, initial_brain_params,
                               Sc_reduced_osc, N, D_function, t, device):
    """
    Stage 2: Train ECGToOscillatorMLP (ECG→Hopf oscillators→brain drive)
    while the brain ODE etas are fixed at 0 (frozen learning).
    BUG FIX: optimizer.zero_grad() moved BEFORE model.solve() (the forward pass).
    """
    print("\n--- Stage 2: ECG → OscillatorLayer → Brain Training ---")

    ecg_to_osc_mlp = ECGToOscillatorMLP(
        ecg_dim=50, N_VNS=128, hidden_dim=64, output_dim=N, device=device
    ).to(device)

    optimizer = optim.Adam(ecg_to_osc_mlp.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    with torch.no_grad():
        sim_osc_input = torch.tensor(
            simulate_coupled_oscillators(T=2, dt=0.01), dtype=torch.float32
        ).to(device)
        ecg_features = trained_heart_model.get_features(sim_osc_input)  # (200, 50)

    D_true = torch.tensor(D_function(t), device=device, dtype=torch.float32)
    losses = []

    for epoch in range(100):
        # BUG FIX: zero_grad BEFORE the forward pass so previous iteration's
        #          gradients don't contaminate this forward pass.
        optimizer.zero_grad()

        model = TorchRevHopfNetwork(
            mu=1.0, eta_omega=0.0, eta_alpha=0.0, eta_theta=0.0,
            D_function=D_function, N=N, Sc=Sc_reduced_osc,
            mlp_model=ecg_to_osc_mlp, hidden_repr=ecg_features, device=device
        )

        r, phi, theta, omega, alpha, _ = model.solve(
            initial_brain_params['r'], initial_brain_params['phi'],
            initial_brain_params['theta'], initial_brain_params['omega'],
            initial_brain_params['alpha'], t
        )

        P_out = torch.sum(alpha * r * torch.cos(phi), dim=1)
        loss = criterion(P_out, D_true)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"  ECG→Osc→Brain Epoch {epoch+1:3d}/100, Loss: {loss.item():.6f}")

    return ecg_to_osc_mlp, losses


# =============================================================================
# BUG FIX: DEFINE train_feedback_loop — was called but never defined.
# =============================================================================
class FeedbackMLP(nn.Module):
    """
    Brain signal (rcos_phi) → MLP → modulation for the two coupled oscillators.
    Output dim = 2 so it can directly index modulation[:, 0] and modulation[:, 1].
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (T,) or (T, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)   # (T, 2)


def train_feedback_loop(trained_heart_model, rcos_phi_numpy, ecg_target_signal,
                        T=2.0, dt=1/100, device="cpu", num_epochs=5000):
    """
    Stage 3: Train a FeedbackMLP mapping brain signal → oscillator modulation,
    so that the HeartModel output (ECG) matches the real ECG after brain feedback.

    rcos_phi_numpy : 1D numpy array, shape (T_steps,)  — brain output at 100 Hz
    ecg_target_signal : 1D numpy array, shape (2000,)   — real ECG at 1000 Hz
    """
    print("\n--- Stage 3: Brain → Feedback → Heart Training ---")

    feedback_mlp = FeedbackMLP(input_dim=1, hidden_dim=64, output_dim=2).to(device)

    # Only fine-tune the heart model; keep feedback_mlp as the main learner
    heart_optimizer    = optim.Adam(trained_heart_model.parameters(), lr=1e-4)
    feedback_optimizer = optim.Adam(feedback_mlp.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Target ECG downsampled to 100 Hz to match rcos_phi
    ecg_target = torch.tensor(ecg_target_signal[::10], dtype=torch.float32).to(device)  # (200,)
    rcos_phi_t = torch.tensor(rcos_phi_numpy, dtype=torch.float32).to(device)           # (200,)

    losses = []
    for epoch in range(num_epochs):
        heart_optimizer.zero_grad()
        feedback_optimizer.zero_grad()

        # Brain → modulation for the two Hopf oscillators (one per step)
        modulation_t = feedback_mlp(rcos_phi_t)   # (200, 2)

        # Build modulated oscillator trajectory — keep in numpy for simulate_coupled_oscillators
        modulation_np = modulation_t.detach().cpu().numpy()   # (200, 2)
        sim_osc = torch.tensor(
            simulate_coupled_oscillators(T=T, dt=dt, modulation=modulation_np),
            dtype=torch.float32
        ).to(device)  # (200, 4)

        # Heart model predicts ECG from modulated oscillator state
        predicted_ecg = trained_heart_model(sim_osc).squeeze()   # (200,)
        loss = criterion(predicted_ecg, ecg_target)

        loss.backward()
        heart_optimizer.step()
        feedback_optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 500 == 0:
            print(f"  Feedback Epoch {epoch+1:5d}/{num_epochs}, Loss: {loss.item():.6f}")

    print("--- Feedback Training Finished ---")
    return trained_heart_model, feedback_mlp, losses

# =============================================================================
# MAIN PIPELINE
# =============================================================================
target_indices = [4]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {device} ---")

# ---------- Stage 0: Heart pre-training ----------
trained_heart_model = train_heart_model(ecg_processed, device)

with torch.no_grad():
    simulated_ecg_input = torch.tensor(
        simulate_coupled_oscillators(T=2, dt=0.01), dtype=torch.float32
    ).to(device)
    hidden_repr = trained_heart_model.get_features(simulated_ecg_input)  # (200, 50)

target_idx = target_indices[0]

t_duration = 2
fs = 100
t = np.arange(0, t_duration, 1 / fs)          # (200,)

# EEG target: region target_idx, downsampled 1000→100 Hz
target_signal = eeg_processed[target_idx, ::10]  # (200,)
D_function = interp1d(t, target_signal, kind='linear', bounds_error=False, fill_value=0.0)

# ---------- Stage 1: Brain warm-up ----------
final_brain_params, Sc_reduced_osc, N, brain_losses = pre_train_brain_model(
    eeg_processed, Sw_all, target_idx, non_zero_indices_per_row, t, D_function, device
)

# ---------- Stage 2: ECG→OscillatorLayer→Brain ----------
trained_mlp_model, mlp_losses = train_mlp_on_frozen_brain(
    trained_heart_model, final_brain_params, Sc_reduced_osc, N, D_function, t, device
)

# ---------- Extract brain rcos_phi for feedback ----------
print("\n--- Extracting Brain rcos_phi for Feedback ---")
model_final = TorchRevHopfNetwork(
    mu=1.0, eta_omega=0.0, eta_alpha=0.0, eta_theta=0.0,
    D_function=D_function, N=N, Sc=Sc_reduced_osc,
    mlp_model=trained_mlp_model, hidden_repr=hidden_repr, device=device
)
r_final, phi_final, theta_final, omega_final, alpha_final, rcos_phi_final = model_final.solve(
    final_brain_params['r'], final_brain_params['phi'], final_brain_params['theta'],
    final_brain_params['omega'], final_brain_params['alpha'], t
)
print(f"rcos_phi_final shape: {rcos_phi_final.shape}, "
      f"range: [{rcos_phi_final.min():.3f}, {rcos_phi_final.max():.3f}]")

# ---------- Stage 3: Brain → Feedback → Heart ----------
trained_heart_model, trained_feedback_mlp, feedback_losses = train_feedback_loop(
    trained_heart_model,
    rcos_phi_final.detach().cpu().numpy(),
    ecg_processed,
    T=t_duration, dt=1/fs, device=device, num_epochs=5000
)

# =============================================================================
# FINAL PREDICTIONS
# =============================================================================
print("\n--- Final Predictions ---")
trained_heart_model.eval()
trained_feedback_mlp.eval()

with torch.no_grad():
    # Baseline ECG (no feedback)
    sim_osc_baseline = simulate_coupled_oscillators(T=t_duration, dt=1/fs)  # (200, 4)
    predicted_ecg_baseline = trained_heart_model(
        torch.tensor(sim_osc_baseline, dtype=torch.float32).to(device)
    ).cpu().numpy().flatten()   # (200,)

    # Feedback ECG — use FeedbackMLP to generate modulation
    rcos_phi_t = torch.tensor(rcos_phi_final.cpu().numpy(), dtype=torch.float32).to(device)
    feedback_output = trained_feedback_mlp(rcos_phi_t)        # (200, 2)
    modulation_np = feedback_output.cpu().numpy()             # (200, 2) — correct shape for simulate_coupled_oscillators
    sim_osc_feedback = simulate_coupled_oscillators(T=t_duration, dt=1/fs, modulation=modulation_np)
    predicted_ecg_feedback = trained_heart_model(
        torch.tensor(sim_osc_feedback, dtype=torch.float32).to(device)
    ).cpu().numpy().flatten()

    # Baseline brain output
    P_out_baseline = torch.sum(alpha_final * r_final * torch.cos(phi_final), dim=1).cpu().numpy()

# =============================================================================
# PLOTTING
# =============================================================================
fig, axes = plt.subplots(5, 1, figsize=(15, 20))

axes[0].plot(brain_losses)
axes[0].set_title('Stage 1: Brain Pre-training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].grid(True)

axes[1].plot(mlp_losses)
axes[1].set_title('Stage 2: ECG→Osc→Brain MLP Training Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE')
axes[1].grid(True)

axes[2].plot(feedback_losses)
axes[2].set_title('Stage 3: Brain→Feedback→Heart Loss')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MSE')
axes[2].grid(True)

# ECG comparison (all at 100 Hz → 200 points over 2 s)
target_ecg = ecg_processed[::10]    # (200,)
timesteps  = np.linspace(0, t_duration, len(target_ecg))

axes[3].plot(timesteps, target_ecg,            label='Target ECG',    linewidth=2)
axes[3].plot(timesteps, predicted_ecg_baseline, label='Baseline ECG', linestyle='--')
axes[3].plot(timesteps, predicted_ecg_feedback, label='Feedback ECG', linestyle=':')
axes[3].set_title('ECG Prediction: Baseline vs Brain Feedback')
axes[3].set_xlabel('Time (s)')
axes[3].legend()
axes[3].grid(True)

axes[4].plot(t, D_function(t),            label='Target EEG',    linewidth=2)
axes[4].plot(t, P_out_baseline,           label='P_out baseline', alpha=0.7)
axes[4].plot(t, rcos_phi_final.cpu().numpy(), label='Brain rcos_phi (MLP)', linestyle='--', linewidth=1.5)
axes[4].set_title('Brain Output: rcos_phi vs Target EEG')
axes[4].set_xlabel('Time (s)')
axes[4].legend()
axes[4].grid(True)

plt.tight_layout()
plt.savefig("/home/shobs/Desktop/DDP/hopf_brain_heart_results.png", dpi=150)
plt.show()
print("Done. Figure saved.")