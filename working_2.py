## SHOBHITH Sent me this code..

## I changed the code.. see the changes in the equation..
from torchdiffeq import odeint

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

# --- Debugging Flags ---
use_half_precision = False
debug_interval = 100

# ---------- DATA LOADING + PREPROCESSING ----------
try:
    file_new_raw = '/home/shobs/Desktop/DDP/transdef_mf2pt2_rest_raw.fif'
    raw = mne.io.read_raw_fif(file_new_raw, preload=False)
    data, times = raw[322, 2000:4000]
    ecg_data = -data[0]

    mat = loadmat("/home/shobs/Desktop/DDP/scout_id_309.mat")
    eeg_data = mat['Value'][:, 2000:4000]
    sc_data = loadmat('/home/shobs/Desktop/DDP/SC_CC120309-27.mat')
    sc_matrix = sc_data["sc"]

    max_val = np.max(sc_matrix)
    Sw_all = (sc_matrix / max_val) * 0.01 if max_val > 0 else sc_matrix

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    sys.exit()

non_zero_indices_per_row = [np.nonzero(Sw_all[i, :])[0] for i in range(Sw_all.shape[0])]

def power_coupling_terms(r, phi, omega, theta):

    omega_safe = torch.clamp(
        omega,
        2*np.pi*0.5,
        2*np.pi*20
    )

    rho = omega_safe[:,None] / omega_safe[None,:]
    rho = torch.clamp(rho, 0.1, 10.0)

    phase_diff = rho * phi[None,:] - phi[:,None] + theta
    phase_diff = torch.clamp(phase_diff, -50.0, 50.0)

    r_safe = torch.clamp(r, 1e-5, 10.0)

    log_r = torch.log(r_safe)
    r_power = torch.exp(rho * log_r[None,:])
    r_power = torch.clamp(r_power, 1e-6, 10.0)

    return phase_diff, r_power, r_safe

def preprocess_signal(signal, fs=1000, lowcut=1.5, highcut=20):
    detrended = detrend(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, detrended)
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    return normalized

ecg_processed = preprocess_signal(ecg_data, fs=1000, lowcut=1.5, highcut=20)
eeg_processed = np.array([preprocess_signal(row, fs=1000, lowcut=0.5, highcut=20) for row in eeg_data])

# --- UTILITY FUNCTIONS ---
def simulate_coupled_oscillators(T=10, dt=1/1000, alpha=1, omega1=5.01, omega2=5.1, A_init=0.0001, theta_init=3.14, n=1.0, modulation=None):
    N = int(T / dt)
    r1, r2, phi1, phi2 = 1.0, 1.0, 0.0, 0.0
    A12, A21 = A_init, A_init
    theta12, theta21 = theta_init, theta_init

    R1, R2, Phi1, Phi2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        R1[i], R2[i], Phi1[i], Phi2[i] = r1, r2, phi1, phi2

        coupling12 = A12 * r2 * np.cos(theta12 + n * (phi2 - phi1))
        coupling21 = A21 * r1 * np.cos(theta21 + n * (phi1 - phi2))

        dr1 = alpha * r1 - r1**3 + coupling12 + (0.1*modulation[i,0] if modulation is not None and i < len(modulation) else 0)
        dr2 = alpha * r2 - r2**3 + coupling21 + (0.1*modulation[i,1] if modulation is not None and i < len(modulation) else 0)

        dphi1 = omega1 + A12 * r2 / r1 * np.sin(theta12 + n * (phi2 - phi1))
        dphi2 = omega2 + A21 * r1 / r2 * np.sin(theta21 + n * (phi1 - phi2))

        r1 += dr1 * dt
        r2 += dr2 * dt
        phi1 += dphi1 * dt
        phi2 += dphi2 * dt

    return np.stack((R1*np.cos(Phi1), R1*np.sin(Phi1), R2*np.cos(Phi2), R2*np.sin(Phi2)), axis=1)

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

# --- NEURAL NETWORK MODELS ---
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
## This section needs to be revised..
# a. check the phase difference equation.. for power coupling
# b. You need to change the r,phi equation too..
# c. In case you are getting 'nan', use torch.clamp for ratio (omega1/omega2)

class OscillatorLayer(nn.Module):
    def __init__(self, N_osc=16, T=2.0, fs=100, device="cpu",
                 coupling_sparsity=0.3, seed=42):

        super().__init__()

        self.N_osc = N_osc
        self.num_steps = int(T * fs)
        self.dt = 1.0 / fs
        self.mu = 1.0
        self.k = 0.01

        torch.manual_seed(seed)

        freqs = 2.0 + torch.rand(N_osc, device=device) * 8.0
        self.omega = nn.Parameter(2 * torch.pi * freqs)

        self.register_buffer("initial_r", torch.ones(N_osc, device=device) * 0.1)
        self.register_buffer("initial_phi", torch.zeros(N_osc, device=device))

        mask = torch.rand(N_osc, N_osc, device=device) > coupling_sparsity
        mask.fill_diagonal_(False)

        C = torch.rand(N_osc, N_osc, device=device) * 0.02
        self.register_buffer("C", C * mask.float())

        theta = torch.zeros(N_osc, N_osc, device=device)
        theta.fill_diagonal_(0.0)
        self.register_buffer("theta", theta)

        # --- Stability clamps ---
        self.r_min = 0.01
        self.r_max = 2.0
        self.power_min = 1e-6
        self.power_max = 10.0
        self.phase_clip = 50.0   # radians

    def forward(self, input_features):

        if input_features.dim() == 1:
            input_features = input_features.unsqueeze(0)

        B = input_features.shape[0]
        device = input_features.device

        r = self.initial_r.repeat(B,1)
        phi = 2*np.pi*torch.rand_like(r)

        drive = input_features

        omega = self.omega

        C = self.C
        theta = self.theta

        for _ in range(self.num_steps):

            phase_diff, r_power, r_safe = \
            power_coupling_terms(r[0], phi[0], omega, theta)

            coupling_r = self.k * torch.sum(
                C * r_power * torch.cos(phase_diff),
                dim=1
            )

            coupling_phi = self.k * torch.sum(
                C * (r_power / r_safe[:,None]) *
                torch.sin(phase_diff),
                dim=1
            )

            # ---- complex forcing ----
            Fx = drive
            Fy = torch.zeros_like(Fx)

            dr = (self.mu - r**2)*r \
                 + coupling_r \
                 + Fx*torch.cos(phi)

            dphi = omega \
                   + coupling_phi \
                   - (Fx/r_safe)*torch.sin(phi)

            r = r + dr*self.dt
            phi = phi + dphi*self.dt

            r = torch.clamp(r,0.01,2.0)
            phi = torch.remainder(phi,2*np.pi)

        return torch.cat([
            r*torch.cos(phi),
            r*torch.sin(phi)
        ],dim=-1)



class ECGToOscillatorMLP(nn.Module):
    """ECG → MLP → OscillatorLayer → MLP → Brain drive [N]"""
    def __init__(self, ecg_dim=50, N_VNS=16, hidden_dim=64, output_dim=16, device="cuda"):
        super().__init__()
        self.pre_osc = nn.Sequential(
            nn.Linear(ecg_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, N_VNS)
        )
        self.osc_layer = OscillatorLayer(N_osc=N_VNS, device=device, coupling_sparsity=0.3, seed=42)
        self.post_osc = nn.Sequential(
            nn.Linear(N_VNS * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, ecg_features):  # [batch, ecg_dim] or [ecg_dim]
        if ecg_features.dim() == 1:
            ecg_features = ecg_features.unsqueeze(0)  # (1, ecg_dim)

        pre = self.pre_osc(ecg_features)        # (B, N_VNS)
        osc_hidden = self.osc_layer(pre)         # (B, N_VNS * 2)
        brain_drive = self.post_osc(osc_hidden)  # (B, output_dim)

        if brain_drive.shape[0] == 1:
            return brain_drive.squeeze(0)  # (output_dim,)
        return brain_drive
class ODEFuc(nn.Module):
    def __init__(self, mu, eta_theta, eta_omega, eta_alpha,
                 D_function, N, Sc,
                 brain_drive_full=None,
                 fs=100):

        super().__init__()
        self.mu = mu
        self.eta_theta = eta_theta
        self.eta_omega = eta_omega
        self.eta_alpha = eta_alpha
        self.D_function = D_function
        self.N = N
        self.fs = fs

        self.register_buffer(
            'Sc',
            torch.tensor(Sc, dtype=torch.float32)
        )

        # brain_drive_full is stored as a plain attribute (NOT a buffer) so that
        # when using standard odeint, the full autograd graph is preserved and
        # gradients flow back through it to ecg_to_osc_mlp.
        self.brain_drive_full = brain_drive_full

    def forward(self, t, state):

        N = self.N

        r = state[:N]
        phi = state[N:2*N]
        theta = state[2*N:2*N+N**2].view(N,N)
        omega = state[2*N+N**2:3*N+N**2]
        alpha = state[3*N+N**2:4*N+N**2]

        phase_diff, r_power, r_safe = \
            power_coupling_terms(r,phi,omega,theta)

        # ----- differentiable target -----
        D = torch.tensor(
            self.D_function(float(t)),
            device=state.device
        )

        brain_state = r*torch.cos(phi)
        P = torch.sum(alpha*brain_state)

        e = D - P

        # ---- smooth drive interpolation ----
        if self.brain_drive_full is not None:
            idx = torch.clamp(
                (t*self.fs).long(),
                0,
                self.brain_drive_full.shape[0]-1
            )
            drive = self.brain_drive_full[idx]
        else:
            drive = torch.zeros(N,device=state.device)

        coupling_r = torch.sum(
            self.Sc * r_power * torch.cos(phase_diff),
            dim=1
        )

        coupling_phi = torch.sum(
            self.Sc *
            (r_power/r_safe[:,None]) *
            torch.sin(phase_diff),
            dim=1
        )

        # ----- complex forcing -----
        Fx = drive + alpha*e
        Fy = torch.zeros_like(Fx)

        drdt = (self.mu-r**2)*r \
            + coupling_r \
            + Fx*torch.cos(phi)

        dphidt = omega \
                + coupling_phi \
                - (Fx/r_safe)*torch.sin(phi)

        dthetadt = (
            self.eta_theta *
            r[:,None]*r_power *
            torch.sin(phase_diff) *
            self.Sc
        )*0.01

        domegadt = (
            -self.eta_omega *
            e*r*torch.sin(phi)
        )*0.01

        dalphadt = (
            self.eta_alpha *
            e*r*torch.cos(phi)
        )*0.01

        phi = torch.remainder(phi,2*np.pi)

        return torch.cat([
            drdt,
            dphidt,
            dthetadt.flatten(),
            domegadt,
            dalphadt
        ])
       


class TorchRevHopfNetwork:
    def __init__(self, mu, eta_omega, eta_alpha, eta_theta,
                 D_function, N, Sc,
                 brain_drive_full=None,
                 fs=100,
                 device="cuda"):

        self.device = torch.device(device)
        self.N = N

        self.ode_func = ODEFuc(
            mu=mu,
            eta_theta=eta_theta,
            eta_omega=eta_omega,
            eta_alpha=eta_alpha,
            D_function=D_function,
            N=N,
            Sc=Sc,
            brain_drive_full=brain_drive_full,
            fs=fs
        ).to(self.device)

    def solve(self, r0, phi0, theta0, omega0, alpha0, t_eval, use_adjoint=True):
        # use_adjoint=True  → odeint_adjoint (memory-efficient, for no-grad stages)
        # use_adjoint=False → odeint (standard autograd graph, required when
        #                    brain_drive_full must carry gradients back to ecg_to_osc_mlp)

        y0 = torch.tensor(
            np.concatenate([r0, phi0, theta0.flatten(), omega0, alpha0]),
            device=self.device,
            dtype=torch.float32
        )

        t_eval_tensor = torch.tensor(
            t_eval,
            device=self.device,
            dtype=torch.float32
        )

        if use_adjoint:
            sol = odeint_adjoint(
                self.ode_func,
                y0,
                t_eval_tensor,
                method="rk4"
            )
        else:
            # Standard odeint builds a full autograd graph so gradients flow
            # back through brain_drive_full to ecg_to_osc_mlp parameters.
            sol = odeint(
                self.ode_func,
                y0,
                t_eval_tensor,
                method="rk4"
            )

        N = self.N

        r = sol[:, :N]
        phi = sol[:, N:2*N]
        theta = sol[:, 2*N:2*N + N**2].view(-1, N, N)
        omega = sol[:, 2*N + N**2:3*N + N**2]
        alpha = sol[:, 3*N + N**2:4*N + N**2]

        rcos_phi = torch.sum(r * torch.cos(phi), dim=1)

        return r, phi, theta, omega, alpha, rcos_phi


def train_heart_model(ecg_target_signal, device):
    print("--- Starting Heart Model Pre-training ---")
    heart_model = HeartModel().to(device)
    optimizer = optim.Adam(heart_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
    criterion = nn.MSELoss()

    sim_osc_input = torch.tensor(simulate_coupled_oscillators(T=2, dt=0.01), dtype=torch.float32).to(device)
    ecg_target = torch.tensor(ecg_target_signal[::10], dtype=torch.float32).to(device).unsqueeze(1)

    # FIX 4 (support): collect loss history so we can plot it later
    heart_losses = []

    for epoch in range(25000):
        predicted_ecg = heart_model(sim_osc_input)
        loss = criterion(predicted_ecg, ecg_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        heart_losses.append(loss.item())
        if (epoch + 1) % 2500 == 0:
            print(f"Heart Epoch {epoch+1}, Loss: {loss.item():.6f}")
    print("--- Heart Pre-training Finished ---")
    return heart_model, heart_losses


def pre_train_brain_model(eeg_processed, Sw_all, target_idx, non_zero_indices_per_row, t, D_function, device):
    print("\n--- Stage 1: Brain Pre-training ---")
    # NOTE: This stage is GRADIENT-FREE forward simulation.
    # It iteratively runs the ODE forward and uses the final state as the next
    # initial condition, effectively "warming up" the brain model's initial conditions.
    # No optimizer or loss.backward() is used here by design.
    connected_indices = np.unique(np.append(non_zero_indices_per_row[target_idx], target_idx))
    N_reduced_regions = len(connected_indices)
    osc_per_region = 3
    N = N_reduced_regions * osc_per_region

    Sc_reduced_regional = Sw_all[np.ix_(connected_indices, connected_indices)]
    Sc_reduced_osc = expand_structural_connectivity(Sc_reduced_regional, osc_per_region, seed=42)

    omega_full = get_random_frequencies(68, osc_per_region, low=1, high=20, seed=42)
    alpha_full = np.random.uniform(0.1, 0.7, 68 * osc_per_region)
    omega0 = np.concatenate([omega_full[i * osc_per_region:(i + 1) * osc_per_region] for i in connected_indices])
    alpha0 = np.clip(np.concatenate([alpha_full[i * osc_per_region:(i + 1) * osc_per_region] for i in connected_indices]), 0.05, 0.5)
    r0 = 0.1 * np.ones(N)
    phi0 = np.zeros(N)
    theta_random = np.pi * (2 * np.random.rand(N, N) - 1)
    theta0 = theta_random - theta_random.T

    model = TorchRevHopfNetwork(
        mu=1.0, eta_omega=0.05, eta_alpha=0.005, eta_theta=0.05,
        D_function=D_function, N=N, Sc=Sc_reduced_osc,
        device=device
    )

    criterion = nn.MSELoss()
    D_true = torch.tensor(D_function(t), device=device, dtype=torch.float32)
    losses = []

    for epoch in range(100):
        with torch.no_grad():
            # use_adjoint=True is fine here since we don't need gradients
            r, phi, theta, omega, alpha, _ = model.solve(r0, phi0, theta0, omega0, alpha0, t, use_adjoint=True)
            P_out = torch.sum(alpha * r * torch.cos(phi), axis=1)
            loss = criterion(P_out, D_true)
            losses.append(loss.item())

            theta0, omega0 = theta[-1].cpu().numpy(), omega[-1].cpu().numpy()
            alpha0 = alpha[-1].cpu().numpy()

        if (epoch + 1) % 10 == 0:
            print(f"Brain Epoch {epoch+1}/30, Loss: {loss.item():.6f}")

        # FIX 3: Free GPU memory after each epoch
        torch.cuda.empty_cache()
        gc.collect()

    final_params = {'r': r0, 'phi': phi0, 'theta': theta0, 'omega': omega0, 'alpha': alpha0}
    return final_params, Sc_reduced_osc, N, losses


def train_mlp_on_frozen_brain(
        trained_heart_model,
        initial_brain_params,
        Sc_reduced_osc,
        N,
        D_function,
        t,
        device):

    print("\n--- Stage 2: ECG → OscillatorLayer → Brain Training ---")

    ecg_to_osc_mlp = ECGToOscillatorMLP(
        ecg_dim=50,
        N_VNS=8,
        hidden_dim=64,
        output_dim=N,
        device=device
    ).to(device)

    # FIX 8a: Lower LR from 1e-2 → 1e-3 to prevent divergence.
    # With standard odeint, gradients backprop through 200 ODE steps and can be
    # very large. A high LR causes the optimizer to overshoot → loss explodes.
    optimizer = torch.optim.Adam(
        ecg_to_osc_mlp.parameters(),
        lr=5e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    criterion = nn.MSELoss()

    # -------- Extract ECG features (frozen heart model) --------
    with torch.no_grad():
        sim_input = torch.tensor(
            simulate_coupled_oscillators(T=2, dt=0.01),
            dtype=torch.float32
        ).to(device)
        hidden_repr = trained_heart_model.get_features(sim_input)  # (T_steps, feature_dim)

    D_true = torch.tensor(D_function(t), device=device, dtype=torch.float32)
    losses = []

    # FIX 1 (CRITICAL): Create TorchRevHopfNetwork ONCE outside the loop.
    # Previously it was re-instantiated every epoch, causing massive GPU memory accumulation.
    # Now we create it once with brain_drive_full=None and update the attribute each epoch.
    model = TorchRevHopfNetwork(
        mu=1.0,
        eta_omega=0.0,
        eta_alpha=0.0,
        eta_theta=0.0,
        D_function=D_function,
        N=N,
        Sc=Sc_reduced_osc,
        brain_drive_full=None,
        fs=100,
        device=device
    )

    for epoch in range(200):

        raw_drive = ecg_to_osc_mlp(hidden_repr)  # (T_steps, N) — has grad_fn

        brain_drive_full = 0.05 * raw_drive

        # Update brain_drive_full on the ODE func each epoch.
        # Stored as a plain attribute (not a buffer) so the autograd graph is preserved.
        model.ode_func.brain_drive_full = brain_drive_full

        # FIX 7 (CRITICAL): use_adjoint=False → standard odeint builds a full autograd
        # graph so gradients flow back through brain_drive_full to ecg_to_osc_mlp.
        # odeint_adjoint only tracks gradients through registered nn.Module parameters
        # and would raise: "element 0 of tensors does not require grad and does not have a grad_fn"
        r, phi, theta, omega, alpha, _ = model.solve(
            initial_brain_params['r'],
            initial_brain_params['phi'],
            initial_brain_params['theta'],
            initial_brain_params['omega'],
            initial_brain_params['alpha'],
            t,
            use_adjoint=False
        )

        P_out = torch.sum(alpha * r * torch.cos(phi), dim=1)
        loss = criterion(P_out, D_true)

        optimizer.zero_grad()
        loss.backward()
        # FIX 8a (cont.): Gradient clipping prevents exploding gradients from the
        # deep ODE unroll (200 steps of backprop through Euler integration).
        torch.nn.utils.clip_grad_norm_(ecg_to_osc_mlp.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

        # FIX 3: Free GPU memory each epoch
        torch.cuda.empty_cache()
        gc.collect()

    return ecg_to_osc_mlp, losses
target_indices = [4]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {device} ---")

# Step 1: Pre-train heart model
# FIX 4 (support): train_heart_model now also returns heart_losses for plotting
trained_heart_model, heart_losses = train_heart_model(ecg_processed, device)

with torch.no_grad():
    simulated_ecg_input = torch.tensor(simulate_coupled_oscillators(T=2, dt=0.01), dtype=torch.float32).to(device)
    hidden_repr = trained_heart_model.get_features(simulated_ecg_input)

results_folder = "simulation_results"
os.makedirs(results_folder, exist_ok=True)

target_idx = target_indices[0]

t_duration = 2
fs = 100
t = np.arange(0, t_duration, 1/fs)
target_signal = eeg_processed[target_idx, ::10]
D_function = interp1d(t, target_signal, kind='linear', bounds_error=False, fill_value=0.0)

# Step 2: Stage 1 - Brain pre-training (gradient-free forward simulation)
final_brain_params, Sc_reduced_osc, N, brain_losses = pre_train_brain_model(
    eeg_processed, Sw_all, target_idx, non_zero_indices_per_row, t, D_function, device
)
# Step 3: Stage 2 - Train ECG→OscillatorLayer→Brain MLP
trained_mlp_model, mlp_losses = train_mlp_on_frozen_brain(
    trained_heart_model, final_brain_params, Sc_reduced_osc, N, D_function, t, device
)

print("\n--- Extracting Brain rcos_phi for Feedback ---")
with torch.no_grad():
    brain_drive_for_final = 0.05*(trained_mlp_model(hidden_repr))  ## torch.tanh is not needed
    brain_drive_full = 0.05*(trained_mlp_model(hidden_repr))  # [T, N]

model = TorchRevHopfNetwork(
    mu=1.0,
    eta_omega=0.0,
    eta_alpha=0.0,
    eta_theta=0.0,
    D_function=D_function,
    N=N,
    Sc=Sc_reduced_osc,
    brain_drive_full=brain_drive_full,
    fs=100,
    device=device
)

r_final, phi_final, theta_final, omega_final, alpha_final, rcos_phi_final = model.solve(
    final_brain_params['r'], final_brain_params['phi'], final_brain_params['theta'],
    final_brain_params['omega'], final_brain_params['alpha'], t
)
print(f"rcos_phi_final shape: {rcos_phi_final.shape}, range: [{rcos_phi_final.min():.3f}, {rcos_phi_final.max():.3f}]")
# ★★★ FINAL PREDICTION & PLOTTING ★★★
print("\n--- Final Predictions ---")
trained_heart_model.eval()
with torch.no_grad():
    # Baseline ECG (no feedback)
    sim_osc_baseline = simulate_coupled_oscillators(T=t_duration, dt=1/fs)
    predicted_ecg_baseline = trained_heart_model(
        torch.tensor(sim_osc_baseline, dtype=torch.float32).to(device)
    ).cpu().numpy().flatten()

    P_out_baseline = torch.sum(alpha_final * r_final * torch.cos(phi_final), axis=1).cpu().numpy()

# ★★★ PLOTTING ★★★
# FIX 4: axes[2] now shows heart model loss (was blank before)
fig, axes = plt.subplots(5, 1, figsize=(15, 20))

axes[0].plot(brain_losses)
axes[0].set_title('Stage 1: Brain Pre-training Loss')
axes[0].grid(True)

axes[1].plot(mlp_losses)
axes[1].set_title('Stage 2: MLP Training Loss')
axes[1].grid(True)

# FIX 4: Previously axes[2] was skipped (blank). Now shows heart model loss.
axes[2].plot(heart_losses)
axes[2].set_title('Heart Model Pre-training Loss')
axes[2].grid(True)

target_ecg = ecg_processed[::10]
timesteps = np.linspace(0, t_duration, len(target_ecg))

axes[3].plot(timesteps, target_ecg, label='Target ECG', linewidth=2)
axes[3].plot(timesteps, predicted_ecg_baseline, label='Baseline ECG', linestyle='--')
axes[3].set_title('ECG Prediction: Baseline vs Brain Feedback')
axes[3].legend()
axes[3].grid(True)

axes[4].plot(t, D_function(t), label='Target EEG', linewidth=2)
axes[4].plot(t, P_out_baseline, label='P_out baseline', alpha=0.7)
axes[4].set_title('Brain Output: rcos_phi vs Target')
axes[4].legend()
axes[4].grid(True)

plt.tight_layout()
plt.savefig(f"{results_folder}/full_feedback_result_idx{target_idx}.png", dpi=300, bbox_inches='tight')
plt.show()

np.savez(f"{results_folder}/results_idx{target_idx}.npz",
        brain_losses=brain_losses,
        mlp_losses=mlp_losses,
        heart_losses=heart_losses,
        rcos_phi_final=rcos_phi_final.detach().cpu().numpy(),
        P_out_baseline=P_out_baseline,
        predicted_ecg_baseline=predicted_ecg_baseline,
        target_ecg=target_ecg,
        target_eeg=D_function(t))

print("✅ COMPLETE! Check simulation_results/ folder")
## Actual vs Predicted Signal Analysis
#This section computes descriptive metrics comparing the actual EEG target signal with the predicted brain response. We split the sequence into Train (80%) and Test (20%) sections to evaluate the model's accuracy.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch

# Compute Actual vs Predicted signals
with torch.no_grad():
    P_predicted = torch.sum(alpha_final * r_final * torch.cos(phi_final), dim=1).cpu().numpy()
    D_actual = D_function(t)

# Define train and test sections (e.g., 80% / 20% split)
# Note: The model trained the Brain components over the entire duration,
# but we can evaluate metrics over a training and a testing slice.
split_idx = int(0.8 * len(t))

t_train, D_train, P_train = t[:split_idx], D_actual[:split_idx], P_predicted[:split_idx]
t_test, D_test, P_test = t[split_idx:], D_actual[split_idx:], P_predicted[split_idx:]

def compute_metrics(actual, predicted, section_name):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    corr, _ = pearsonr(actual, predicted)
    
    # Signal to Noise Ratio (SNR)
    signal_power = np.mean(actual ** 2)
    noise_power = np.mean((actual - predicted) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # Normalized Root Mean Squared Error (NRMSE)
    nrmse = rmse / (np.max(actual) - np.min(actual)) if (np.max(actual) - np.min(actual)) != 0 else float('inf')
    
    print(f"--- {section_name} Metrics ---")
    print(f"MSE (Mean Squared Error):        {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.6f}")
    print(f"NRMSE (Normalized RMSE):         {nrmse:.6f}")
    print(f"MAE (Mean Absolute Error):       {mae:.6f}")
    print(f"R-squared (R2 Score):            {r2:.6f}")
    print(f"Pearson Correlation (r):         {corr:.6f}")
    print(f"SNR (Signal-to-Noise Ratio):     {snr:.2f} dB")
    print("-" * 40)
    
    return mse, rmse, mae, r2, corr, snr

print("=" * 40)
print("          SIGNAL ANALYSIS REPORT        ")
print("=" * 40)
train_metrics = compute_metrics(D_train, P_train, "TRAIN SECTION")
test_metrics = compute_metrics(D_test, P_test, "TEST SECTION")
overall_metrics = compute_metrics(D_actual, P_predicted, "OVERALL")

# Plotting the results
plt.figure(figsize=(15, 6))
plt.plot(t, D_actual, label='Actual Target EEG', color='black', linewidth=2, alpha=0.7)
plt.plot(t, P_predicted, label='Predicted Brain Response', color='red', linestyle='--', linewidth=2)
plt.axvline(x=t[split_idx], color='blue', linestyle=':', linewidth=2, label='Train/Test Split')

plt.title('Actual vs Predicted Signal', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate metrics on plot
metrics_text = (
    f"Overall r: {overall_metrics[4]:.3f}\n"
    f"Overall R2: {overall_metrics[3]:.3f}\n"
    f"Overall RMSE: {overall_metrics[1]:.3f}"
)
plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.show()

# Additional Plot: Error over time
plt.figure(figsize=(15, 4))
error = D_actual - P_predicted
plt.plot(t, error, label='Prediction Error', color='purple', alpha=0.7)
plt.axvline(x=t[split_idx], color='blue', linestyle=':', linewidth=2)
plt.fill_between(t, error, 0, where=(error > 0), color='salmon', alpha=0.3)
plt.fill_between(t, error, 0, where=(error < 0), color='lightblue', alpha=0.3)
plt.title('Prediction Error Over Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Error Amplitude', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
## Actual vs Predicted Signal Analysis
#This section computes descriptive metrics comparing the actual EEG target signal with the predicted brain response. We split the sequence into Train (80%) and Test (20%) sections to evaluate the model's accuracy.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch

# Compute Actual vs Predicted signals
with torch.no_grad():
    P_predicted = torch.sum(alpha_final * r_final * torch.cos(phi_final), dim=1).cpu().numpy()
    D_actual = D_function(t)

# Define train and test sections (e.g., 80% / 20% split)
# Note: The model trained the Brain components over the entire duration,
# but we can evaluate metrics over a training and a testing slice.
split_idx = int(0.8 * len(t))

t_train, D_train, P_train = t[:split_idx], D_actual[:split_idx], P_predicted[:split_idx]
t_test, D_test, P_test = t[split_idx:], D_actual[split_idx:], P_predicted[split_idx:]

def compute_metrics(actual, predicted, section_name):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    corr, _ = pearsonr(actual, predicted)
    
    # Signal to Noise Ratio (SNR)
    signal_power = np.mean(actual ** 2)
    noise_power = np.mean((actual - predicted) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # Normalized Root Mean Squared Error (NRMSE)
    nrmse = rmse / (np.max(actual) - np.min(actual)) if (np.max(actual) - np.min(actual)) != 0 else float('inf')
    
    print(f"--- {section_name} Metrics ---")
    print(f"MSE (Mean Squared Error):        {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.6f}")
    print(f"NRMSE (Normalized RMSE):         {nrmse:.6f}")
    print(f"MAE (Mean Absolute Error):       {mae:.6f}")
    print(f"R-squared (R2 Score):            {r2:.6f}")
    print(f"Pearson Correlation (r):         {corr:.6f}")
    print(f"SNR (Signal-to-Noise Ratio):     {snr:.2f} dB")
    print("-" * 40)
    
    return mse, rmse, mae, r2, corr, snr

print("=" * 40)
print("          SIGNAL ANALYSIS REPORT        ")
print("=" * 40)
train_metrics = compute_metrics(D_train, P_train, "TRAIN SECTION")
test_metrics = compute_metrics(D_test, P_test, "TEST SECTION")
overall_metrics = compute_metrics(D_actual, P_predicted, "OVERALL")

# Plotting the results
plt.figure(figsize=(15, 6))
plt.plot(t, D_actual, label='Actual Target EEG', color='black', linewidth=2, alpha=0.7)
plt.plot(t, P_predicted, label='Predicted Brain Response', color='red', linestyle='--', linewidth=2)
plt.axvline(x=t[split_idx], color='blue', linestyle=':', linewidth=2, label='Train/Test Split')

plt.title('Actual vs Predicted Signal', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate metrics on plot
metrics_text = (
    f"Overall r: {overall_metrics[4]:.3f}\n"
    f"Overall R2: {overall_metrics[3]:.3f}\n"
    f"Overall RMSE: {overall_metrics[1]:.3f}"
)
plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.show()

# Additional Plot: Error over time
plt.figure(figsize=(15, 4))
error = D_actual - P_predicted
plt.plot(t, error, label='Prediction Error', color='purple', alpha=0.7)
plt.axvline(x=t[split_idx], color='blue', linestyle=':', linewidth=2)
plt.fill_between(t, error, 0, where=(error > 0), color='salmon', alpha=0.3)
plt.fill_between(t, error, 0, where=(error < 0), color='lightblue', alpha=0.3)
plt.title('Prediction Error Over Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Error Amplitude', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()