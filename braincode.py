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

def load_data(ecg,eeg,sc):
    raw = mne.io.read_raw_fif(ecg, preload=True)
    data,times=raw[322,2000:4000]
    ecg_data=-data[0]
    mat=loadmat(eeg)
    eeg_data = mat['Value'][:, 2000:4000]
    sc_matrix = loadmat(sc)['sc']
    max_val = np.max(sc_matrix)
    Sw_all = (sc_matrix / max_val) * 0.01 if max_val > 0 else sc_matrix
    return ecg_data,eeg_data,Sw_all

def preprocess_signal(signal, fs=1000, lowcut=1.5, highcut=20):
    detrended = detrend(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, detrended)
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    return normalized

def heart_osc(T,dt,alpha=1,omega1=5.01,omega2=5.1,A_init=0.0001,theta_init=3.14,n=1):
    N=int(T/dt)
    r1,r2,phi1,phi2=1.0,1.0,0.0,0.0
    A12,A21=A_init,A_init
    theta12,theta21=theta_init,theta_init
    R1,R2,Phi1,Phi2=np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
    for i in range(N):
        R1[i], R2[i], Phi1[i], Phi2[i] = r1, r2, phi1, phi2

        coupling12 = A12 * r2 * np.cos(theta12 + n * (phi2 - phi1))
        coupling21 = A21 * r1 * np.cos(theta21 + n * (phi1 - phi2))

        dr1 = alpha * r1 - r1**3 + coupling12 
        dr2 = alpha * r2 - r2**3 + coupling21 

        dphi1 = omega1 + A12 * r2 / r1 * np.sin(theta12 + n * (phi2 - phi1))
        dphi2 = omega2 + A21 * r1 / r2 * np.sin(theta21 + n * (phi1 - phi2))

        r1 += dr1 * dt
        r2 += dr2 * dt
        phi1 += dphi1 * dt
        phi2 += dphi2 * dt

    return np.stack((R1*np.cos(Phi1), R1*np.sin(Phi1), R2*np.cos(Phi2), R2*np.sin(Phi2)), axis=1)

def get_random_frequencies(num_regions, osc_per_region, low=1, high=20, seed=42):
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
    

class OscillatorLayer(nn.Module):
    def __init__(
        self,
        N_osc=16,
        T=2.0,
        fs=100,
        device="cpu",
        coupling_sparsity=0.3,
        seed=42,
        coupling_strength=0.05,
    ):
        super().__init__()

        self.N_osc = N_osc
        self.num_steps = int(T * fs)
        self.dt = 1.0 / fs
        self.mu = 1.0  

        torch.manual_seed(seed)

        freqs_hz = 2.0 + torch.rand(N_osc) * 8.0
        self.omega = nn.Parameter(2 * np.pi * freqs_hz)

        # ---- Fixed initial conditions ----
        self.register_buffer("initial_r", torch.ones(N_osc) * 0.1)
        self.register_buffer("initial_phi", torch.zeros(N_osc))

        # ---- Structural coupling matrix ----
        coupling_mask = (torch.rand(N_osc, N_osc) > coupling_sparsity).float()
        coupling_mask.fill_diagonal_(0.0)  # disable self-coupling

        random_coupling = torch.rand(N_osc, N_osc) * 0.02
        C = random_coupling * coupling_mask

        self.register_buffer("C", C)
        self.register_buffer("coupling_strength", torch.tensor(coupling_strength))

        self.to(device)

    def forward(self, input_features):
 
        batch_size = input_features.shape[0]
        N = self.N_osc
        T = self.num_steps

        r = self.initial_r.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        phi = self.initial_phi.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)

        omega = self.omega.unsqueeze(0).unsqueeze(-1)
        C = self.C.unsqueeze(0)

        # Storage tensors
        x_traj = torch.zeros(batch_size, N, T, device=r.device)
        y_traj = torch.zeros(batch_size, N, T, device=r.device)

        for t in range(T):

            phase_diff = phi - phi.transpose(1, 2)
            r_safe = torch.clamp(r, 1e-4, 10.0)
            r_j = r.transpose(1, 2)

            coupling_r = self.coupling_strength * torch.sum(
                C * r_j * torch.cos(phase_diff),
                dim=-1,
                keepdim=True
            )

            coupling_phi = self.coupling_strength * torch.sum(
                C * (r_j / r_safe) * torch.sin(phase_diff),
                dim=-1,
                keepdim=True
            )
            D = input_features.unsqueeze(-1)
            forcing_r = D * torch.cos(phi)
            forcing_phi = -(D / r_safe) * torch.sin(phi)
            dr_dt = (self.mu - r**2) * r + coupling_r + forcing_r
            dphi_dt = omega + coupling_phi + forcing_phi
            r = r + dr_dt * self.dt
            phi = phi + dphi_dt * self.dt
    
            x_traj[:, :, t] = (r.squeeze(-1) * torch.cos(phi.squeeze(-1)))
            y_traj[:, :, t] = (r.squeeze(-1) * torch.sin(phi.squeeze(-1)))

        # Final shape: (B, N, 2, T)
        return torch.stack([x_traj, y_traj], dim=2)
    



class ECGToOscillatorMLP(nn.Module):
    """ECG → MLP → OscillatorLayer → MLP → Brain drive [N]"""
    def __init__(self, ecg_dim=50, N_VNS=128, hidden_dim=200,output_dim=16, device="cuda"): # Added device here
        super().__init__()
        self.pre_osc = nn.Sequential(
            nn.Linear(ecg_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, N_VNS)

        )
        self.osc_layer = OscillatorLayer(N_osc=N_VNS, device=device, coupling_sparsity=0.3, seed=42) # Pass device to OscillatorLayer
        self.post_osc = nn.Sequential(
            nn.Linear(N_VNS * 2, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)  # Matches your brain N
        )

    def forward(self, ecg_features):  # [batch, ecg_dim] or [T, ecg_dim]
        # Add a batch dimension if it's a single feature vector
        if ecg_features.dim() == 1:
            ecg_features = ecg_features.unsqueeze(0) # Makes it (1, ecg_dim)

        pre = self.pre_osc(ecg_features) # Shape (batch_size, N_VNS)
        osc_hidden = self.osc_layer(pre)  # Oscillator magic happens here! Shape (batch_size, N_VNS * 2)
        brain_drive = self.post_osc(osc_hidden) # Shape (batch_size, output_dim)

        if brain_drive.shape[0] == 1: # If it was a single input, remove batch dim for consistency with ODEFuc
            return brain_drive.squeeze(0) # Returns (output_dim,)
        return brain_drive


# --- ODE FUNCTION (UNCHANGED) ---
class ODEFuc(nn.Module):
    def __init__(self, mu, eta_theta, eta_omega, eta_alpha, D_function, N, Sc, mlp_model=None, hidden_repr=None):
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
        r, phi = state[:N], state[N:2*N]
        theta = state[2*N:2*N + N**2].view(N, N)
        omega, alpha = state[2*N + N**2:3*N + N**2], state[3*N + N**2:4*N + N**2]

        omega_safe = torch.clamp(omega, 2 * np.pi * 0.5, 2 * np.pi * 20)
        r = torch.clamp(r, 1e-1, 2.0)
        alpha = torch.clamp(alpha, -1.0, 1.0)
        r_safe = torch.clamp(torch.where(r < 1e-6, torch.tensor(1e-6, device=r.device, dtype=r.dtype), r), 1e-5, 10.0)
        r = r_safe
        phase_diff = torch.clamp(
            phi[None, :] / omega_safe[None, :] - phi[:, None] / omega_safe[:, None] + theta / (omega_safe[:, None] * omega_safe[None, :]), -1e2, 1e2)

        D = torch.tensor(self.D_function(t.item()), device=state.device, dtype=state.dtype)
        P = torch.sum(alpha * r * torch.cos(phi))
        e = (D - P)

        ecg_input = torch.zeros(N, device=state.device, dtype=state.dtype)
        if (self.mlp_model is not None) and (self.hidden_repr is not None):
            t_idx = min(int(t.item() * 100), self.hidden_repr.shape[0] - 1)
            ecg_features = self.hidden_repr[t_idx].to(device=state.device, dtype=state.dtype)
            ecg_input = self.mlp_model(ecg_features)
            ecg_input = torch.clamp(ecg_input.squeeze(), 0.01, 5.0)

        coupling_r = torch.sum(torch.abs(self.Sc) * r[None, :] * torch.cos(phase_diff), dim=1)
        drdt = (self.mu - r**2) * r + coupling_r + e * torch.cos(phi) + ecg_input

        coupling_phi = torch.sum(torch.abs(self.Sc) * (r[None, :] / r_safe[:, None]) * torch.sin(phase_diff), dim=1)
        dphidt = omega + coupling_phi - (e / r_safe) * torch.sin(phi)

        dthetadt = self.eta_theta * torch.sin(phase_diff) * torch.abs(self.Sc)
        domegadt = -self.eta_omega * e * torch.sin(phi)
        dalphadt = self.eta_alpha * e * r * torch.cos(phi)

        drdt = torch.clamp(drdt, -1e2, 1e2)
        dphidt = torch.clamp(dphidt, -1e2, 1e2)
        dthetadt = torch.clamp(dthetadt, -1e2, 1e2)
        domegadt = torch.clamp(domegadt, -1e2, 1e2)
        dalphadt = torch.clamp(dalphadt, -1e2, 1e2)

        return torch.cat([drdt.flatten(), dphidt.flatten(), dthetadt.flatten(), domegadt.flatten(), dalphadt.flatten()])

class TorchRevHopfNetwork:
    def __init__(self, mu, eta_omega, eta_alpha, eta_theta, D_function, N, Sc, mlp_model, hidden_repr, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.N = N
        self.ode_func = ODEFuc(
            mu=mu, eta_theta=eta_theta, eta_omega=eta_omega, eta_alpha=eta_alpha,
            D_function=D_function, N=N, Sc=torch.tensor(Sc, device=self.device, dtype=torch.float32),
            mlp_model=mlp_model, hidden_repr=hidden_repr.to(self.device) if hidden_repr is not None else None
        ).to(self.device)

    def solve(self, r0, phi0, theta0, omega0, alpha0, t_eval):
        dtype = torch.float32
        y0 = torch.tensor(np.concatenate([r0, phi0, theta0.flatten(), omega0, alpha0]), device=self.device, dtype=dtype)
        t_eval_tensor = torch.tensor(t_eval, device=self.device, dtype=dtype)

        sol = odeint_adjoint(self.ode_func, y0, t_eval_tensor, method='rk4', rtol=1e-5, atol=1e-7)

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
    criterion = nn.MSELoss()

    sim_osc_input = torch.tensor(heart_osc(T=2, dt=0.01), dtype=torch.float32).to(device)
    ecg_target = torch.tensor(ecg_target_signal[::10], dtype=torch.float32).to(device).unsqueeze(1)

    for epoch in range(25000):
        predicted_ecg = heart_model(sim_osc_input)
        loss = criterion(predicted_ecg, ecg_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2500 == 0:
            print(f"Heart Epoch {epoch+1}, Loss: {loss.item():.6f}")
    print("--- Heart Pre-training Finished ---")
    return heart_model


def pre_train_brain_model(eeg_processed, Sw_all, target_idx, non_zero_indices_per_row, t, D_function, device):
    print("\n--- Stage 1: Brain Pre-training ---")
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
        mlp_model=None, hidden_repr=None, device=device
    )

    criterion = nn.MSELoss()
    D_true = torch.tensor(D_function(t), device=device, dtype=torch.float32)
    losses = []

    for epoch in range(30):
        with torch.no_grad():
            r, phi, theta, omega, alpha, _ = model.solve(r0, phi0, theta0, omega0, alpha0, t)
            P_out = torch.sum(alpha * r * torch.cos(phi), axis=1)
            loss = criterion(P_out, D_true)
            losses.append(loss.item())

            theta0, omega0 = theta[-1].cpu().numpy(), omega[-1].cpu().numpy()
            alpha0 = alpha[-1].cpu().numpy()

        if (epoch + 1) % 10 == 0:
            print(f"Brain Epoch {epoch+1}/100, Loss: {loss.item():.6f}")

    final_params = {'r': r0, 'phi': phi0, 'theta': theta0, 'omega': omega0, 'alpha': alpha0}
    return final_params, Sc_reduced_osc, N, losses

def train_mlp_on_frozen_brain(trained_heart_model, initial_brain_params, Sc_reduced_osc, N, D_function, t, device):
    print("\n--- Stage 2: ECG → OscillatorLayer → Brain Training ---")

    ecg_to_osc_mlp = ECGToOscillatorMLP(
        ecg_dim=50, N_VNS=128, hidden_dim=200,output_dim=N, device=device # Changed output_dim to N
    ).to(device)

    optimizer = torch.optim.Adam(ecg_to_osc_mlp.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    # ECG features from heart model (your existing pipeline)
    with torch.no_grad():
        simulated_ecg_input = torch.tensor(heart_osc(T=2, dt=0.01),
                                         dtype=torch.float32).to(device)
        ecg_features = trained_heart_model.get_features(simulated_ecg_input)  # [T, 50]

    losses = []
    for epoch in range(100):
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

        P_out = torch.sum(alpha * r * torch.cos(phi), axis=1)
        D_true = torch.tensor(D_function(t), device=device, dtype=torch.float32)
        loss = criterion(P_out, D_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"ECG→Oscillator→Brain Epoch {epoch+1}, Loss: {loss.item():.6f}")

    return ecg_to_osc_mlp, losses





if __name__ == "__main__":

    ecg_file_path='/home/shobs/Desktop/DDP/transdef_mf2pt2_rest_raw.fif'
    eeg_file_path="/home/shobs/Desktop/DDP/scout_id_309.mat"
    sc_file_path='/home/shobs/Desktop/DDP/SC_CC120309-27.mat'
    ecg_data,eeg_data,Sw_all=load_data(ecg_file_path,eeg_file_path,sc_file_path)
    ecg_processed = preprocess_signal(ecg_data, fs=1000, lowcut=1.5, highcut=20)
    eeg_processed = np.array([preprocess_signal(row, fs=1000, lowcut=0.5, highcut=20) for row in eeg_data])
    target_indices = [4]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    non_zero_indices_per_row = [np.nonzero(Sw_all[i, :])[0] for i in range(Sw_all.shape[0])]
    use_half_precision = False
    debug_interval = 100

    print(f"--- Using device: {device} ---")
    trained_heart_model = train_heart_model(ecg_processed, device)
    #simulated_ecg_input = torch.tensor(simulate_coupled_oscillators(T=t[-1]+1/fs, dt=1/fs), dtype=torch.float32).to(device)
    with torch.no_grad():
        simulated_ecg_input = torch.tensor(heart_osc(T=2, dt=0.01), dtype=torch.float32).to(device)

        hidden_repr = trained_heart_model.get_features(simulated_ecg_input)

    results_folder = "simulation_results"
    os.makedirs(results_folder, exist_ok=True)
    target_idx=4
    t_duration = 2
    fs = 100
    t = np.arange(0, t_duration, 1/fs)
    target_signal = eeg_processed[target_idx, ::10]
    D_function = interp1d(t, target_signal, kind='linear', bounds_error=False, fill_value=0.0)
    final_brain_params, Sc_reduced_osc, N, brain_losses = pre_train_brain_model(
        eeg_processed, Sw_all, target_idx, non_zero_indices_per_row, t, D_function, device
    )

    trained_mlp_model, mlp_losses = train_mlp_on_frozen_brain(
        trained_heart_model, final_brain_params, Sc_reduced_osc, N, D_function, t, device
    )

    model_final = TorchRevHopfNetwork(
        mu=1.0, eta_omega=0.0, eta_alpha=0.0, eta_theta=0.0,
        D_function=D_function, N=N, Sc=Sc_reduced_osc,
        mlp_model=trained_mlp_model, hidden_repr=hidden_repr, device=device
    )
    r_final, phi_final, theta_final, omega_final, alpha_final, rcos_phi_final = model_final.solve(
        final_brain_params['r'], final_brain_params['phi'], final_brain_params['theta'],
        final_brain_params['omega'], final_brain_params['alpha'], t
    )
    print(f"rcos_phi_final shape: {rcos_phi_final.shape}, range: [{rcos_phi_final.min():.3f}, {rcos_phi_final.max():.3f}]")
