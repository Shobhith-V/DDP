"""
Coupled Hopf OscillatorLayer with correct dynamics.

Radial:  dr_i/dt = (mu - r_i^2)*r_i + k * sum_j C_ij * r_j * cos(phi_j - phi_i - theta_ij) + ECG_amp_input_i
Phase:   dphi_i/dt = omega_i + k * sum_j C_ij * (r_j/r_i) * sin(phi_j - phi_i - theta_ij) + ECG_phase_input_i

Minimal numerical safety clamp on r only.
"""

import torch
import torch.nn as nn


class OscillatorLayer(nn.Module):
    def __init__(
        self,
        N_osc: int = 64,
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
        self.N_osc = N_osc
        self.num_steps = int(T * fs)
        self.dt = 1.0 / fs
        self.device = torch.device(device) if isinstance(device, str) else device

        self.mu = nn.Parameter(torch.tensor(1.0))
        torch.manual_seed(seed)
        freqs_hz = freq_hz_min + torch.rand(N_osc) * (freq_hz_max - freq_hz_min)
        self.omega = nn.Parameter(2 * torch.pi * freqs_hz)

        self.initial_r = nn.Parameter(torch.ones(N_osc) * 0.1)
        self.initial_phi = nn.Parameter(torch.zeros(N_osc))

        torch.manual_seed(seed + 1)
        mask = torch.rand(N_osc, N_osc, device=self.device) > coupling_sparsity
        mask.fill_diagonal_(False)
        random_c = torch.rand(N_osc, N_osc, device=self.device) * 0.02
        self.register_buffer("C", random_c * mask.float())
        self.register_buffer("k", torch.tensor(coupling_strength, device=self.device))

        theta_random = torch.rand(N_osc, N_osc, device=self.device) * 2 * torch.pi - torch.pi
        self.theta = nn.Parameter(theta_random - theta_random.T)

    def forward(
        self,
        ecg_amp_input: torch.Tensor,
        ecg_phase_input: torch.Tensor | None = None,
        mu_mod: torch.Tensor | None = None,
        omega_mod: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        ecg_amp_input: (batch, N_osc)
        ecg_phase_input: (batch, N_osc) or None -> zeros
        Returns: (batch, N_osc*2) = [r*cos(phi), r*sin(phi)] or full trajectory (batch, T, N_osc*2)
        """
        batch = ecg_amp_input.shape[0]
        r = self.initial_r.unsqueeze(0).repeat(batch, 1).unsqueeze(-1)
        phi = self.initial_phi.unsqueeze(0).repeat(batch, 1).unsqueeze(-1)
        if ecg_phase_input is None:
            ecg_phase_input = torch.zeros_like(ecg_amp_input)
        ecg_phase_input = ecg_phase_input.unsqueeze(-1)

        trajectory = []
        for _ in range(self.num_steps):
            mu_t = self.mu + (mu_mod.unsqueeze(-1) if mu_mod is not None else 0.0)
            omega_t = self.omega + (omega_mod if omega_mod is not None else 0.0)

            # phase_diff[i,j] = phi_j - phi_i - theta_ij
            phi_diff = phi.transpose(-2, -1) - phi - self.theta.unsqueeze(0)
            r_j = r.transpose(-2, -1).expand(-1, self.N_osc, -1)
            r_i_safe = torch.clamp(r, 1e-6, 10.0).expand(-1, -1, self.N_osc)

            coupling_r = self.k * torch.sum(
                self.C.unsqueeze(0) * r_j * torch.cos(phi_diff), dim=-1
            ).unsqueeze(-1)
            coupling_phi = self.k * torch.sum(
                self.C.unsqueeze(0) * (r_j / r_i_safe) * torch.sin(phi_diff), dim=-1
            ).unsqueeze(-1)

            dr_dt = (mu_t - r**2) * r + coupling_r + ecg_amp_input.unsqueeze(-1)
            dphi_dt = (
                omega_t.unsqueeze(0).unsqueeze(-1)
                + coupling_phi
                + ecg_phase_input
            )

            r = r + dr_dt * self.dt
            r = torch.clamp(r, 1e-6, 10.0)
            phi = phi + dphi_dt * self.dt

            xy = torch.cat([r * torch.cos(phi), r * torch.sin(phi)], dim=1).squeeze(-1)
            trajectory.append(xy)

        r_final = r.squeeze(-1)
        phi_final = phi.squeeze(-1)
        out = torch.cat(
            [r_final * torch.cos(phi_final), r_final * torch.sin(phi_final)],
            dim=-1,
        )

        if return_trajectory:
            traj = torch.stack(trajectory, dim=1)
            return out, traj
        return out
