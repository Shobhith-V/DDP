"""
Brain Hopf ODE with standard phase_diff = phi_j - phi_i - theta_ij.
D_tensor precomputed; no NumPy inside ODE.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint


class ODEFunc(nn.Module):
    """
    ODE: dr/dt, dphi/dt, dtheta/dt, domega/dt, dalpha/dt.
    phase_diff_ij = phi_j - phi_i - theta_ij (standard Kuramoto-style).
    """

    def __init__(
        self,
        mu: float,
        eta_theta: float,
        eta_omega: float,
        eta_alpha: float,
        N: int,
        Sc: torch.Tensor,
        D_tensor: torch.Tensor,
        t_eval: torch.Tensor,
        mlp_model: nn.Module | None = None,
        hidden_repr: torch.Tensor | None = None,
    ):
        super().__init__()
        self.mu = mu
        self.eta_theta = eta_theta
        self.eta_omega = eta_omega
        self.eta_alpha = eta_alpha
        self.N = N
        self.register_buffer("Sc", Sc)
        self.register_buffer("D_tensor", D_tensor)
        self.register_buffer("t_eval", t_eval)
        self.mlp_model = mlp_model
        self.hidden_repr = hidden_repr

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        N = self.N
        r, phi = state[:N], state[N : 2 * N]
        theta = state[2 * N : 2 * N + N**2].view(N, N)
        omega = state[2 * N + N**2 : 3 * N + N**2]
        alpha = state[3 * N + N**2 : 4 * N + N**2]

        omega_safe = torch.clamp(omega, 2 * 3.14159 * 0.5, 2 * 3.14159 * 20)
        r = torch.clamp(r, 1e-1, 2.0)
        alpha = torch.clamp(alpha, -1.0, 1.0)
        r_safe = torch.clamp(r, 1e-5, 10.0)

        # phase_diff[i,j] = phi_j - phi_i - theta_ij
        phase_diff = phi[None, :] - phi[:, None] - theta

        t_val = t.item() if t.dim() == 0 else float(t)
        t_idx = (self.t_eval - t_val).abs().argmin().item()
        D = self.D_tensor[t_idx]

        P = torch.sum(alpha * r * torch.cos(phi))
        e = D - P

        ecg_input = torch.zeros(N, device=state.device, dtype=state.dtype)
        if self.mlp_model is not None and self.hidden_repr is not None:
            idx = min(int((t / self.t_eval[-1]).item() * (self.hidden_repr.shape[0] - 1)), self.hidden_repr.shape[0] - 1)
            idx = max(0, idx)
            feats = self.hidden_repr[idx].to(device=state.device, dtype=state.dtype)
            ecg_input = self.mlp_model(feats)
            ecg_input = torch.clamp(ecg_input.squeeze(), 0.01, 5.0)

        coupling_r = torch.sum(torch.abs(self.Sc) * r[None, :] * torch.cos(phase_diff), dim=1)
        coupling_phi = torch.sum(
            torch.abs(self.Sc) * (r[None, :] / (r_safe[:, None] + 1e-8)) * torch.sin(phase_diff),
            dim=1,
        )

        drdt = (self.mu - r**2) * r + coupling_r + e * torch.cos(phi) + ecg_input
        dphidt = omega_safe + coupling_phi - (e / (r_safe + 1e-8)) * torch.sin(phi)

        dthetadt = self.eta_theta * torch.sin(phase_diff) * torch.abs(self.Sc)
        domegadt = -self.eta_omega * e * torch.sin(phi)
        dalphadt = self.eta_alpha * e * r * torch.cos(phi)

        drdt = torch.clamp(drdt, -1e2, 1e2)
        dphidt = torch.clamp(dphidt, -1e2, 1e2)
        dthetadt = torch.clamp(dthetadt, -1e2, 1e2)
        domegadt = torch.clamp(domegadt, -1e2, 1e2)
        dalphadt = torch.clamp(dalphadt, -1e2, 1e2)

        return torch.cat(
            [
                drdt.flatten(),
                dphidt.flatten(),
                dthetadt.flatten(),
                domegadt.flatten(),
                dalphadt.flatten(),
            ]
        )


class TorchRevHopfNetwork:
    def __init__(
        self,
        mu: float,
        eta_omega: float,
        eta_alpha: float,
        eta_theta: float,
        D_tensor: torch.Tensor,
        t_eval: torch.Tensor,
        N: int,
        Sc: torch.Tensor,
        mlp_model: nn.Module | None = None,
        hidden_repr: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.N = N
        Sc_buf = Sc if isinstance(Sc, torch.Tensor) else torch.tensor(Sc, device=self.device, dtype=torch.float32)
        D_buf = D_tensor.to(self.device) if D_tensor.device != self.device else D_tensor
        t_buf = t_eval.to(self.device) if t_eval.device != self.device else t_eval
        hr = hidden_repr.to(self.device) if hidden_repr is not None else None

        self.ode_func = ODEFunc(
            mu=mu,
            eta_theta=eta_theta,
            eta_omega=eta_omega,
            eta_alpha=eta_alpha,
            N=N,
            Sc=Sc_buf,
            D_tensor=D_buf,
            t_eval=t_buf,
            mlp_model=mlp_model,
            hidden_repr=hr,
        ).to(self.device)

    def solve(
        self,
        r0,
        phi0,
        theta0,
        omega0,
        alpha0,
    ):
        import numpy as np

        y0 = torch.tensor(
            np.concatenate([r0, phi0, theta0.flatten(), omega0, alpha0]),
            device=self.device,
            dtype=torch.float32,
        )
        t_eval = self.ode_func.t_eval

        sol = odeint_adjoint(
            self.ode_func,
            y0,
            t_eval,
            method="rk4",
            rtol=1e-5,
            atol=1e-7,
        )

        N = self.N
        r = sol[:, :N]
        phi = sol[:, N : 2 * N]
        theta = sol[:, 2 * N : 2 * N + N**2].view(-1, N, N)
        omega = sol[:, 2 * N + N**2 : 3 * N + N**2]
        alpha = sol[:, 3 * N + N**2 : 4 * N + N**2]

        rcos_phi = torch.sum(r * torch.cos(phi), dim=1)
        return r, phi, theta, omega, alpha, rcos_phi
