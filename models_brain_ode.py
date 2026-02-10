"""
ODE-based brain model (reversed Hopf network) and solver.
"""

from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

from config import use_half_precision


class ODEFuc(nn.Module):
    """
    ODE function for the reversed Hopf network with plastic parameters.

    The implementation follows the original notebook closely and exposes
    a call signature compatible with `odeint_adjoint`.
    """

    def __init__(
        self,
        mu: float,
        eta_theta: float,
        eta_omega: float,
        eta_alpha: float,
        D_function: Callable[[float], float],
        N: int,
        Sc: torch.Tensor,
        mlp_model: nn.Module | None = None,
        hidden_repr: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.eta_theta = eta_theta
        self.eta_omega = eta_omega
        self.eta_alpha = eta_alpha
        self.D_function = D_function
        self.N = N
        self.register_buffer("Sc", Sc)
        self.mlp_model = mlp_model
        self.hidden_repr = hidden_repr

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        N = self.N
        r, phi = state[:N], state[N : 2 * N]
        theta = state[2 * N : 2 * N + N**2].view(N, N)
        omega = state[2 * N + N**2 : 3 * N + N**2]
        alpha = state[3 * N + N**2 : 4 * N + N**2]

        omega_safe = torch.clamp(omega, 2 * np.pi * 0.5, 2 * np.pi * 20)
        r = torch.clamp(r, 1e-1, 2.0)
        alpha = torch.clamp(alpha, -1.0, 1.0)
        r_safe = torch.clamp(
            torch.where(
                r < 1e-6,
                torch.tensor(1e-6, device=r.device, dtype=r.dtype),
                r,
            ),
            1e-5,
            10.0,
        )
        r = r_safe

        phase_diff = torch.clamp(
            phi[None, :] / omega_safe[None, :]
            - phi[:, None] / omega_safe[:, None]
            + theta / (omega_safe[:, None] * omega_safe[None, :]),
            -1e2,
            1e2,
        )

        D = torch.tensor(self.D_function(float(t.item())), device=state.device, dtype=state.dtype)
        P = torch.sum(alpha * r * torch.cos(phi))
        e = D - P

        # Placeholder for ECGlike input modulation, kept from the original
        ecg_input = torch.zeros(N, device=state.device, dtype=state.dtype)

        # Plasticity updates
        drdt = self.mu * r - r**3 + torch.sum(
            self.Sc * alpha[None, :] * r[None, :] * torch.cos(phase_diff), dim=1
        )
        dphidt = omega_safe + torch.sum(
            self.Sc * alpha[None, :] * r[None, :] * torch.sin(phase_diff) / r[:, None],
            dim=1,
        )

        # Hebbian learning rules (same as notebook)
        dthetadt = self.eta_theta * e * self.Sc * r[:, None] * r[None, :]
        domegadt = self.eta_omega * e * torch.ones_like(omega_safe)
        dalphadt = self.eta_alpha * e * r * torch.cos(phi)

        drdt = torch.clamp(drdt, -1e2, 1e2)
        dphidt = torch.clamp(dphidt, -1e2, 1e2)
        dthetadt = torch.clamp(dthetadt, -1e2, 1e2)
        domegadt = torch.clamp(domegadt, -1e2, 1e2)
        dalphadt = torch.clamp(dalphadt, -1e2, 1e2)

        return torch.cat(
            [drdt, dphidt, dthetadt.flatten(), domegadt, dalphadt]
        )


class TorchRevHopfNetwork:
    """
    High‑level wrapper around `ODEFuc` to integrate the system in time.
    """

    def __init__(
        self,
        mu: float,
        eta_omega: float,
        eta_alpha: float,
        eta_theta: float,
        D_function: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
        N: int,
        Sc: np.ndarray,
        mlp_model: nn.Module | None,
        hidden_repr: torch.Tensor | None,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.N = N
        self.ode_func = ODEFuc(
            mu=mu,
            eta_theta=eta_theta,
            eta_omega=eta_omega,
            eta_alpha=eta_alpha,
            D_function=D_function,
            N=N,
            Sc=torch.tensor(Sc, device=self.device, dtype=torch.float32),
            mlp_model=mlp_model,
            hidden_repr=hidden_repr.to(self.device) if hidden_repr is not None else None,
        ).to(self.device)

    def solve(
        self,
        r0: np.ndarray,
        phi0: np.ndarray,
        theta0: np.ndarray,
        omega0: np.ndarray,
        alpha0: np.ndarray,
        t_eval: np.ndarray,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Integrate the ODE from given initial conditions.

        Returns r, phi, theta, omega, alpha and the scalar brain output
        r⋅cos(ϕ) over time.
        """
        dtype = torch.float16 if use_half_precision else torch.float32
        y0_np = np.concatenate(
            [r0, phi0, theta0.flatten(), omega0, alpha0]
        )
        y0 = torch.tensor(y0_np, device=self.device, dtype=dtype)
        t_eval_tensor = torch.tensor(t_eval, device=self.device, dtype=dtype)

        sol = odeint_adjoint(
            self.ode_func, y0, t_eval_tensor, method="rk4", rtol=1e-5, atol=1e-7
        )

        N = self.N
        r = sol[:, :N]
        phi = sol[:, N : 2 * N]
        theta = sol[:, 2 * N : 2 * N + N**2].view(-1, N, N)
        omega = sol[:, 2 * N + N**2 : 3 * N + N**2]
        alpha = sol[:, 3 * N + N**2 : 4 * N + N**2]

        # Brain output signal used for feedback: sum_k α_k r_k cos(ϕ_k)
        rcos_phi = torch.sum(r * torch.cos(phi), dim=1)

        return r, phi, theta, omega, alpha, rcos_phi

