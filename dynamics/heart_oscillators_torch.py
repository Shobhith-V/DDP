"""PyTorch heart oscillator for differentiable feedback."""

import torch
import torch.nn as nn

# dr_i/dt = (alpha - r_i^2)*r_i + A_ij*r_j*cos(theta + n*(phi_j - phi_i)) + modulation_i
# dphi_i/dt = omega_i + A_ij*(r_j/r_i)*sin(theta + n*(phi_j - phi_i))


class HeartOscillatorTorch(nn.Module):
    """
    Differentiable two-oscillator heart model.
    modulation: (T, 2) tensor, time-varying. No .detach().
    """

    def __init__(
        self,
        omega1_hz: float = 1.0,
        omega2_hz: float = 1.2,
        alpha: float = 1.0,
        A_init: float = 0.0001,
        theta_init: float = 3.14159,
        n: float = 1.0,
    ):
        super().__init__()
        self.omega1 = 2 * torch.pi * omega1_hz
        self.omega2 = 2 * torch.pi * omega2_hz
        self.alpha = alpha
        self.A12 = A_init
        self.A21 = A_init
        self.theta12 = theta_init
        self.theta21 = theta_init
        self.n = n

    def forward(
        self,
        T: float,
        dt: float,
        modulation: torch.Tensor,
        r0: torch.Tensor | None = None,
        phi0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        modulation: (T_steps, 2) or (batch, T_steps, 2)
        Returns: (T_steps, 4) or (batch, T_steps, 4) = (x1,y1,x2,y2)
        """
        squeeze = False
        if modulation.dim() == 2:
            squeeze = True
            modulation = modulation.unsqueeze(0)

        batch = modulation.shape[0]
        n_steps = modulation.shape[1]
        device = modulation.device
        dtype = modulation.dtype

        r1 = torch.ones(batch, 1, device=device, dtype=dtype)
        r2 = torch.ones(batch, 1, device=device, dtype=dtype)
        phi1 = torch.zeros(batch, 1, device=device, dtype=dtype)
        phi2 = torch.zeros(batch, 1, device=device, dtype=dtype)
        if r0 is not None:
            r1 = r0[:, 0:1]
            r2 = r0[:, 1:2]
        if phi0 is not None:
            phi1 = phi0[:, 0:1]
            phi2 = phi0[:, 1:2]

        outputs = []
        for i in range(n_steps):
            mod = modulation[:, i, :]

            phase12 = self.theta12 + self.n * (phi2 - phi1)
            phase21 = self.theta21 + self.n * (phi1 - phi2)

            coupling12 = self.A12 * r2 * torch.cos(phase12)
            coupling21 = self.A21 * r1 * torch.cos(phase21)

            dr1 = (self.alpha - r1**2) * r1 + coupling12 + 0.1 * mod[:, 0:1]
            dr2 = (self.alpha - r2**2) * r2 + coupling21 + 0.1 * mod[:, 1:2]

            dphi1 = self.omega1 + self.A12 * (r2 / (r1 + 1e-8)) * torch.sin(phase12)
            dphi2 = self.omega2 + self.A21 * (r1 / (r2 + 1e-8)) * torch.sin(phase21)

            r1 = torch.clamp(r1 + dr1 * dt, 0.01, 2.0)
            r2 = torch.clamp(r2 + dr2 * dt, 0.01, 2.0)
            phi1 = phi1 + dphi1 * dt
            phi2 = phi2 + dphi2 * dt

            x1 = r1 * torch.cos(phi1)
            y1 = r1 * torch.sin(phi1)
            x2 = r2 * torch.cos(phi2)
            y2 = r2 * torch.sin(phi2)
            outputs.append(torch.cat([x1, y1, x2, y2], dim=-1))

        out = torch.stack(outputs, dim=1)
        if squeeze:
            out = out.squeeze(0)
        return out
