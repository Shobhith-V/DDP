"""FeedbackMLP: brain rcos_phi -> time-varying heart modulation. No clamp that breaks gradients."""

import torch
import torch.nn as nn


class FeedbackMLP(nn.Module):
    """rcos_phi[t] -> modulation[t] for each time step. Tanh for bounded output."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, rcos_phi: torch.Tensor) -> torch.Tensor:
        """rcos_phi: (T,) or (batch, T) or (T, 1). Returns (T, 2) or (batch, T, 2)."""
        if rcos_phi.dim() == 1:
            rcos_phi = rcos_phi.unsqueeze(-1)
        out = self.net(rcos_phi)
        return 0.5 * torch.tanh(out)
