"""
Neural network models for ECG prediction and feedback.
"""

from typing import Optional

import torch
import torch.nn as nn


class HeartModel(nn.Module):
    """
    MLP that maps oscillator state vectors → ECG samples.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 100,
        feature_dim: int = 50,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.output_layer = nn.Linear(feature_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.output_layer(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the internal feature representation (before final layer)."""
        return self.feature_extractor(x)


class ECGToOscillatorMLP(nn.Module):
    """
    MLP that maps heart features → brain oscillator parameters.
    """

    def __init__(
        self,
        input_dim: int = 50,
        hidden1: int = 64,
        hidden2: int = 64,
        output_dim: int = 16,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.act = nn.Sigmoid()

    def forward(
        self, hidden_repr: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h1 = self.act(self.fc1(hidden_repr))
        h2 = self.act(self.fc2(h1))
        out = self.fc3(h2)
        return out


class FeedbackMLP(nn.Module):
    """
    MLP that takes ONLY brain r⋅cos(ϕ) (rcos_phi) as input for heart modulation.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, rcos_phi: torch.Tensor) -> torch.Tensor:
        # Ensure shape [time, 1] for 1D input sequences
        if rcos_phi.dim() == 1:
            rcos_phi = rcos_phi.unsqueeze(1)
        # Clamp output for stability (as in the notebook)
        return torch.clamp(self.net(rcos_phi), -0.5, 0.5)

