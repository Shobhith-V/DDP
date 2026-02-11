"""Heart model: oscillator state -> ECG prediction."""

import torch.nn as nn


class HeartModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 100,
        feature_dim: int = 50,
        output_dim: int = 1,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.output_layer = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.output_layer(features)

    def get_features(self, x):
        return self.feature_extractor(x)
