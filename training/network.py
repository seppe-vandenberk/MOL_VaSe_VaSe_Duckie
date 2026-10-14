"""Small CNN that supports 32x32 inputs: convolutions -> 128 features."""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DrivingCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        channels, height, width = observation_space.shape
        self.convolutions = nn.Sequential(
            nn.Conv2d(channels, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            size = self.convolutions(torch.zeros(1, channels, height, width)).shape[1]
        self.features = nn.Sequential(nn.Linear(size, 128), nn.ReLU())

    def forward(self, images):
        # SB3 already normalizes uint8 pixels to [0, 1].
        return self.features(self.convolutions(images))
