from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


MODEL_VERSION = "v2_spatial_depthwise_residual"


class SpatialCNN(nn.Module):
    """A compact CNN that preserves coarse spatial layout before classification."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DepthwiseCNN(nn.Module):
    """A parameter-efficient CNN with depthwise separable convolutions."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(40, 64),
            DepthwiseSeparableBlock(64, 96, stride=2),
            DepthwiseSeparableBlock(96, 128, stride=2),
            DepthwiseSeparableBlock(128, 160),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(160 * 2 * 2, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class ResidualCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 48, stride=2),
            ResidualBlock(48, 48),
            ResidualBlock(48, 72, stride=2),
            ResidualBlock(72, 72),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(72 * 2 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(self.stem(inputs)).flatten(1))


MODEL_BUILDERS: dict[str, Callable[[int, int], nn.Module]] = {
    "cnn1_spatial": SpatialCNN,
    "cnn2_depthwise": DepthwiseCNN,
    "cnn3_residual": ResidualCNN,
}


def build_models(in_channels: int, num_classes: int) -> dict[str, nn.Module]:
    return {
        name: builder(in_channels, num_classes) for name, builder in MODEL_BUILDERS.items()
    }
