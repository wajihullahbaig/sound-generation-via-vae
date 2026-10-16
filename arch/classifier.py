import torch
import torch.nn as nn
from typing import Sequence


class DigitClassifier(nn.Module):
    """Small CNN digit classifier; its pooled features feed the Frechet distance."""

    def __init__(self, num_classes: int = 10, channels: Sequence[int] = (32, 64, 128, 128), dropout: float = 0.3):
        super().__init__()
        layers, c = [], 1
        for ch in channels:
            layers += [nn.Conv2d(c, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(), nn.MaxPool2d(2)]
            c = ch
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
