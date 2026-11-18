"""Original PyTorch implementation of a lightweight 2D U-Net."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels: int) -> nn.GroupNorm:
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Keep the original pixel resolution even for odd MRI image dimensions.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2D(nn.Module):
    """Four-level U-Net producing uncalibrated logits of shape `[B,1,H,W]`."""

    def __init__(self, in_channels: int = 4, base_channels: int = 16) -> None:
        super().__init__()
        if in_channels <= 0 or base_channels <= 0:
            raise ValueError("Channel counts must be positive")
        c = base_channels
        self.in_channels = in_channels
        self.encoder1 = DoubleConv(in_channels, c)
        self.encoder2 = Down(c, 2 * c)
        self.encoder3 = Down(2 * c, 4 * c)
        self.encoder4 = Down(4 * c, 8 * c)
        self.bottleneck = Down(8 * c, 16 * c)
        self.decoder4 = Up(16 * c, 8 * c, 8 * c)
        self.decoder3 = Up(8 * c, 4 * c, 4 * c)
        self.decoder2 = Up(4 * c, 2 * c, 2 * c)
        self.decoder1 = Up(2 * c, c, c)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B,{self.in_channels},H,W], found {tuple(x.shape)}")
        if min(x.shape[-2:]) < 16:
            raise ValueError("Input height and width must be >= 16 for four downsampling stages")
        s1 = self.encoder1(x)
        s2 = self.encoder2(s1)
        s3 = self.encoder3(s2)
        s4 = self.encoder4(s3)
        x = self.bottleneck(s4)
        x = self.decoder4(x, s4)
        x = self.decoder3(x, s3)
        x = self.decoder2(x, s2)
        x = self.decoder1(x, s1)
        return self.head(x)
