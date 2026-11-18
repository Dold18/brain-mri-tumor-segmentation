"""Run a tiny CPU neural-network optimization test without downloading MRI data."""

from __future__ import annotations

import math

import torch

from brain_mri.losses import DiceBCELoss
from brain_mri.model import UNet2D


def run_smoke_test(steps: int = 3) -> tuple[float, float]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(2026)
    model = UNet2D(in_channels=4, base_channels=8).cpu().train()
    size = 48
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    circle = (((xx - 24) ** 2 + (yy - 24) ** 2) <= 9**2).float()
    targets = circle[None, None]
    features = 0.05 * torch.randn(1, 4, size, size)
    features[0, 0] += circle * 2
    features[0, 1] += circle * 0.8
    features[0, 2] += circle * 1.2
    features[0, 3] += circle * 1.8
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = DiceBCELoss()
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Neural network produced a nonfinite loss")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    if not all(math.isfinite(value) for value in history):
        raise RuntimeError("Smoke-test loss is not finite")
    return history[0], history[-1]


if __name__ == "__main__":
    first, last = run_smoke_test()
    print(f"CPU smoke test passed | initial loss={first:.4f} | final loss={last:.4f}")
    print("Synthetic signals only; these values are NOT medical evaluation results.")
