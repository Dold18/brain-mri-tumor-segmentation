"""Loss components for binary whole-tumor-region segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        if not 0 <= dice_weight <= 1:
            raise ValueError("dice_weight must lie in [0,1]")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"Logits {tuple(logits.shape)} != target {tuple(targets.shape)}")
        if not bool(torch.all((targets == 0) | (targets == 1)).item()):
            raise ValueError("Binary segmentation targets must be exactly 0 or 1")
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        probs = torch.sigmoid(logits).flatten(start_dim=1)
        truth = targets.float().flatten(start_dim=1)
        intersection = (probs * truth).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + truth.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice.mean()
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce
