import torch
import pytest

from brain_mri.losses import DiceBCELoss
from brain_mri.model import UNet2D
from scripts.smoke_test import run_smoke_test


def test_unet_output_shape_with_odd_spatial_dimensions():
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = UNet2D(in_channels=4, base_channels=8)
    x = torch.randn(1, 4, 33, 39)
    assert model(x).shape == (1, 1, 33, 39)


def test_forward_backward_and_finite_loss():
    model = UNet2D(in_channels=4, base_channels=8)
    features = torch.randn(1, 4, 32, 32)
    targets = torch.zeros(1, 1, 32, 32)
    targets[..., 8:16, 10:18] = 1
    loss = DiceBCELoss()(model(features), targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_invalid_target_or_input_rejected():
    model = UNet2D(4, 8)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(1, 3, 32, 32))
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        DiceBCELoss()(torch.ones(1, 1, 16, 16), torch.full((1, 1, 16, 16), 0.5))


def test_cpu_optimizer_smoke_test():
    initial, final = run_smoke_test(steps=3)
    assert 0 < initial < 5
    assert 0 < final < 5
