import numpy as np
import pytest

from brain_mri.preprocessing import axial_slice, normalize_mri, whole_tumor_mask


def _volume():
    rng = np.random.default_rng(7)
    image = np.zeros((18, 20, 5, 4), dtype=np.float32)
    image[2:-2, 3:-3] = rng.normal(70, 10, (14, 14, 5, 4)).astype(np.float32)
    raw = np.zeros(image.shape[:3], dtype=np.uint8)
    raw[6:10, 7:12, 2:4] = 2
    raw[8:11, 9:12, 3:4] = 3
    return image, raw


def test_shape_mask_alignment_and_background():
    volume, labels = _volume()
    processed = normalize_mri(volume)
    binary = whole_tumor_mask(labels)
    features, target = axial_slice(processed, binary, 3)
    assert processed.shape == volume.shape
    assert features.shape == (4, 18, 20)
    assert target.shape == (1, 18, 20)
    assert target[0, 9, 10] == 1
    assert target[0, 0, 0] == 0
    assert np.all(processed[0] == 0)
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed[2:-2, 3:-3].mean(axis=(0, 1, 2)))) < 1e-4


def test_input_validation():
    volume, labels = _volume()
    with pytest.raises(ValueError, match="nonzero foreground"):
        normalize_mri(np.zeros_like(volume))
    with pytest.raises(ValueError, match="NaN"):
        normalize_mri(np.full_like(volume, np.nan))
    with pytest.raises(ValueError, match="integral"):
        whole_tumor_mask(labels.astype(float) + 0.1)
    with pytest.raises(IndexError):
        axial_slice(normalize_mri(volume), whole_tumor_mask(labels), 99)
