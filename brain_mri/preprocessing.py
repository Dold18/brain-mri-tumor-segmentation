"""Deterministic MRI preprocessing and axial slice conversion."""

from __future__ import annotations

import numpy as np


def normalize_mri(
    image: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Clip/standardize each modality using foreground voxels of the full 3D volume.

    Accepts `[X,Y,Z,C]`. Outside-brain background is fixed to zero. Compute
    these statistics separately for each subject, never across dataset splits.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 4 or image.shape[-1] < 1:
        raise ValueError(f"Expected [X,Y,Z,C] MRI, got shape {image.shape}")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    if not np.isfinite(image).all():
        raise ValueError("MRI contains NaN or infinity")

    foreground = np.any(image != 0, axis=-1)
    if not foreground.any():
        raise ValueError("MRI has no nonzero foreground voxels")
    output = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        voxels = image[..., channel][foreground]
        p_low, p_high = np.percentile(voxels, [lower_percentile, upper_percentile])
        if p_high - p_low < eps:
            raise ValueError(f"MRI channel {channel} is nearly constant in foreground")
        clipped = np.clip(voxels, p_low, p_high)
        mean, std = float(clipped.mean()), float(clipped.std())
        if std < eps:
            raise ValueError(f"MRI channel {channel} has zero foreground variance")
        output[..., channel][foreground] = (clipped - mean) / std
    return output


def whole_tumor_mask(raw_labels: np.ndarray) -> np.ndarray:
    """Combine every non-background label, including edema, into binary ROI."""
    labels = np.asarray(raw_labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 3D segmentation, found {labels.shape}")
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("Invalid segmentation label values")
    if not np.all(labels == np.rint(labels)):
        raise ValueError("Segmentation must contain integral labels")
    return np.asarray(labels != 0, dtype=np.uint8)


def axial_slice(
    normalized_image: np.ndarray, binary_mask: np.ndarray, index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a 2D input `[C,X,Y]` and target `[1,X,Y]` for axial Z slice."""
    if normalized_image.ndim != 4 or binary_mask.shape != normalized_image.shape[:3]:
        raise ValueError("Expected image [X,Y,Z,C] and aligned binary mask [X,Y,Z]")
    if not 0 <= index < normalized_image.shape[2]:
        raise IndexError(f"Axial slice {index} is out of range")
    if not set(np.unique(binary_mask)).issubset({0, 1}):
        raise ValueError("axial_slice requires a binary mask")
    features = np.ascontiguousarray(normalized_image[:, :, index, :].transpose(2, 0, 1))
    target = np.ascontiguousarray(binary_mask[:, :, index][None], dtype=np.float32)
    return features, target
