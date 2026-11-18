"""Dependency-light visualizations for segmentation overlays."""

from __future__ import annotations

import numpy as np


def grayscale_uint8(slice_image: np.ndarray) -> np.ndarray:
    """Convert one MRI modality slice into a display-only 8-bit grayscale image."""
    x = np.asarray(slice_image, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Display image must be a finite 2D array")
    nonzero = x[x != 0]
    if nonzero.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    low, high = np.percentile(nonzero, (1, 99))
    if high <= low:
        return np.zeros(x.shape, dtype=np.uint8)
    # Background voxels are exactly zero after MRI preprocessing; keep them black.
    display = (255 * np.clip((x - low) / (high - low), 0, 1)).astype(np.uint8)
    display[x == 0] = 0
    return display


def color_overlay(image: np.ndarray, binary_mask: np.ndarray, *,
                  color: tuple[int, int, int] = (230, 68, 63),
                  alpha: float = 0.42) -> np.ndarray:
    """Overlay an aligned binary mask on a display-only 2D MRI. Returns RGB uint8."""
    m = np.asarray(binary_mask)
    if image.ndim != 2 or m.shape != image.shape:
        raise ValueError("Image and segmentation must be aligned 2D arrays")
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between zero and one")
    if not set(np.unique(m)).issubset({0, 1, False, True}):
        raise ValueError("Overlay requires a binary mask")
    if any(not isinstance(c, int) or not 0 <= c <= 255 for c in color):
        raise ValueError("Color must be RGB integers between 0 and 255")
    base = np.repeat(grayscale_uint8(image)[..., None], 3, axis=-1)
    blended = base.astype(np.float32)
    mask = m.astype(bool)
    blended[mask] = (1.0 - alpha) * blended[mask] + alpha * np.array(color)
    return np.rint(blended).astype(np.uint8)


def save_case_comparison(path, image: np.ndarray, truth: np.ndarray,
                         prediction: np.ndarray, *, case_id: str,
                         slice_index: int) -> None:
    """Export a real scan/prediction/annotation comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("MRI (normalized)", grayscale_uint8(image)),
        ("Model prediction", color_overlay(image, prediction)),
        ("Ground-truth annotation", color_overlay(image, truth, color=(50, 205, 220))),
    ]
    for ax, (label, pixels) in zip(axes, panels):
        ax.imshow(pixels, cmap="gray" if pixels.ndim == 2 else None)
        ax.set_title(label)
        ax.set_axis_off()
    fig.suptitle(f"{case_id} · axial slice {slice_index} · research-only segmentation")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
