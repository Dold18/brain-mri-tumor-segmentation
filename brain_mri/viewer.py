"""Presentation-only, testable helpers for the research MRI viewer.

The trained checkpoint and binary decision threshold are never changed here.
Patient-level metrics use all slices; displayed slices are illustrations only.
"""

from __future__ import annotations

import numpy as np

from .metrics import Confusion
from .visualization import grayscale_uint8


def patient_metrics(probabilities: np.ndarray, truth: np.ndarray, *, threshold: float) -> dict:
    """Compute whole-patient voxel metrics for aligned `[Z,H,W]` arrays."""
    p = np.asarray(probabilities)
    t = np.asarray(truth)
    if p.ndim != 3 or p.shape != t.shape:
        raise ValueError("Probability and reference volumes must be aligned [Z,H,W] arrays")
    if not 0 < threshold < 1 or not np.isfinite(p).all():
        raise ValueError("Invalid threshold or nonfinite probability data")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be within [0,1]")
    if not np.isin(t, [0, 1]).all():
        raise ValueError("The reference mask must be binary")
    confusion = Confusion()
    confusion.add(p >= threshold, t.astype(bool))
    return confusion.result()


def disagreement_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    alpha: float = 0.72,
) -> np.ndarray:
    """Render aligned segmentation agreement: TP green, FP red, FN blue.

    The grayscale input is only a display background, never an inference input.
    Input arrays are not modified.
    """
    x = np.asarray(image)
    p = np.asarray(prediction)
    t = np.asarray(truth)
    if x.ndim != 2 or p.shape != x.shape or t.shape != x.shape:
        raise ValueError("MRI and masks must be aligned 2D arrays")
    if not 0 <= alpha <= 1:
        raise ValueError("Opacity must fall in [0,1]")
    if not np.isin(p, [0, 1]).all() or not np.isin(t, [0, 1]).all():
        raise ValueError("Masks must be binary")

    base = np.repeat(grayscale_uint8(x)[..., None], 3, axis=2).astype(np.float32)
    foreground = p.astype(bool) | t.astype(bool)
    palette = np.zeros_like(base)
    both = p.astype(bool) & t.astype(bool)
    extra = p.astype(bool) & ~t.astype(bool)
    missed = ~p.astype(bool) & t.astype(bool)
    palette[both] = (42, 204, 151)       # true positive: green
    palette[extra] = (242, 93, 105)      # false positive: red
    palette[missed] = (71, 171, 250)     # false negative: blue
    base[foreground] = (1 - alpha) * base[foreground] + alpha * palette[foreground]
    return np.rint(base).astype(np.uint8)


def foreground_slice_index(masks: np.ndarray) -> int:
    """Select the slice with the largest reference region, or the middle slice."""
    m = np.asarray(masks)
    if m.ndim != 3 or m.shape[0] == 0:
        raise ValueError("Expected nonempty reference mask [Z,H,W]")
    counts = np.count_nonzero(m, axis=(1, 2))
    if counts.max() == 0:
        return len(m) // 2
    return int(np.argmax(counts))
