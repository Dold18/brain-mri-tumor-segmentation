"""Presentation-only tests: no clinical data, no Streamlit runtime needed."""

import numpy as np
import pytest

from brain_mri.viewer import disagreement_overlay, foreground_slice_index, patient_metrics
from brain_mri.visualization import grayscale_uint8


def test_patient_metrics_use_entire_volume():
    probability = np.zeros((2, 16, 16), dtype=np.float32)
    reference = np.zeros_like(probability, dtype=np.uint8)
    probability[0, 1, 1] = 0.90  # correct
    probability[0, 2, 2] = 0.80  # extra
    reference[0, 1, 1] = 1
    reference[1, 4, 4] = 1  # missed on a *different* slice
    metrics = patient_metrics(probability, reference, threshold=0.5)
    assert metrics["true_positive_voxels"] == 1
    assert metrics["false_positive_voxels"] == 1
    assert metrics["false_negative_voxels"] == 1
    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_patient_metrics_reject_invalid_inputs():
    p = np.zeros((2, 16, 16), dtype=np.float32)
    t = np.zeros_like(p, dtype=np.uint8)
    with pytest.raises(ValueError, match="aligned"):
        patient_metrics(p, t[:1], threshold=0.5)
    with pytest.raises(ValueError, match="threshold"):
        patient_metrics(p, t, threshold=0.0)
    p[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        patient_metrics(p, t, threshold=0.5)


def test_disagreement_overlay_has_distinct_error_colors():
    im = np.ones((16, 16), dtype=np.float32)
    pred = np.zeros((16, 16), dtype=np.uint8)
    truth = np.zeros_like(pred)
    pred[1, 1] = truth[1, 1] = 1  # TP
    pred[2, 2] = 1  # FP
    truth[3, 3] = 1  # FN
    original = im.copy()
    out = disagreement_overlay(im, pred, truth, alpha=1.0)
    assert tuple(out[1, 1]) == (42, 204, 151)
    assert tuple(out[2, 2]) == (242, 93, 105)
    assert tuple(out[3, 3]) == (71, 171, 250)
    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(im, original)
    with pytest.raises(ValueError, match="aligned"):
        disagreement_overlay(im, pred[:10], truth)


def test_default_slice_finds_largest_label_and_handles_empty():
    m = np.zeros((5, 32, 32), dtype=np.uint8)
    m[1, 2:5, 2:5] = 1
    m[3, 2:8, 2:8] = 1
    assert foreground_slice_index(m) == 3
    assert foreground_slice_index(np.zeros_like(m)) == 2


def test_visualization_background_is_black():
    m = np.zeros((20, 20), dtype=np.float32)
    m[5:15, 5:15] = np.linspace(-1, 1, 100).reshape(10, 10)
    displayed = grayscale_uint8(m)
    assert displayed[0, 0] == 0
    assert displayed.dtype == np.uint8
