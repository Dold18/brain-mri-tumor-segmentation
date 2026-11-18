"""Synthetic unit tests for validation-only threshold selection."""
import numpy as np
import pytest

from scripts.validation_threshold_sweep import score_case, pick_threshold


def test_score_case_uses_complete_volume():
    p = np.array([[[0.90, 0.60], [0.45, 0.05]]], dtype=np.float32)
    truth = np.array([[[1, 1], [0, 0]]], dtype=np.uint8)
    scores = score_case(p, truth, (0.4, 0.5, 0.7))
    assert scores[0.5].result()["dice"] == pytest.approx(1.0)
    assert scores[0.4].result()["false_positive_voxels"] == 1
    assert scores[0.7].result()["false_negative_voxels"] == 1


def test_score_case_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        score_case(np.ones((1, 2, 2)), np.ones((2, 2, 2)), (0.5,))


def test_score_case_rejects_invalid_probabilities_or_threshold():
    arr = np.ones((1, 2, 2))
    with pytest.raises(ValueError):
        score_case(arr * 1.1, np.ones_like(arr), (0.5,))
    with pytest.raises(ValueError):
        score_case(arr, np.ones_like(arr), (1.0,))


def test_threshold_selection_prefers_baseline_on_tie():
    scores = {0.4: {"mean_dice": 0.8}, 0.5: {"mean_dice": 0.8},
              0.6: {"mean_dice": 0.8}}
    assert pick_threshold(scores, 0.5) == 0.5


def test_threshold_selection_optimizes_patient_mean():
    scores = {0.4: {"mean_dice": 0.87}, 0.5: {"mean_dice": 0.85}}
    assert pick_threshold(scores, 0.5) == 0.4
