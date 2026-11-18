"""Release audit catches altered/mismatched metric artifacts."""

import json

import pytest

from scripts.audit_release import sha256_file, validate_evaluation_report, validate_patient_csv


def _report():
    cases = {
        "CASE_A": {"true_positive_voxels": 3, "false_positive_voxels": 1,
                   "false_negative_voxels": 0, "dice": 6 / 7, "iou": 0.75,
                   "precision": 0.75, "recall": 1.0},
        "CASE_B": {"true_positive_voxels": 1, "false_positive_voxels": 0,
                   "false_negative_voxels": 1, "dice": 2 / 3, "iou": 0.5,
                   "precision": 1.0, "recall": 0.5},
    }
    return {
        "evaluation_partition": "test", "split_hash": "test_hash", "checkpoint_epoch": 24,
        "decision_threshold": 0.5, "patients_evaluated": 2,
        "exploratory_checkpoint": False, "exploratory_evaluation": False,
        "cases": cases,
        **{f"mean_{metric}": sum(c[metric] for c in cases.values()) / 2
           for metric in ("dice", "iou", "precision", "recall")},
    }


def _validate(report):
    return validate_evaluation_report(report, partition="test",
                                      expected_ids={"CASE_A", "CASE_B"},
                                      split_hash="test_hash", checkpoint_epoch=24,
                                      threshold=0.5)


def test_release_report_consistency():
    summary = _validate(_report())
    assert summary["patients"] == 2
    assert summary["mean_dice"] == pytest.approx((6 / 7 + 2 / 3) / 2)


def test_release_report_fails_for_perturbed_score():
    r = _report()
    r["cases"]["CASE_A"]["dice"] = 0.9
    with pytest.raises(ValueError, match="Incorrect dice"):
        _validate(r)


def test_release_report_fails_for_extra_case_and_split_hash():
    r = _report()
    r["cases"]["CASE_C"] = r["cases"]["CASE_B"]
    with pytest.raises(ValueError, match="Missing or extra"):
        _validate(r)
    r = _report()
    r["split_hash"] = "wrong"
    with pytest.raises(ValueError, match="Split hash"):
        _validate(r)


def test_release_file_sha256(tmp_path):
    import hashlib
    p = tmp_path / "file.txt"
    p.write_bytes(b"test")
    assert sha256_file(p) == hashlib.sha256(b"test").hexdigest()


def test_patient_csv_matches_report(tmp_path):
    import csv
    cases = _report()["cases"]
    path = tmp_path / "per_patient.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", *cases["CASE_A"]])
        writer.writeheader()
        for cid, score in cases.items():
            writer.writerow({"case_id": cid, **score})
    validate_patient_csv(path, cases)
    altered = path.read_text().replace("0.75", "0.11", 1)
    path.write_text(altered)
    with pytest.raises(ValueError, match="metric mismatch"):
        validate_patient_csv(path, cases)
