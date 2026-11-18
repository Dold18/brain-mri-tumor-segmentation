"""Audit the frozen checkpoint, patient split and original evaluation reports.

This is a read-only verification of model/reports followed by a small JSON audit.
It never retrains, re-evaluates, or accesses individual MRI voxels. Run it on
macOS after final held-out evaluation. Never upload raw MRI scans to GitHub.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from brain_mri.data import read_catalog
from brain_mri.inference import load_checkpoint_model
from brain_mri.splits import validate_splits
from brain_mri.storage_paths import (
    BASELINE_RUN_DIR, DATASET_DIR, SPLIT_MANIFEST, require_ssd_if_targeted,
)
from brain_mri.train import _hash, load_config


DEFAULT_VAL = BASELINE_RUN_DIR / "validation_review" / "metrics.json"
DEFAULT_TEST = BASELINE_RUN_DIR / "final_test_epoch24" / "metrics.json"
DEFAULT_CHECKPOINT = BASELINE_RUN_DIR / "best.pt"
DEFAULT_OUTPUT = BASELINE_RUN_DIR / "release_audit.json"
DEFAULT_VAL_CSV = BASELINE_RUN_DIR / "validation_review" / "per_patient.csv"
DEFAULT_TEST_CSV = BASELINE_RUN_DIR / "final_test_epoch24" / "per_patient.csv"


def sha256_file(path: Path) -> str:
    """Compute artifact fingerprint without loading large checkpoints into RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_evaluation_report(report: dict, *, partition: str,
                               expected_ids: set[str], split_hash: str,
                               checkpoint_epoch: int, threshold: float) -> dict:
    """Verify the original evaluation report is internally consistent.

    This checks reporting integrity, not the correctness of medical predictions.
    """
    if report.get("evaluation_partition") != partition:
        raise ValueError(f"Expected {partition} report")
    if report.get("split_hash") != split_hash:
        raise ValueError(f"Split hash mismatch in {partition} report")
    if report.get("checkpoint_epoch") != checkpoint_epoch:
        raise ValueError(f"Checkpoint epoch mismatch in {partition} report")
    if not np.isclose(report.get("decision_threshold", -1), threshold, atol=1e-12):
        raise ValueError(f"Decision threshold mismatch in {partition} report")
    if report.get("exploratory_checkpoint") is not False or report.get("exploratory_evaluation") is not False:
        raise ValueError(f"Exploratory {partition} results are not release results")
    cases = report.get("cases", {})
    if set(cases) != expected_ids or report.get("patients_evaluated") != len(expected_ids):
        raise ValueError(f"Missing or extra patients in {partition} report")
    keys = ("dice", "iou", "recall", "precision")
    for case_id, data in cases.items():
        if any(not np.isfinite(float(data[key])) or not 0 <= float(data[key]) <= 1
               for key in keys):
            raise ValueError(f"Invalid patient metric for {case_id}")
        tp, fp, fn = (int(data[k]) for k in (
            "true_positive_voxels", "false_positive_voxels", "false_negative_voxels"))
        if min(tp, fp, fn) < 0:
            raise ValueError(f"Negative confusion count for {case_id}")
        union = tp + fp + fn
        denom = 2 * tp + fp + fn
        expected_dice = 2 * tp / denom if denom else 1.0
        expected_iou = tp / union if union else 1.0
        expected_precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
        expected_recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
        for key, calculated in zip(keys, (expected_dice, expected_iou,
                                           expected_recall, expected_precision)):
            if not np.isclose(float(data[key]), calculated, atol=1e-9):
                raise ValueError(f"Incorrect {key} for {case_id}")
    summary = {}
    for key in keys:
        actual_mean = float(np.mean([float(d[key]) for d in cases.values()]))
        if not np.isclose(actual_mean, report[f"mean_{key}"], atol=1e-9):
            raise ValueError(f"Aggregated {key} does not match individual {partition} scores")
        summary[f"mean_{key}"] = actual_mean
    summary["patients"] = len(expected_ids)
    return summary


def validate_patient_csv(path: Path, cases: dict) -> None:
    """Verify that the human-readable CSV matches the original JSON case records."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(cases) or {r["case_id"] for r in rows} != set(cases):
        raise ValueError(f"Patient CSV case IDs differ from JSON report: {path}")
    for row in rows:
        data = cases[row["case_id"]]
        for key, original in data.items():
            if key not in row or not np.isclose(float(row[key]), float(original), atol=1e-9):
                raise ValueError(f"Patient CSV metric mismatch: {row['case_id']} {key}")


def audit_release(*, checkpoint: Path, config: Path, dataset_dir: Path,
                  split_manifest: Path, validation_report: Path,
                  test_report: Path, validation_csv: Path, test_csv: Path, output: Path) -> dict:
    paths = {
        "checkpoint": Path(checkpoint), "config": Path(config),
        "split_manifest": Path(split_manifest),
        "validation_report": Path(validation_report), "test_report": Path(test_report),
        "validation_csv": Path(validation_csv), "test_csv": Path(test_csv),
    }
    for value in (*paths.values(), dataset_dir, output):
        require_ssd_if_targeted(value)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    catalog = read_catalog(dataset_dir)
    split_data = json.loads(split_manifest.read_text(encoding="utf-8"))
    validate_splits(split_data, {r.case_id for r in catalog.cases})
    counts = {k: len(split_data[k]) for k in ("train", "validation", "test")}
    split_hash = _hash(split_data)
    _, state = load_checkpoint_model(checkpoint, device="cpu")
    provenance = state.get("provenance", {})
    if provenance.get("split_hash") != split_hash:
        raise ValueError("Checkpoint split fingerprint does not match the supplied manifest")
    if provenance.get("exploratory_subset") is not False:
        raise ValueError("Release audit requires a full-cohort checkpoint")
    cfg = load_config(config)
    # Training fingerprints deliberately exclude epochs so resume permits more epochs.
    comparison = json.loads(json.dumps(cfg))
    del comparison["training"]["epochs"]
    if _hash(comparison) != provenance.get("config_hash"):
        raise ValueError("Baseline configuration fingerprint does not match checkpoint")
    if (int(state["model_settings"]["in_channels"]) != int(cfg["dataset"]["modalities"]) or
            int(state["model_settings"]["base_channels"]) != int(cfg["model"]["base_channels"])):
        raise ValueError("Checkpoint architecture and configuration differ")
    threshold = float(state["threshold"])
    epoch = int(state["epoch"])
    if not 0 < threshold < 1:
        raise ValueError("Checkpoint threshold is invalid")
    summaries = {}
    for group, path in (("validation", validation_report), ("test", test_report)):
        report = json.loads(path.read_text(encoding="utf-8"))
        summaries[group] = validate_evaluation_report(
            report, partition=group, expected_ids=set(split_data[group]),
            split_hash=split_hash, checkpoint_epoch=epoch, threshold=threshold,
        )
        validate_patient_csv(validation_csv if group == "validation" else test_csv, report["cases"])
    audit = {
        "status": "passed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Artifact consistency only; no medical predictions were recomputed",
        "checkpoint_epoch": epoch,
        "decision_threshold": threshold,
        "patient_counts": counts,
        "split_hash": split_hash,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "metrics": summaries,
        "warning": "This verifies artifacts, not clinical safety or external generalization.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--splits", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--validation-report", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-report", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit_release(
        checkpoint=args.checkpoint, config=args.config,
        dataset_dir=args.dataset_dir, split_manifest=args.splits,
        validation_report=args.validation_report,
        test_report=args.test_report, validation_csv=args.validation_csv,
        test_csv=args.test_csv, output=args.output,
    )
    print(json.dumps(result, indent=2))
    print(f"\nWrote release audit: {args.output}")


if __name__ == "__main__":
    main()
