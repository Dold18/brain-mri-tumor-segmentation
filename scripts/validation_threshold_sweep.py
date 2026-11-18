"""Compare segmentation thresholds on the COMPLETE validation cohort only.

This is model-selection analysis, not an independent performance estimate.
Never operates on the locked test partition. Loads the trained checkpoint
unchanged and runs inference only once per validation patient.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brain_mri.cache import PreprocessSettings, case_directory, read_case_metadata
from brain_mri.data import read_catalog
from brain_mri.inference import load_checkpoint_model, predict_cached_case
from brain_mri.metrics import Confusion, summarize_cases
from brain_mri.splits import validate_splits
from brain_mri.storage_paths import (
    BASELINE_RUN_DIR, CACHE_DIR, DATASET_DIR, SPLIT_MANIFEST,
    require_ssd_if_targeted,
)
from brain_mri.train import _hash, select_device


DEFAULT_THRESHOLDS = tuple(round(0.20 + 0.05 * i, 2) for i in range(13))


def score_case(probabilities: np.ndarray, truth: np.ndarray, thresholds: tuple[float, ...]) -> dict[float, Confusion]:
    """Return whole-volume confusion counts for each fixed threshold."""
    prob = np.asarray(probabilities)
    mask = np.asarray(truth, dtype=bool)
    if prob.shape != mask.shape or prob.ndim != 3 or not np.isfinite(prob).all():
        raise ValueError("Expected finite [Z,H,W] probabilities aligned with the binary mask")
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("Probabilities must fall within [0,1]")
    scores = {}
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError(f"Invalid probability threshold: {threshold}")
        c = Confusion()
        c.add(prob >= threshold, mask)
        scores[threshold] = c
    return scores


def pick_threshold(scores: dict[float, dict], baseline: float) -> float:
    """Prefer baseline/near-baseline thresholds when mean Dice is tied."""
    if not scores or not 0 < baseline < 1:
        raise ValueError("Missing threshold scores or invalid baseline")
    return max(scores, key=lambda t: (
        scores[t]["mean_dice"], -abs(t - baseline), -t,
    ))


def run_sweep(*, checkpoint: Path, dataset_dir: Path, splits_path: Path,
              cache_dir: Path, output_dir: Path, device: str = "auto",
              batch_size: int = 4, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> dict:
    for p in (checkpoint, dataset_dir, splits_path, cache_dir, output_dir):
        require_ssd_if_targeted(p)
    if batch_size < 1 or not thresholds:
        raise ValueError("Specify a positive batch size and at least one threshold")
    target_device = select_device(device)
    model, ckpt = load_checkpoint_model(checkpoint, device=str(target_device))
    provenance = ckpt.get("provenance", {})
    if provenance.get("exploratory_subset") is not False:
        raise ValueError("Threshold calibration requires a full-cohort checkpoint")
    baseline = float(ckpt["threshold"])
    thresholds = tuple(sorted(set(round(float(t), 5) for t in (*thresholds, baseline))))
    if any(not 0 < t < 1 for t in thresholds):
        raise ValueError("All thresholds must be strictly between 0 and 1")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    catalog = read_catalog(dataset_dir)
    lookup = {r.case_id: r for r in catalog.cases}
    validate_splits(splits, set(lookup))
    if provenance.get("split_hash") != _hash(splits):
        raise ValueError("Checkpoint and patient split manifest do not match")
    patient_ids = list(splits["validation"])  # always the complete validation cohort
    pp = ckpt["preprocessing"]
    settings = PreprocessSettings(
        float(pp["clip_low_percentile"]), float(pp["clip_high_percentile"]),
        int(ckpt["model_settings"]["in_channels"]),
    )
    for cid in patient_ids:
        read_case_metadata(cache_dir, lookup[cid], settings)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_threshold = {t: {} for t in thresholds}
    for i, cid in enumerate(patient_ids, 1):
        folder = case_directory(cache_dir, cid)
        images = np.load(folder / "image.npy", mmap_mode="r")
        masks = np.load(folder / "mask.npy", mmap_mode="r")
        if masks.shape != (len(images), 1, images.shape[-2], images.shape[-1]):
            raise ValueError(f"Invalid cache mask geometry for {cid}")
        probabilities = predict_cached_case(model, images, device=target_device, batch_size=batch_size)
        confusion_for_case = score_case(probabilities, np.asarray(masks[:, 0], dtype=bool), thresholds)
        for threshold, confusion in confusion_for_case.items():
            per_threshold[threshold][cid] = confusion
        print(f"[{i}/{len(patient_ids)}] {cid} baseline_Dice="
              f"{confusion_for_case[baseline].result()['dice']:.4f}", flush=True)

    summaries = {t: summarize_cases(cases) for t, cases in per_threshold.items()}
    optimal = pick_threshold(summaries, baseline)
    ref_cases = summaries[baseline]["cases"]
    best_cases = summaries[optimal]["cases"]
    deltas = {cid: best_cases[cid]["dice"] - ref_cases[cid]["dice"] for cid in patient_ids}
    report = {
        "cohort": "validation (FULL cohort only)",
        "n_patients": len(patient_ids),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "split_hash": provenance["split_hash"],
        "baseline_threshold": baseline,
        "chosen_validation_threshold": optimal,
        "baseline_mean_dice": summaries[baseline]["mean_dice"],
        "chosen_mean_dice": summaries[optimal]["mean_dice"],
        "mean_dice_delta": summaries[optimal]["mean_dice"] - summaries[baseline]["mean_dice"],
        "patient_changes": {
            "improved": sum(d > 1e-8 for d in deltas.values()),
            "worsened": sum(d < -1e-8 for d in deltas.values()),
            "unchanged": sum(abs(d) <= 1e-8 for d in deltas.values()),
        },
        "five_lowest_baseline_patients": [
            {"case_id": cid, "baseline_dice": ref_cases[cid]["dice"],
             "chosen_dice": best_cases[cid]["dice"], "delta": deltas[cid]}
            for cid in sorted(patient_ids, key=lambda x: (ref_cases[x]["dice"], x))[:5]
        ],
        "caution": "Validation is used for checkpoint selection AND threshold tuning. "
                   "This is not independent test-set performance. No held-out test cases accessed.",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["threshold", "mean_dice", "mean_iou", "mean_precision", "mean_recall"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for threshold in thresholds:
            writer.writerow({"threshold": threshold, **{key: summaries[threshold][key] for key in keys[1:]}})
    with (output_dir / "per_patient_thresholds.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["case_id", "threshold", "dice", "iou", "precision", "recall",
                "false_positive_voxels", "false_negative_voxels"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for cid in sorted(patient_ids):
            for threshold in thresholds:
                writer.writerow({"case_id": cid, "threshold": threshold, **{
                    key: summaries[threshold]["cases"][cid][key] for key in keys[2:]}})

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(list(thresholds), [summaries[t]["mean_dice"] for t in thresholds], marker="o")
    ax.axvline(baseline, linestyle="--", alpha=0.5, label=f"Baseline {baseline:.2f}")
    ax.axvline(optimal, linestyle=":", alpha=0.7, label=f"Chosen {optimal:.2f}")
    ax.set(xlabel="Probability threshold", ylabel="Mean patient Dice",
           title="Whole-validation-cohort threshold sensitivity")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_dice.png", dpi=180)
    plt.close(fig)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=BASELINE_RUN_DIR / "best.pt")
    p.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    p.add_argument("--splits", type=Path, default=SPLIT_MANIFEST)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=BASELINE_RUN_DIR / "threshold_review")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    a = p.parse_args()
    report = run_sweep(checkpoint=a.checkpoint, dataset_dir=a.dataset_dir,
                       splits_path=a.splits, cache_dir=a.cache_dir,
                       output_dir=a.output_dir, device=a.device, batch_size=a.batch_size,
                       thresholds=tuple(a.thresholds))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
