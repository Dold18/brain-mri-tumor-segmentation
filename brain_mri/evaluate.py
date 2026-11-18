"""Volume-level evaluation on locked patient splits using a real model checkpoint.

A test-set evaluation is an explicit final action: exploratory checkpoints are
refused, and the operator must pass --confirm-heldout. Neither training nor
checkpoint selection depends on the held-out test cases.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cache import PreprocessSettings, case_directory, read_case_metadata
from .data import read_catalog
from .inference import load_checkpoint_model, predict_cached_case
from .metrics import Confusion, summarize_cases
from .splits import validate_splits
from .train import _hash, select_device
from .storage_paths import DATASET_DIR, SPLIT_MANIFEST, CACHE_DIR, FINAL_TEST_RUN_DIR, require_ssd_if_targeted
from .visualization import save_case_comparison


METRIC_COLUMNS = (
    "case_id", "dice", "iou", "precision", "recall",
    "false_positive_voxels", "true_positive_voxels", "false_negative_voxels",
)


def run_evaluation(
    *,
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    split_manifest: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    partition: str = "validation",
    confirm_heldout: bool = False,
    device: str = "auto",
    batch_size: int = 8,
    examples: int = 0,
    max_patients: int | None = None,
) -> dict:
    for path in (dataset_dir, split_manifest, cache_dir, output_dir):
        require_ssd_if_targeted(path)
    if partition not in {"validation", "test"}:
        raise ValueError("Evaluation partition must be validation or test (never train)")
    if batch_size < 1 or examples < 0:
        raise ValueError("batch_size must be positive and examples nonnegative")
    if max_patients is not None and max_patients < 1:
        raise ValueError("max_patients must be positive")
    if partition == "test" and max_patients is not None:
        raise ValueError("The held-out test must be evaluated in full")
    if partition == "test" and not confirm_heldout:
        raise PermissionError("Test set is locked; explicitly pass --confirm-heldout for final evaluation")
    target_device = select_device(device)
    catalog = read_catalog(dataset_dir)
    splits = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    validate_splits(splits, {record.case_id for record in catalog.cases})

    model, checkpoint = load_checkpoint_model(checkpoint_path, device=str(target_device))
    provenance = checkpoint.get("provenance", {})
    if provenance.get("split_hash") != _hash(splits):
        raise ValueError("Checkpoint/split-manifest mismatch: evaluation refused")
    if partition == "test" and provenance.get("exploratory_subset") is not False:
        raise ValueError("Exploratory checkpoint cannot be reported as final held-out evaluation")

    preprocess = checkpoint["preprocessing"]
    settings = PreprocessSettings(
        clip_low_percentile=float(preprocess["clip_low_percentile"]),
        clip_high_percentile=float(preprocess["clip_high_percentile"]),
        modalities=int(checkpoint["model_settings"]["in_channels"]),
    )
    threshold = float(checkpoint["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Checkpoint decision threshold is invalid")
    lookup = {record.case_id: record for record in catalog.cases}

    # Validate cache existence/provenance before opening an output directory.
    ids = splits[partition][:max_patients]
    for case_id in ids:
        read_case_metadata(cache_dir, lookup[case_id], settings)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation folder {output} must be new/empty")
    output.mkdir(parents=True, exist_ok=True)

    case_metrics = {}
    exported_examples = []
    for i, case_id in enumerate(ids):
        folder = case_directory(cache_dir, case_id)
        images = np.load(folder / "image.npy", mmap_mode="r")
        masks = np.load(folder / "mask.npy", mmap_mode="r")
        if masks.shape != (len(images), 1, images.shape[-2], images.shape[-1]):
            raise ValueError(f"Corrupt cached arrays for {case_id}")
        probabilities = predict_cached_case(
            model, images, batch_size=batch_size, device=target_device
        )
        predicted = probabilities >= threshold
        truth = np.asarray(masks[:, 0], dtype=bool)
        confusion = Confusion()
        confusion.add(predicted, truth)
        case_metrics[case_id] = confusion

        print(f"[{i + 1}/{len(ids)}] {case_id} Dice={confusion.result()['dice']:.4f}", flush=True)

    summary = summarize_cases(case_metrics)
    # Deterministic failure/typical/success examples selected by measured Dice,
    # avoiding manual cherry-picking. Render only after all patient scores exist.
    if examples:
        ranked = sorted(ids, key=lambda cid: (summary["cases"][cid]["dice"], cid))
        sample_count = min(examples, len(ranked))
        if sample_count == 1:
            positions = [len(ranked) // 2]
        else:
            positions = np.linspace(0, len(ranked) - 1, sample_count).round().astype(int)
        for j, position in enumerate(positions, start=1):
            case_id = ranked[int(position)]
            folder = case_directory(cache_dir, case_id)
            images = np.load(folder / "image.npy", mmap_mode="r")
            truth = np.asarray(np.load(folder / "mask.npy", mmap_mode="r")[:, 0], dtype=bool)
            predictions = predict_cached_case(model, images, batch_size=batch_size,
                                              device=target_device) >= threshold
            slice_index = int(np.argmax(truth.sum(axis=(1, 2))))
            dest = output / f"example_{j:02d}_{case_id}.png"
            save_case_comparison(dest, np.asarray(images[slice_index, 0], dtype=np.float32),
                                 truth[slice_index], predictions[slice_index],
                                 case_id=case_id, slice_index=slice_index)
            exported_examples.append(str(dest))
    report = {
        "evaluation_partition": partition,
        "patients_evaluated": len(ids),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "exploratory_checkpoint": bool(provenance.get("exploratory_subset", True)),
        "exploratory_evaluation": bool(max_patients is not None or provenance.get("exploratory_subset", True)),
        "decision_threshold": threshold,
        "split_hash": provenance["split_hash"],
        "mean_dice": summary["mean_dice"],
        "mean_iou": summary["mean_iou"],
        "mean_recall": summary["mean_recall"],
        "mean_precision": summary["mean_precision"],
        "cases": summary["cases"],
        "example_images": exported_examples,
        "limitations": (
            "Research-only glioma-region segmentation; binary mask includes edema. "
            "No healthy-vs-tumor diagnostic validation."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "per_patient.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for case_id in sorted(summary["cases"]):
            writer.writerow({"case_id": case_id, **summary["cases"][case_id]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--splits", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=FINAL_TEST_RUN_DIR)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--confirm-heldout", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--max-patients", type=int,
                        help="EXPLORATORY validation only: evaluate the first N patient IDs")
    args = parser.parse_args()
    result = run_evaluation(
        checkpoint_path=args.checkpoint, dataset_dir=args.dataset_dir,
        split_manifest=args.splits, cache_dir=args.cache_dir,
        output_dir=args.output_dir, partition=args.partition,
        confirm_heldout=args.confirm_heldout, device=args.device,
        batch_size=args.batch_size, examples=args.examples, max_patients=args.max_patients,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
