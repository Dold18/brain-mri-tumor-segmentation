"""One-command real-MRI exploratory run: inspect, split, cache, train, evaluate.

The subset results are EXPERIMENTAL and never represent held-out test metrics.
Requires the official labeled MSD Task01_BrainTumour files already extracted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from brain_mri.cache import PreprocessSettings, prepare_case
from brain_mri.data import read_catalog
from brain_mri.evaluate import run_evaluation
from brain_mri.splits import validate_splits, write_splits
from brain_mri.train import run_training
from scripts.inspect_dataset import inspect
from brain_mri.storage_paths import (
    DATASET_DIR, SPLIT_MANIFEST, CACHE_DIR, QUICKSTART_RUN_DIR, require_ssd_if_targeted,
)


def run(*, dataset_dir: Path, split_manifest: Path, cache_dir: Path,
        run_dir: Path, config_path: Path, train_patients: int,
        validation_patients: int, device: str) -> dict:
    for path in (dataset_dir, split_manifest, cache_dir, run_dir):
        require_ssd_if_targeted(path)
    if train_patients < 1 or validation_patients < 1:
        raise ValueError("Subset case counts must be positive")
    catalog = read_catalog(dataset_dir, require_files=True)
    if not split_manifest.exists():
        split_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_splits(catalog, split_manifest, seed=2026)
        print(f"Created immutable patient split manifest: {split_manifest}", flush=True)
    else:
        print(f"Reusing patient split manifest: {split_manifest}", flush=True)
    splits = json.loads(split_manifest.read_text(encoding="utf-8"))
    validate_splits(splits, {r.case_id for r in catalog.cases})
    if train_patients >= len(splits["train"]) or validation_patients >= len(splits["validation"]):
        raise ValueError(
            "Quickstart must use strict subsets of both train and validation: "
            f"available {len(splits['train'])} train / {len(splits['validation'])} validation cases"
        )
    report = inspect(dataset_dir, limit=1)
    print("Validated first real NIfTI case:", json.dumps(report["cases"][0]), flush=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = PreprocessSettings(
        float(cfg["preprocessing"]["clip_low_percentile"]),
        float(cfg["preprocessing"]["clip_high_percentile"]),
        int(cfg["dataset"]["modalities"]),
    )
    ids = splits["train"][:train_patients] + splits["validation"][:validation_patients]
    lookup = {record.case_id: record for record in catalog.cases}
    for i, case_id in enumerate(ids, start=1):
        meta = prepare_case(lookup[case_id], cache_dir, settings)
        print(f"Cached patient {i}/{len(ids)}: {case_id}, "
              f"{len(meta['foreground_slices'])} annotated slices", flush=True)
    trained = run_training(
        config_path=config_path, dataset_dir=dataset_dir,
        split_manifest=split_manifest, cache_dir=cache_dir, output_dir=run_dir,
        device=device, max_train_patients=train_patients,
        max_validation_patients=validation_patients,
    )
    evaluation = run_evaluation(
        checkpoint_path=Path(trained["best_checkpoint"]), dataset_dir=dataset_dir,
        split_manifest=split_manifest, cache_dir=cache_dir,
        output_dir=run_dir.parent / f"{run_dir.name}_validation",
        partition="validation", device=device, max_patients=validation_patients,
        examples=min(2, validation_patients),
    )
    return {
        "status": "exploratory real MRI training completed",
        "checkpoint": trained["best_checkpoint"],
        "validation_report": str(run_dir.parent / f"{run_dir.name}_validation" / "metrics.json"),
        "patients_train": train_patients,
        "patients_validation": validation_patients,
        "exploratory_only": True,
        "mean_validation_dice": evaluation["mean_dice"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--splits", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=QUICKSTART_RUN_DIR)
    parser.add_argument("--config", type=Path, default=Path("configs/quickstart.yaml"))
    parser.add_argument("--train-patients", type=int, default=8)
    parser.add_argument("--validation-patients", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    summary = run(dataset_dir=args.dataset_dir, split_manifest=args.splits,
                  cache_dir=args.cache_dir, run_dir=args.output_dir, config_path=args.config,
                  train_patients=args.train_patients,
                  validation_patients=args.validation_patients, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
