"""Preprocess and cache labeled NIfTI studies into memory-mappable 2D slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from brain_mri.storage_paths import DATASET_DIR, SPLIT_MANIFEST, CACHE_DIR, require_ssd_if_targeted
from brain_mri.cache import PreprocessSettings, prepare_case
from brain_mri.data import read_catalog
from brain_mri.splits import validate_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--splits", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--include-test", action="store_true",
                        help="Prepare locked test patients for final evaluation only")
    parser.add_argument("--max-train-patients", type=int,
                        help="EXPLORATORY ONLY: cache first N fixed training patient IDs")
    parser.add_argument("--max-validation-patients", type=int,
                        help="EXPLORATORY ONLY: cache first N fixed validation patient IDs")
    args = parser.parse_args()
    for path in (args.dataset_dir, args.splits, args.cache_dir):
        require_ssd_if_targeted(path)
    for value in (args.max_train_patients, args.max_validation_patients):
        if value is not None and value < 1:
            parser.error("Patient subset size must be positive")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    p = cfg["preprocessing"]
    settings = PreprocessSettings(
        clip_low_percentile=float(p["clip_low_percentile"]),
        clip_high_percentile=float(p["clip_high_percentile"]),
        modalities=int(cfg["dataset"]["modalities"]),
    )
    catalog = read_catalog(args.dataset_dir)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    validate_splits(splits, {record.case_id for record in catalog.cases})
    train_ids = splits["train"][:args.max_train_patients]
    val_ids = splits["validation"][:args.max_validation_patients]
    ids = train_ids + val_ids + (splits["test"] if args.include_test else [])
    lookup = {record.case_id: record for record in catalog.cases}
    cases = [lookup[case_id] for case_id in ids]
    for i, case in enumerate(cases, start=1):
        meta = prepare_case(case, args.cache_dir, settings)
        print(f"[{i}/{len(cases)}] {case.case_id}: {meta['image_shape']} "
              f"({len(meta['foreground_slices'])} tumor-containing slices)", flush=True)
    print(f"Prepared {len(cases)} cases; test cache included={args.include_test}.")


if __name__ == "__main__":
    main()
