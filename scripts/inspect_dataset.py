"""Read the MSD catalog and optionally validate representative real NIfTI cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brain_mri.storage_paths import DATASET_DIR, require_ssd_if_targeted
from brain_mri.data import load_case, read_catalog


def inspect(dataset_dir: str | Path, *, limit: int = 3, metadata_only: bool = False) -> dict:
    catalog = read_catalog(dataset_dir, require_files=not metadata_only)
    report: dict = {
        "dataset_root": str(catalog.root),
        "labeled_case_count": len(catalog.cases),
        "modality_metadata": catalog.modalities,
        "label_metadata": catalog.labels,
        "inspected_count": 0,
        "cases": [],
    }
    if metadata_only:
        return report
    if limit < 0:
        raise ValueError("--limit must be >= 0 (zero means all)")
    sample = catalog.cases if limit == 0 else catalog.cases[:limit]
    valid_labels = None
    if catalog.labels:
        try:
            valid_labels = {int(v) for v in catalog.labels.keys()}
        except ValueError:
            # Some versions provide label->number instead of number->label.
            try:
                valid_labels = {int(v) for v in catalog.labels.values()}
            except ValueError:
                valid_labels = None
    for case in sample:
        image, label, affine = load_case(case, valid_labels=valid_labels)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        report["cases"].append(
            {
                "case_id": case.case_id,
                "image_shape_XYZC": list(image.shape),
                "label_shape_XYZ": list(label.shape),
                "observed_labels": [int(x) for x in np.unique(label)],
                "voxel_spacing_mm": [round(float(x), 6) for x in spacing],
                "foreground_voxels": int(np.count_nonzero(label)),
            }
        )
    report["inspected_count"] = len(sample)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--limit", type=int, default=3, help="Cases to inspect; 0 inspects all")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON output")
    args = parser.parse_args()
    require_ssd_if_targeted(args.dataset_dir)
    report = inspect(args.dataset_dir, limit=args.limit, metadata_only=args.metadata_only)
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
