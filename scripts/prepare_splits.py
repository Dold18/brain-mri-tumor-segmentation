"""Create a once-only patient-level split manifest for the labeled MSD cohort."""

import argparse
from pathlib import Path

from brain_mri.storage_paths import DATASET_DIR, SPLIT_MANIFEST, require_ssd_if_targeted
from brain_mri.data import read_catalog
from brain_mri.splits import write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for path in (args.dataset_dir, args.output):
        require_ssd_if_targeted(path)
    catalog = read_catalog(args.dataset_dir, require_files=True)
    result = write_splits(catalog, args.output, seed=args.seed)
    print(f"Saved immutable patient-level split manifest: {result}")


if __name__ == "__main__":
    main()
