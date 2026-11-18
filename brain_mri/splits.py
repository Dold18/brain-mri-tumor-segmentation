"""Persistent patient-level train/validation/test partitioning."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .data import DatasetCatalog


def build_patient_splits(
    case_ids: list[str] | tuple[str, ...],
    *,
    seed: int = 2026,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, object]:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")
    ids = [str(i) for i in case_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient identifiers: cannot create leakage-safe splits")
    if len(ids) < 3:
        raise ValueError("At least three patients are required")
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = round(n * test_fraction)
    n_val = round(n * validation_fraction)
    if n_test == 0 or n_val == 0 or n_test + n_val >= n:
        raise ValueError("The dataset is too small for nonempty splits")
    result: dict[str, object] = {
        "seed": seed,
        "partition_unit": "patient_case_id",
        "total_cases": n,
        "train": sorted(ids[: n - n_val - n_test]),
        "validation": sorted(ids[n - n_val - n_test : n - n_test]),
        "test": sorted(ids[n - n_test :]),
    }
    validate_splits(result, set(case_ids))
    return result


def validate_splits(splits: dict[str, object], available_case_ids: set[str]) -> None:
    sets: list[set[str]] = []
    for group in ("train", "validation", "test"):
        values = splits.get(group)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Missing or empty split: {group}")
        if len(values) != len(set(values)):
            raise ValueError(f"Repeated patient within {group}")
        sets.append(set(values))
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Data leakage: a patient appears in multiple partitions")
    if set.union(*sets) != available_case_ids:
        raise ValueError("Split manifest does not cover exactly the available patients")


def write_splits(catalog: DatasetCatalog, output: str | Path, *, seed: int = 2026) -> Path:
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"Will not overwrite immutable split manifest: {path}")
    result = build_patient_splits([record.case_id for record in catalog.cases], seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally regenerating a held-out split.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return path
