import json

import pytest

from brain_mri.data import CaseRecord, DatasetCatalog
from brain_mri.splits import build_patient_splits, validate_splits, write_splits


def test_484_patient_partition_and_determinism():
    ids = [f"BRATS_{i:03d}" for i in range(1, 485)]
    a = build_patient_splits(ids, seed=2026)
    b = build_patient_splits(ids[::-1], seed=2026)
    assert a == b
    assert (len(a["train"]), len(a["validation"]), len(a["test"])) == (338, 73, 73)
    validate_splits(a, set(ids))


def test_leakage_and_duplicates_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        build_patient_splits(["one", "one", "two"])
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(
            {"train": ["a"], "validation": ["a"], "test": ["b"]}, {"a", "b"}
        )


def test_manifest_is_immutable(tmp_path):
    records = tuple(CaseRecord(k, tmp_path / k, tmp_path / k) for k in ("a", "b", "c", "d"))
    catalog = DatasetCatalog(tmp_path, records, {}, {})
    out = tmp_path / "manifest.json"
    write_splits(catalog, out)
    saved = json.loads(out.read_text())
    assert saved["partition_unit"] == "patient_case_id"
    with pytest.raises(FileExistsError):
        write_splits(catalog, out)
