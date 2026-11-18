import hashlib
import json

import pytest

from scripts.export_split_manifest import export_manifest


def fixture_files(tmp_path):
    doc = {
        "seed": 2026, "partition_unit": "patient_case_id", "total_cases": 3,
        "train": ["BRATS_001"], "validation": ["BRATS_002"], "test": ["BRATS_003"],
    }
    raw = (json.dumps(doc, indent=2) + "\n").encode()
    source = tmp_path / "source.json"
    source.write_bytes(raw)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "sha256": {"split_manifest": hashlib.sha256(raw).hexdigest()},
        "split_hash": hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest(),
        "patient_counts": {"train": 1, "validation": 1, "test": 1},
    }))
    return source, audit, tmp_path / "out" / "manifest.json"


def test_valid_export_and_idempotency(tmp_path):
    source, audit, target = fixture_files(tmp_path)
    assert export_manifest(source, target, audit) == target
    assert target.read_bytes() == source.read_bytes()
    assert export_manifest(source, target, audit) == target


def test_rejects_wrong_input_sha(tmp_path):
    source, audit, target = fixture_files(tmp_path)
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_manifest(source, target, audit)
    assert not target.exists()


def test_rejects_destination_conflict(tmp_path):
    source, audit, target = fixture_files(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text("other manifest")
    with pytest.raises(FileExistsError):
        export_manifest(source, target, audit)


def test_rejects_extra_metadata_even_if_hashes_match(tmp_path):
    source, audit, target = fixture_files(tmp_path)
    doc = json.loads(source.read_text())
    doc["patient_notes"] = "not for release"
    raw = json.dumps(doc).encode()
    source.write_bytes(raw)
    a = json.loads(audit.read_text())
    a["sha256"]["split_manifest"] = hashlib.sha256(raw).hexdigest()
    a["split_hash"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    audit.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="Unexpected split-manifest fields"):
        export_manifest(source, target, audit)
