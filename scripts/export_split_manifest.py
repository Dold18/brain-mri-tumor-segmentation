"""Export a hash-verified copy of the published patient split (never regenerate)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from brain_mri.storage_paths import SPLIT_MANIFEST

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def export_manifest(source: Path, destination: Path, audit_file: Path) -> Path:
    """Check audited hashes and only export public case IDs to a new file."""
    source = Path(source)
    destination = Path(destination)
    audit_file = Path(audit_file)
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    raw = source.read_bytes()
    expected = audit["sha256"]["split_manifest"]
    if _sha256(raw) != expected:
        raise ValueError("Split manifest SHA-256 mismatch: refusing export")
    manifest = json.loads(raw)
    allowed = {"seed", "partition_unit", "total_cases", "train", "validation", "test"}
    if set(manifest) != allowed:
        raise ValueError("Unexpected split-manifest fields: refusing possible data disclosure")
    if manifest["partition_unit"] != "patient_case_id":
        raise ValueError("Unexpected split unit")
    seen: set[str] = set()
    for partition in ("train", "validation", "test"):
        ids = manifest[partition]
        if not isinstance(ids, list) or not all(
            isinstance(case_id, str) and re.fullmatch(r"BRATS_\d{3}", case_id)
            for case_id in ids
        ) or len(set(ids)) != len(ids):
            raise ValueError(f"Unsafe/invalid case identifiers for {partition}")
        if seen.intersection(ids):
            raise ValueError("Patient leakage between split partitions")
        seen.update(ids)
        if len(ids) != audit["patient_counts"][partition]:
            raise ValueError(f"Unexpected {partition} count")
    if len(seen) != manifest["total_cases"]:
        raise ValueError("Total case count mismatch")
    semantic_hash = _sha256(json.dumps(manifest, sort_keys=True).encode("utf-8"))
    if semantic_hash != audit["split_hash"]:
        raise ValueError("Split semantic fingerprint mismatch")
    if destination.exists():
        if _sha256(destination.read_bytes()) != expected:
            raise FileExistsError("Destination exists but differs from audited manifest")
        print(f"Audited manifest already exported: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(raw)
    print(f"Exported {len(seen)} patient identifiers to {destination}")
    print(f"SHA-256: {expected}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "manifests" / "patient_splits.json")
    parser.add_argument("--audit", type=Path,
                        default=REPO_ROOT / "results" / "release_audit.json")
    args = parser.parse_args()
    export_manifest(args.source, args.output, args.audit)


if __name__ == "__main__":
    main()
