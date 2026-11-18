"""On-disk, memory-mappable patient cache for repeatable 2D MRI training.

Each case is prepared *once* from a canonical NIfTI volume. The two .npy
files contain channels-first axial slices and are never committed to Git.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .data import CaseRecord, load_case
from .preprocessing import normalize_mri, whole_tumor_mask

CACHE_VERSION = 1


@dataclass(frozen=True)
class PreprocessSettings:
    clip_low_percentile: float = 0.5
    clip_high_percentile: float = 99.5
    modalities: int = 4
    allowed_labels: tuple[int, ...] = (0, 1, 2, 3)

    def __post_init__(self) -> None:
        if not 0 <= self.clip_low_percentile < self.clip_high_percentile <= 100:
            raise ValueError("Clipping percentiles must satisfy 0 <= low < high <= 100")
        if self.modalities < 1 or 0 not in self.allowed_labels:
            raise ValueError("Invalid modalities/allowed_labels preprocessing settings")


def _identity(record: CaseRecord, settings: PreprocessSettings) -> str:
    sources = []
    for path in (record.image_path, record.label_path):
        stats = path.stat()
        sources.append({"path": str(path.resolve()), "size": stats.st_size,
                        "mtime_ns": stats.st_mtime_ns})
    material = {"version": CACHE_VERSION, "settings": asdict(settings), "sources": sources}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def case_directory(cache_root: str | Path, case_id: str) -> Path:
    """Keep patient IDs from creating arbitrary paths."""
    if not case_id or case_id in {".", ".."} or any(c in case_id for c in ("/", "\\")):
        raise ValueError(f"Unsafe case identifier: {case_id!r}")
    return Path(cache_root) / case_id


def read_case_metadata(cache_root: str | Path, record: CaseRecord,
                       settings: PreprocessSettings) -> dict:
    directory = case_directory(cache_root, record.case_id)
    path = directory / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache for {record.case_id}: run prepare_cache first")
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("identity") != _identity(record, settings):
        raise ValueError(f"Stale/incompatible cache for {record.case_id}; remove it and regenerate")
    for filename in ("image.npy", "mask.npy"):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Cache missing {directory / filename}")
    return metadata


def prepare_case(record: CaseRecord, cache_root: str | Path,
                 settings: PreprocessSettings) -> dict:
    """Create one case atomically. Existing entries must match source/config."""
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = case_directory(root, record.case_id)
    if destination.exists():
        return read_case_metadata(root, record, settings)

    expected_identity = _identity(record, settings)
    image, raw_labels, affine = load_case(
        record, expected_channels=settings.modalities, valid_labels=set(settings.allowed_labels)
    )
    normalized = normalize_mri(
        image,
        lower_percentile=settings.clip_low_percentile,
        upper_percentile=settings.clip_high_percentile,
    )
    binary = whole_tumor_mask(raw_labels)
    # [X,Y,Z,C] -> [Z,C,X,Y], and [X,Y,Z] -> [Z,1,X,Y]
    channels_first = np.moveaxis(normalized, (2, 3), (0, 1))
    masks = np.moveaxis(binary, 2, 0)[:, None, :, :]
    foreground_slices = np.flatnonzero(np.any(masks != 0, axis=(1, 2, 3))).tolist()
    metadata = {
        "case_id": record.case_id,
        "identity": expected_identity,
        "version": CACHE_VERSION,
        "image_shape": list(channels_first.shape),
        "mask_shape": list(masks.shape),
        "foreground_slices": [int(v) for v in foreground_slices],
        "source_affine": affine.tolist(),
        "source_case": record.image_path.name,
    }
    with tempfile.TemporaryDirectory(prefix=".brain_mri-", dir=root) as tmp:
        folder = Path(tmp) / "entry"
        folder.mkdir()
        np.save(folder / "image.npy", channels_first.astype(np.float16))
        np.save(folder / "mask.npy", masks.astype(np.uint8))
        (folder / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # Concurrent preparations fail rather than corrupting an existing case.
        if destination.exists():
            return read_case_metadata(root, record, settings)
        folder.rename(destination)
    return metadata
