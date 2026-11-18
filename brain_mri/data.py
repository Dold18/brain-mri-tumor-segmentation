"""MSD Task01 catalog parsing and geometry-safe NIfTI loading.

Native spatial axes are X, Y, Z; modality/channel is last. We canonicalize
both image and label to RAS orientation without silently resampling either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class DatasetCatalog:
    root: Path
    cases: tuple[CaseRecord, ...]
    modalities: dict[str, str]
    labels: dict[str, str]


def _safe_file(root: Path, value: str) -> Path:
    """Reject absolute paths and paths that escape the supplied dataset root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid file path in dataset.json: {value!r}")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"Absolute dataset path is not allowed: {value}")
    full = (root / relative).resolve()
    if not full.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset file escapes root: {value}")
    if not full.name.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz, got {value}")
    return full


def read_catalog(dataset_dir: str | Path, *, require_files: bool = True) -> DatasetCatalog:
    """Load MSD dataset.json; records refer to labeled training cases only."""
    root = Path(dataset_dir).expanduser().resolve()
    manifest_path = root / "dataset.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot find {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    entries = metadata.get("training")
    if not isinstance(entries, list) or not entries:
        raise ValueError("dataset.json must include a nonempty 'training' list")

    records: list[CaseRecord] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not {"image", "label"} <= entry.keys():
            raise ValueError(f"Invalid training entry #{i}: expected image and label")
        image_path = _safe_file(root, entry["image"])
        label_path = _safe_file(root, entry["label"])
        if image_path.stem.removesuffix(".nii") != label_path.stem.removesuffix(".nii"):
            raise ValueError(f"Image/mask case identifier mismatch in entry #{i}")
        case_id = image_path.name.removesuffix(".nii.gz")
        if case_id in seen:
            raise ValueError(f"Duplicate patient case identifier: {case_id}")
        seen.add(case_id)
        if require_files:
            for path in (image_path, label_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Case {case_id}: missing {path}")
        records.append(CaseRecord(case_id, image_path, label_path))

    n = metadata.get("numTraining")
    if n is not None and int(n) != len(records):
        raise ValueError(f"numTraining={n}, but training list has {len(records)} entries")
    modalities = metadata.get("modality", metadata.get("channel_names", {}))
    labels = metadata.get("labels", {})
    if not isinstance(modalities, dict) or not isinstance(labels, dict):
        raise ValueError("dataset.json modality/labels metadata must be objects")
    return DatasetCatalog(
        root=root,
        cases=tuple(records),
        modalities={str(k): str(v) for k, v in modalities.items()},
        labels={str(k): str(v) for k, v in labels.items()},
    )


def load_case(
    record: CaseRecord,
    *,
    expected_channels: int = 4,
    valid_labels: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(image[X,Y,Z,C], raw_mask[X,Y,Z], affine[4,4])` in RAS orientation.

    Affine equality is checked *after* orientation conversion. Mismatched grids
    fail rather than applying implicit spatial transformations to the labels.
    """
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError("NiBabel is required: python -m pip install nibabel") from error

    image_nii = nib.as_closest_canonical(nib.load(str(record.image_path)))
    label_nii = nib.as_closest_canonical(nib.load(str(record.label_path)))
    if len(image_nii.shape) != 4 or image_nii.shape[-1] != expected_channels:
        raise ValueError(
            f"{record.case_id}: expected 4D MRI with {expected_channels} channels, "
            f"found {image_nii.shape}"
        )
    if len(label_nii.shape) != 3 or image_nii.shape[:3] != label_nii.shape:
        raise ValueError(
            f"{record.case_id}: incompatible image {image_nii.shape} "
            f"and segmentation {label_nii.shape}"
        )
    if not np.allclose(image_nii.affine, label_nii.affine, atol=1e-3, rtol=1e-5):
        raise ValueError(f"{record.case_id}: image/mask affines differ; manual registration required")

    image = np.asarray(image_nii.get_fdata(dtype=np.float32), dtype=np.float32)
    label_float = np.asarray(label_nii.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(label_float)):
        raise ValueError(f"{record.case_id}: nonfinite voxels found")
    if not np.all(np.abs(label_float - np.rint(label_float)) < 1e-4):
        raise ValueError(f"{record.case_id}: segmentation contains fractional labels")
    mask = np.rint(label_float).astype(np.int16)
    observed = set(int(v) for v in np.unique(mask))
    if min(observed) < 0:
        raise ValueError(f"{record.case_id}: negative segmentation labels: {observed}")
    if valid_labels is not None and not observed <= valid_labels:
        raise ValueError(f"{record.case_id}: unexpected labels: {observed - valid_labels}")
    return image, mask, np.asarray(image_nii.affine, dtype=np.float64)
