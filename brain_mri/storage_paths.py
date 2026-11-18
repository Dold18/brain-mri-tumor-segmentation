"""Default dataset, manifest, and run paths for the research workflow.

The development Mac uses /Volumes/Storage by default. For other computers,
set BRAIN_MRI_STORAGE_ROOT before starting Python to override the data root.
CLI options can still override individual paths.
"""
from __future__ import annotations

import os
from pathlib import Path

SSD_MOUNT = Path("/Volumes/Storage")
# Keep the validated development Mac paths unchanged unless explicitly overridden.
# Avoid a silent fallback to the internal disk if the SSD is disconnected.
_DEFAULT_STORAGE_ROOT = SSD_MOUNT / "brain-mri-tumor-segmentation"
_STORAGE_OVERRIDE = os.environ.get("BRAIN_MRI_STORAGE_ROOT", "").strip()
STORAGE_ROOT = (
    Path(_STORAGE_OVERRIDE).expanduser().absolute()
    if _STORAGE_OVERRIDE else _DEFAULT_STORAGE_ROOT
)
DATA_ROOT = STORAGE_ROOT / "data"
DATASET_DIR = DATA_ROOT / "Task01_BrainTumour"
SPLIT_MANIFEST = STORAGE_ROOT / "manifests" / "patient_splits.json"
CACHE_DIR = STORAGE_ROOT / "cache" / "msd"
QUICKSTART_RUN_DIR = STORAGE_ROOT / "runs" / "real_quickstart"
QUICKSTART_CHECKPOINT = QUICKSTART_RUN_DIR / "best.pt"
BASELINE_RUN_DIR = STORAGE_ROOT / "runs" / "baseline"
FINAL_TEST_RUN_DIR = STORAGE_ROOT / "runs" / "final_test"


def require_ssd_if_targeted(path: Path | str) -> None:
    """Fail before writing if a target on the configured SSD isn't mounted.

    Without this check, macOS might allow creation of an ordinary directory
    at /Volumes/Storage after an external disk has been disconnected.
    Custom destinations outside /Volumes/Storage remain supported.
    """
    target = Path(path).expanduser().absolute()
    if target == SSD_MOUNT or SSD_MOUNT in target.parents:
        if not SSD_MOUNT.is_mount():
            raise RuntimeError(
                "External SSD 'Storage' is not mounted at /Volumes/Storage. "
                "Connect it and verify with `ls /Volumes` or `diskutil info /Volumes/Storage`. "
                "Aborting before creating files; no fallback to internal storage."
            )
