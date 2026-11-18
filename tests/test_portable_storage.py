"""Portable root override; checks use isolated Python processes."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _paths_with(root: str | None) -> list[str]:
    env = dict(os.environ)
    if root is None:
        env.pop("BRAIN_MRI_STORAGE_ROOT", None)
    else:
        env["BRAIN_MRI_STORAGE_ROOT"] = root
    script = (
        "from brain_mri.storage_paths import STORAGE_ROOT, DATASET_DIR, CACHE_DIR; "
        "print(STORAGE_ROOT); print(DATASET_DIR); print(CACHE_DIR)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True, env=env, cwd=Path(__file__).resolve().parents[1],
    )
    return completed.stdout.splitlines()


def test_unset_override_retains_original_ssd_paths():
    paths = _paths_with(None)
    assert paths[0] == "/Volumes/Storage/brain-mri-tumor-segmentation"
    assert paths[1].endswith("/data/Task01_BrainTumour")


def test_custom_root_changes_every_default(tmp_path):
    paths = _paths_with(str(tmp_path))
    assert paths[0] == str(tmp_path)
    assert paths[1] == str(tmp_path / "data" / "Task01_BrainTumour")
    assert paths[2] == str(tmp_path / "cache" / "msd")
