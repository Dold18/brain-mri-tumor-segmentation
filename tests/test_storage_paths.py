"""External SSD defaults must never silently fall back to internal storage."""
from pathlib import Path

import pytest

from brain_mri import storage_paths as sp
from scripts.fetch_msd import fetch_dataset


def test_default_paths_all_target_storage():
    for path in (sp.DATA_ROOT, sp.DATASET_DIR, sp.SPLIT_MANIFEST,
                 sp.CACHE_DIR, sp.QUICKSTART_RUN_DIR, sp.BASELINE_RUN_DIR,
                 sp.FINAL_TEST_RUN_DIR):
        assert str(path).startswith('/Volumes/Storage/')


def test_disconnected_external_drive_refuses_writes(monkeypatch, tmp_path):
    mount = tmp_path / 'Storage'
    monkeypatch.setattr(sp, 'SSD_MOUNT', mount)
    with pytest.raises(RuntimeError, match='not mounted'):
        sp.require_ssd_if_targeted(mount / 'data')
    with pytest.raises(RuntimeError, match='not mounted'):
        fetch_dataset(mount / 'data')
    assert not (mount / 'data').exists()


def test_explicit_non_ssd_path_is_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, 'SSD_MOUNT', tmp_path / 'Storage')
    sp.require_ssd_if_targeted(tmp_path / 'local-data')


def test_connected_external_drive_is_allowed(monkeypatch, tmp_path):
    mount = tmp_path / 'Storage'
    mount.mkdir()
    monkeypatch.setattr(sp, 'SSD_MOUNT', mount)
    old_is_mount = Path.is_mount
    monkeypatch.setattr(Path, 'is_mount', lambda self: True if self == mount else old_is_mount(self))
    sp.require_ssd_if_targeted(mount / 'data')
