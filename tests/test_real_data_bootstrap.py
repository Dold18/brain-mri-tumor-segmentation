"""Archive safety and orchestration tests that do not need patient MRI data."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest

from brain_mri.data import CaseRecord, DatasetCatalog, read_catalog
from scripts.fetch_msd import _archive_member_path, extract_archive, md5sum
from scripts import real_data_quickstart as quickstart


def _tar_member(tar: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _mini_tar(path: Path, malicious_name: str | None = None):
    base = "Task01_BrainTumour"
    entries = [
        {"image": f"./imagesTr/BRATS_{i:03d}.nii.gz",
         "label": f"./labelsTr/BRATS_{i:03d}.nii.gz"}
        for i in range(1, 4)
    ]
    metadata = {"numTraining": 3, "modality": {"0": "FLAIR"},
                "labels": {"0": "background", "1": "edema", "2": "core", "3": "tumor"},
                "training": entries}
    with tarfile.open(path, mode="w") as tar:
        _tar_member(tar, f"{base}/dataset.json", json.dumps(metadata).encode())
        for i in range(1, 4):
            for folder in ("imagesTr", "labelsTr"):
                _tar_member(tar, f"{base}/{folder}/BRATS_{i:03d}.nii.gz", b"fake nifti")
        if malicious_name:
            _tar_member(tar, malicious_name, b"invalid")
    return path


def test_safe_extract_accepts_dataset_catalog(tmp_path):
    archive = _mini_tar(tmp_path / "example.tar")
    root = tmp_path / "data"
    result = extract_archive(archive, root)
    assert len(read_catalog(result).cases) == 3
    assert result == extract_archive(archive, root)  # idempotent
    assert len(md5sum(archive)) == 32


@pytest.mark.parametrize("bad_path", [
    "Task01_BrainTumour/../../private.txt",
    "/private/absolute.txt",
    "AnotherTask/imagesTr/BRATS_999.nii.gz",
    "Task01_BrainTumour\\bad.txt",
])
def test_safe_extract_rejects_invalid_paths_without_committing(tmp_path, bad_path):
    archive = _mini_tar(tmp_path / "example.tar", malicious_name=bad_path)
    output = tmp_path / "data"
    with pytest.raises(ValueError):
        extract_archive(archive, output)
    assert not (output / "Task01_BrainTumour").exists()


def test_safe_extract_refuses_symlinks():
    entry = tarfile.TarInfo("Task01_BrainTumour/imagesTr/link.nii.gz")
    entry.type = tarfile.SYMTYPE
    entry.linkname = "../../secret"
    with pytest.raises(ValueError, match="special"):
        _archive_member_path(entry)


def test_real_quickstart_reuses_same_split_and_orchestrates(monkeypatch, tmp_path):
    records = []
    for i in range(12):
        name = f"BRATS_{i:03d}"
        records.append(CaseRecord(name, tmp_path / f"{name}.nii.gz",
                                  tmp_path / f"{name}-mask.nii.gz"))
    catalog = DatasetCatalog(tmp_path, tuple(records), {"0": "FLAIR"}, {"0": "background"})
    monkeypatch.setattr(quickstart, "read_catalog", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(quickstart, "inspect", lambda *args, **kwargs: {"cases": [{"case_id": "BRATS_000"}]})
    visited = []
    def fake_prepare(record, cache_dir, settings):
        visited.append(record.case_id)
        return {"foreground_slices": [0, 1]}
    monkeypatch.setattr(quickstart, "prepare_case", fake_prepare)
    monkeypatch.setattr(quickstart, "run_training", lambda **kwargs: {"best_checkpoint":
                                                                     str(tmp_path / "run/best.pt")})
    monkeypatch.setattr(quickstart, "run_evaluation", lambda **kwargs: {"mean_dice": 0.15})
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("preprocessing:\n  clip_low_percentile: 0.5\n  clip_high_percentile: 99.5\ndataset:\n  modalities: 4\n")
    splits = tmp_path / "splits.json"
    kwargs = dict(dataset_dir=tmp_path, split_manifest=splits, cache_dir=tmp_path / "cache",
                  run_dir=tmp_path / "run", config_path=cfg, train_patients=3,
                  validation_patients=1, device="cpu")
    first = quickstart.run(**kwargs)
    assert first["exploratory_only"] is True
    assert first["mean_validation_dice"] == 0.15
    assert len(visited) == 4
    original_manifest = splits.read_bytes()
    quickstart.run(**kwargs)
    assert splits.read_bytes() == original_manifest


def test_real_quickstart_rejects_nonsubsets(monkeypatch, tmp_path):
    records = tuple(CaseRecord(f"BRATS_{i:03d}", tmp_path / "i", tmp_path / "l")
                    for i in range(6))
    monkeypatch.setattr(quickstart, "read_catalog", lambda *args, **kwargs:
                        DatasetCatalog(tmp_path, records, {}, {}))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("{}")
    with pytest.raises(ValueError, match="strict subsets"):
        quickstart.run(dataset_dir=tmp_path, split_manifest=tmp_path / "splits.json",
                       cache_dir=tmp_path / "cache", run_dir=tmp_path / "run",
                       config_path=cfg, train_patients=100, validation_patients=1,
                       device="cpu")

class _Response(io.BytesIO):
    def __init__(self, body: bytes, *, status: int, headers: dict):
        super().__init__(body)
        self.status = status
        self.headers = headers


def test_resumable_dataset_download(monkeypatch, tmp_path):
    from scripts import fetch_msd
    target = tmp_path / "downloads" / "Task01_BrainTumour.tar"
    target.parent.mkdir()
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(b"abc")

    def open_response(request, timeout):
        assert request.get_header("Range") == "bytes=3-"
        return _Response(b"defg", status=206,
                         headers={"Content-Length": "4", "Content-Range": "bytes 3-6/7"})

    monkeypatch.setattr(fetch_msd.urllib.request, "urlopen", open_response)
    assert fetch_msd.download_archive("https://example.org/medical.tar", target) == target
    assert target.read_bytes() == b"abcdefg"
    assert not partial.exists()


def test_range_ignored_does_not_corrupt_partial(monkeypatch, tmp_path):
    from scripts import fetch_msd
    target = tmp_path / "medical.tar"
    partial = tmp_path / "medical.tar.part"
    partial.write_bytes(b"abc")
    monkeypatch.setattr(fetch_msd.urllib.request, "urlopen",
                        lambda request, timeout: _Response(b"abcdef", status=200,
                                                          headers={"Content-Length": "6"}))
    with pytest.raises(RuntimeError, match="ignored HTTP Range"):
        fetch_msd.download_archive("https://example.org/test.tar", target)
    assert partial.read_bytes() == b"abc"
    assert not target.exists()
