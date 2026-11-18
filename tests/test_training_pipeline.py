"""End-to-end training and cache tests use synthetic MRI volumes, not clinical data."""

import csv
import json

import numpy as np
import pytest
import torch

from brain_mri.cache import PreprocessSettings, prepare_case, read_case_metadata
from brain_mri.data import read_catalog
from brain_mri.datasets import MRISliceDataset, balanced_sampler
from brain_mri.metrics import Confusion, summarize_cases
from brain_mri.inference import load_checkpoint_model, predict_volume
from scripts.plot_history import create_plots
from brain_mri.splits import write_splits
from brain_mri.train import run_training


def _fixture_cases(tmp_path, count=6):
    dataset = tmp_path / "dataset"
    (dataset / "imagesTr").mkdir(parents=True)
    (dataset / "labelsTr").mkdir()
    entries = []
    for i in range(count):
        filename = f"BRATS_{i:03d}.nii.gz"
        (dataset / "imagesTr" / filename).touch()
        (dataset / "labelsTr" / filename).touch()
        entries.append({"image": "./imagesTr/" + filename,
                        "label": "./labelsTr/" + filename})
    (dataset / "dataset.json").write_text(json.dumps({
        "numTraining": count,
        "modality": {"0": "FLAIR", "1": "T1", "2": "T1ce", "3": "T2"},
        "labels": {"0": "background", "1": "edema", "2": "nonenhancing", "3": "enhancing"},
        "training": entries,
    }))
    return read_catalog(dataset)


def _fake_nifti_volume(record, **_kwargs):
    i = int(record.case_id[-3:])
    rng = np.random.default_rng(i)
    x = np.zeros((32, 32, 5, 4), dtype=np.float32)
    x[3:29, 3:29] = rng.normal(100, 8, size=(26, 26, 5, 4))
    y = np.zeros((32, 32, 5), dtype=np.uint8)
    y[10:20, 12:23, 1 + i % 3] = 2
    return x, y, np.eye(4)


def test_cache_shapes_staleness_and_sampler(tmp_path, monkeypatch):
    monkeypatch.setattr("brain_mri.cache.load_case", _fake_nifti_volume)
    catalog = _fixture_cases(tmp_path)
    settings = PreprocessSettings()
    for record in catalog.cases[:2]:
        prepare_case(record, tmp_path / "cache", settings)
    meta = read_case_metadata(tmp_path / "cache", catalog.cases[0], settings)
    assert meta["image_shape"] == [5, 4, 32, 32]
    assert len(meta["foreground_slices"]) == 1
    ds = MRISliceDataset(list(catalog.cases[:2]), tmp_path / "cache", settings)
    assert len(ds) == 10
    sample = ds[1]
    assert sample["image"].shape == (4, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert sample["mask"].sum().item() > 0
    sampler = balanced_sampler(ds, samples_per_epoch=20, positive_fraction=0.5, seed=42)
    assert len(list(sampler)) == 20
    (catalog.cases[0].image_path).write_bytes(b"modified")
    with pytest.raises(ValueError, match="Stale"):
        read_case_metadata(tmp_path / "cache", catalog.cases[0], settings)


def test_patient_volume_metrics_are_not_slice_averages():
    a = Confusion()
    a.add(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    b = Confusion()
    b.add(np.array([0, 0]), np.array([0, 0]))
    metric = summarize_cases({"a": a, "b": b})
    assert metric["cases"]["a"]["dice"] == 0.5
    assert metric["cases"]["b"]["dice"] == 1.0
    assert metric["mean_dice"] == 0.75
    assert metric["cases"]["a"]["false_positive_voxels"] == 1


def test_training_checkpoint_resume_and_split_isolation(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr("brain_mri.cache.load_case", _fake_nifti_volume)
    catalog = _fixture_cases(tmp_path)
    splits_path = write_splits(catalog, tmp_path / "splits.json")
    splits = json.loads(splits_path.read_text())
    cache_root = tmp_path / "cache"
    # Prepare training + validation only: the test patients are never loaded.
    allowed = set(splits["train"] + splits["validation"])
    for record in catalog.cases:
        if record.case_id in allowed:
            prepare_case(record, cache_root, PreprocessSettings())
    assert not (cache_root / splits["test"][0]).exists()

    config = {
        "seed": 123,
        "dataset": {"modalities": 4},
        "preprocessing": {"clip_low_percentile": 0.5, "clip_high_percentile": 99.5},
        "model": {"name": "UNet2D", "base_channels": 4},
        "training": {
            "epochs": 1, "batch_size": 2, "samples_per_epoch": 4,
            "positive_fraction": 0.7, "validation_every": 1, "patience_checks": 3,
            "num_workers": 0, "learning_rate": 0.0001, "weight_decay": 0.0001,
            "threshold": 0.5, "optimizer": "AdamW", "loss": "DiceBCELoss",
        },
    }
    config_path = tmp_path / "train.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "run"
    common = dict(config_path=config_path, dataset_dir=catalog.root,
                  split_manifest=splits_path, cache_dir=cache_root,
                  output_dir=out, device="cpu")
    result = run_training(**common)
    assert result["epochs_completed"] == 1
    assert result["exploratory_subset"] is False
    assert (out / "best.pt").is_file()
    ckpt = torch.load(out / "best.pt", weights_only=True)
    assert ckpt["epoch"] == 1
    assert ckpt["provenance"]["exploratory_subset"] is False
    assert 0 <= ckpt["best_dice"] <= 1
    model, checkpoint = load_checkpoint_model(out / "best.pt")
    raw_image, _, _ = _fake_nifti_volume(catalog.cases[0])
    probability, mask = predict_volume(model, raw_image, checkpoint, batch_size=2)
    assert probability.shape == mask.shape == (32, 32, 5)
    assert np.all((probability >= 0) & (probability <= 1))
    assert set(np.unique(mask)).issubset({0, 1})
    with (out / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["validation_mean_dice"] != ""
    paths = create_plots(out / "history.csv", out / "figures")
    assert all(path.stat().st_size > 5000 for path in paths)
    with pytest.raises(FileExistsError):
        run_training(**common)
    config["training"]["epochs"] = 2
    config_path.write_text(yaml.safe_dump(config))
    resumed = run_training(**common, resume=True)
    assert resumed["epochs_completed"] == 2
    assert torch.load(out / "latest.pt", weights_only=True)["epoch"] == 2
    with (out / "history.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    with pytest.raises(ValueError, match="Cannot resume"):
        run_training(**common, resume=True, max_train_patients=1)
