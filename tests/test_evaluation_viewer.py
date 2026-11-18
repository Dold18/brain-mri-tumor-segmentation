"""Test locked evaluation and actual model inference using explicitly synthetic MRI scans."""

import csv
import json

import numpy as np
import pytest
import torch

from brain_mri.cache import PreprocessSettings, prepare_case
from brain_mri.data import read_catalog
from brain_mri.evaluate import run_evaluation
from brain_mri.inference import predict_cached_case
from brain_mri.model import UNet2D
from brain_mri.splits import write_splits
from brain_mri.train import _hash
from brain_mri.visualization import color_overlay, grayscale_uint8


def _fake_case(record, **kwargs):
    n = int(record.case_id[-2:])
    r = np.random.default_rng(n)
    image = np.zeros((32, 32, 4, 4), dtype=np.float32)
    image[4:28, 4:28] = r.normal(80, 7, size=(24, 24, 4, 4))
    label = np.zeros((32, 32, 4), dtype=np.uint8)
    label[8:15, 10:18, n % 4] = 2
    return image, label, np.eye(4)


def _make_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr("brain_mri.cache.load_case", _fake_case)
    root = tmp_path / "data"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    rows = []
    for n in range(8):
        name = f"CASE_{n:02d}.nii.gz"
        (root / "imagesTr" / name).touch()
        (root / "labelsTr" / name).touch()
        rows.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})
    (root / "dataset.json").write_text(json.dumps({
        "numTraining": 8,
        "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"},
        "labels": {"0": "background", "1": "edema", "2": "tumor", "3": "enhancing"},
        "training": rows,
    }))
    catalog = read_catalog(root)
    splits_path = write_splits(catalog, tmp_path / "manifest.json")
    split_data = json.loads(splits_path.read_text())
    settings = PreprocessSettings()
    cache_dir = tmp_path / "cache"
    for record in catalog.cases:
        prepare_case(record, cache_dir, settings)

    model = UNet2D(in_channels=4, base_channels=4).eval()
    ckpt = tmp_path / "best.pt"
    state = {
        "schema": 1, "epoch": 3, "model_state": model.state_dict(),
        "model_settings": {"in_channels": 4, "base_channels": 4},
        "preprocessing": {"clip_low_percentile": 0.5, "clip_high_percentile": 99.5},
        "provenance": {"split_hash": _hash(split_data), "exploratory_subset": False},
        "threshold": 0.5,
    }
    torch.save(state, ckpt)
    return root, splits_path, cache_dir, ckpt, model, split_data


def test_checkpoint_inference_matches_slice_batching(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    _, _, cache, _, model, splits = _make_fixture(tmp_path, monkeypatch)
    arr = np.load(cache / splits["validation"][0] / "image.npy", mmap_mode="r")
    p1 = predict_cached_case(model, arr, batch_size=1)
    p3 = predict_cached_case(model, arr, batch_size=3)
    assert p1.shape == (4, 32, 32)
    np.testing.assert_allclose(p1, p3, rtol=1e-5, atol=1e-6)
    assert np.isfinite(p1).all()
    assert (p1 >= 0).all() and (p1 <= 1).all()


def test_evaluator_validates_partition_and_checkpoint_provenance(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    root, splits_path, cache, checkpoint, _, splits = _make_fixture(tmp_path, monkeypatch)
    args = dict(checkpoint_path=checkpoint, dataset_dir=root,
                split_manifest=splits_path, cache_dir=cache,
                output_dir=tmp_path / "evaluation", device="cpu")
    with pytest.raises(PermissionError, match="locked"):
        run_evaluation(**args, partition="test")
    with pytest.raises(ValueError, match="never train"):
        run_evaluation(**args, partition="train")

    report = run_evaluation(**args, partition="test", confirm_heldout=True, examples=1)
    assert report["evaluation_partition"] == "test"
    assert report["patients_evaluated"] == len(splits["test"])
    assert 0 <= report["mean_dice"] <= 1
    assert len(report["cases"]) == len(splits["test"])
    assert (tmp_path / "evaluation" / "metrics.json").is_file()
    assert report["exploratory_evaluation"] is False
    with (tmp_path / "evaluation" / "per_patient.csv").open() as f:
        assert len(list(csv.DictReader(f))) == len(splits["test"])
    assert len(report["example_images"]) == 1
    assert __import__("pathlib").Path(report["example_images"][0]).is_file()

    with pytest.raises(ValueError, match="in full"):
        run_evaluation(**{**args, "output_dir": tmp_path / "test_partial"},
                       partition="test", confirm_heldout=True, max_patients=1)
    with pytest.raises(FileExistsError, match="must be new"):
        run_evaluation(**args, partition="test", confirm_heldout=True)

    val_report = run_evaluation(**{**args, "output_dir": tmp_path / "val_subset"},
                                partition="validation", max_patients=1)
    assert val_report["patients_evaluated"] == 1
    assert val_report["exploratory_evaluation"] is True

    state = torch.load(checkpoint, weights_only=True)
    state["provenance"]["exploratory_subset"] = True
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="Exploratory checkpoint"):
        run_evaluation(**{**args, "output_dir": tmp_path / "eval_other"},
                       partition="test", confirm_heldout=True)
    state["provenance"]["exploratory_subset"] = False
    state["provenance"]["split_hash"] = "changed"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="Checkpoint/split-manifest mismatch"):
        run_evaluation(**{**args, "output_dir": tmp_path / "eval_third"},
                       partition="validation")


def test_overlay_does_not_modify_source_and_rejects_bad_shapes():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    original = image.copy()
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:6, 2:8] = 1
    displayed = color_overlay(image, mask, alpha=0.5)
    assert displayed.shape == (10, 10, 3)
    assert displayed.dtype == np.uint8
    np.testing.assert_array_equal(image, original)
    assert grayscale_uint8(image).shape == (10, 10)
    with pytest.raises(ValueError, match="aligned"):
        color_overlay(image, mask[:9])
    with pytest.raises(ValueError, match="binary"):
        color_overlay(image, mask + 2)
