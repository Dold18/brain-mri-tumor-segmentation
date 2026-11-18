import importlib.util
import json

import numpy as np
import pytest

from brain_mri.data import load_case, read_catalog
from scripts.inspect_dataset import inspect


def _fake_dataset(tmp_path, n=3):
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    training = []
    for i in range(n):
        name = f"BRATS_{i+1:03d}.nii.gz"
        (tmp_path / "imagesTr" / name).touch()
        (tmp_path / "labelsTr" / name).touch()
        training.append({"image": "./imagesTr/" + name, "label": "./labelsTr/" + name})
    (tmp_path / "dataset.json").write_text(
        json.dumps({"numTraining": n, "modality": {"0": "FLAIR"},
                    "labels": {"0": "background", "1": "tumor"}, "training": training})
    )
    return tmp_path


def test_catalog_and_metadata_only_inspector(tmp_path):
    root = _fake_dataset(tmp_path)
    catalog = read_catalog(root)
    assert [r.case_id for r in catalog.cases] == ["BRATS_001", "BRATS_002", "BRATS_003"]
    report = inspect(root, metadata_only=True)
    assert report["labeled_case_count"] == 3
    assert report["inspected_count"] == 0


def test_reject_unsafe_paths_and_wrong_counts(tmp_path):
    root = _fake_dataset(tmp_path)
    doc = json.loads((root / "dataset.json").read_text())
    doc["training"][0]["image"] = "../../private.nii.gz"
    (root / "dataset.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="escapes root"):
        read_catalog(root)


@pytest.mark.skipif(importlib.util.find_spec("nibabel") is None,
                    reason="NiBabel not available in this execution environment")
def test_actual_nifti_io_and_affine_mismatch(tmp_path):
    import nibabel as nib

    root = _fake_dataset(tmp_path, n=1)
    case = read_catalog(root).cases[0]
    image = np.ones((16, 17, 5, 4), dtype=np.float32)
    labels = np.zeros((16, 17, 5), dtype=np.uint8)
    labels[2:5, 3:6, 2] = 2
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(image, affine), case.image_path)
    nib.save(nib.Nifti1Image(labels, affine), case.label_path)
    x, y, observed_affine = load_case(case, valid_labels={0, 1, 2, 3})
    assert x.shape == (16, 17, 5, 4)
    assert y[2, 3, 2] == 2
    assert np.allclose(observed_affine, affine)
    nib.save(nib.Nifti1Image(labels, affine + np.diag([0.5, 0.0, 0.0, 0.0])), case.label_path)
    with pytest.raises(ValueError, match="affines differ"):
        load_case(case)
