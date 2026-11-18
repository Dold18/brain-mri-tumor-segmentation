# Brain MRI Tumor Segmentation — Developer Setup

**For the configured external SSD `Storage`, use `docs/STORAGE_SETUP.md` first.**
The manual examples below show alternative explicit project-relative locations; when using the default CLI, inputs, caches, and outputs now live under `/Volumes/Storage/brain-mri-tumor-segmentation/`.


This is a **nonclinical research application** for binary glioma-associated
region segmentation (including edema), not patient diagnosis or brain-tumor
classification. Its outputs are not validated for clinical decisions.

## 1. Install on macOS (Python 3.11)

Extract the current ZIP into a **new folder**. Do not merge the folder with
older code archives, because the Python package has been renamed `brain_mri`.
Open Terminal in the folder containing `pyproject.toml`.

```bash
# If missing: brew install python@3.11
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python --version  # must show Python 3.11.x
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,app]'
python -m pytest -q
python -m scripts.smoke_test
```

Do not reuse a `.venv` built by the macOS system Python 3.9. The code can use
CPU, CUDA, or Apple Silicon MPS if available. `--device auto` prefers CUDA,
then MPS, then CPU. CPU is fine for unit tests; full training on CPU may be slow.

## 2. Obtain and inspect real MRI data

Download **Medical Segmentation Decathlon — Task01_BrainTumour** from
https://medicaldecathlon.com/dataaws/ and extract it locally so the layout is:

```
data/Task01_BrainTumour/
  dataset.json
  imagesTr/*.nii.gz
  labelsTr/*.nii.gz
```

The source is a four-modality glioma dataset, not a healthy-vs-tumor diagnosis
dataset. Inspect actual labels and geometry before proceeding:

```bash
python -m scripts.inspect_dataset --dataset-dir data/Task01_BrainTumour --metadata-only
python -m scripts.inspect_dataset --dataset-dir data/Task01_BrainTumour --limit 3
python -m scripts.prepare_splits --dataset-dir data/Task01_BrainTumour \
  --output manifests/patient_splits.json --seed 2026
```

Store the split manifest permanently; do not recreate it to optimize scores.
No individual slices from a subject can appear in different splits.

## 3. Quick functional real-data run

This step is **exploratory**. It prepares eight training and two validation
patients, trains a small network, and produces a validation-subset report.
Its scores are *not* a medical or publication-grade benchmark.

```bash
python -m scripts.prepare_cache --dataset-dir data/Task01_BrainTumour \
  --config configs/quickstart.yaml --max-train-patients 8 \
  --max-validation-patients 2

python -m brain_mri.train --dataset-dir data/Task01_BrainTumour \
  --config configs/quickstart.yaml --max-train-patients 8 \
  --max-validation-patients 2 --output-dir runs/quickstart --device auto

python -m brain_mri.evaluate --checkpoint runs/quickstart/best.pt \
  --dataset-dir data/Task01_BrainTumour --partition validation \
  --max-patients 2 --output-dir runs/quickstart_validation --examples 2
```

Generated results go to `runs/` and are ignored by Git. The inference
checkpoint contains the configuration and split fingerprints, and the report
is labeled `exploratory_evaluation: true`.

## 4. Baseline training and validation

Training/validation slices are cached locally as memory-mappable arrays. The
cache may require tens of gigabytes of SSD space with the full cohort.

```bash
python -m scripts.prepare_cache --dataset-dir data/Task01_BrainTumour \
  --config configs/baseline.yaml

python -m brain_mri.train --dataset-dir data/Task01_BrainTumour \
  --config configs/baseline.yaml --output-dir runs/baseline --device auto

python -m scripts.plot_history --history runs/baseline/history.csv \
  --output-dir runs/baseline/figures
```

If interrupted, rerun the same `brain_mri.train` command with `--resume`.
Only extend `training.epochs`; incompatible models or split manifests are
rejected. Inspect validation curves and select a model without viewing test
performance.

## 5. Final locked-test evaluation

Use this **only after the model and evaluation configuration are frozen**.
The CLI will refuse exploratory checkpoints and requires explicit held-out
confirmation. It computes Dice/IoU/precision/recall per patient over entire
MRI volumes, writes a CSV and JSON report, and can export scan comparisons.
Examples are selected by rank (poor, typical, strong), not hand-picked.

```bash
# Includes locked-test cache; only do this for final evaluation.
python -m scripts.prepare_cache --dataset-dir data/Task01_BrainTumour \
  --config configs/baseline.yaml --include-test

python -m brain_mri.evaluate --dataset-dir data/Task01_BrainTumour \
  --checkpoint runs/baseline/best.pt --partition test \
  --confirm-heldout --output-dir runs/final_test --examples 3
```

`runs/final_test/metrics.json` and `per_patient.csv` contain **real outputs only
when those commands run successfully on real held-out studies**. Do not present
unit-test/synthetic results as actual medical-model performance.

## 6. Interactive segmentation viewer

```bash
streamlit run app/streamlit_app.py
```

In the sidebar, provide the valid checkpoint, dataset, split manifest and cache
paths. Select a validation case, choose an MRI modality and axial slice, then
click **Run model inference**. Predictions come from the selected U-Net
checkpoint. The reference mask is from the source annotation. If the checkpoint
is still exploratory, the app displays that warning.

The viewer's display contrast and orientation are for visualization and do not
constitute a radiologist workstation. Test-case selection should be used only
after final model selection/evaluation is frozen.

## Limitations and artifact policy

- NiBabel integration must be checked on real `.nii.gz` files on the target machine.
- Scanner generalization, healthy screening, and clinical accuracy are **not validated**.
- Existing metrics describe segmentation of the annotated glioma-associated
  abnormality including edema; not histopathology or clinical diagnosis.
- Raw MRI files, generated caches, experiment folders, and large `.pt`
  checkpoints are excluded from Git. Check source dataset licensing and
  attribution before distributing any derivatives or model artifacts.

## v4 accelerated on-ramp

For the recommended single-command official dataset download and first complete
exploratory training run, see **`docs/REAL_DATA_RUN.md`**. It replaces the
manual steps in sections 2–3 for new installations and includes disk-space
checks, checksum verification, safe extraction, and end-to-end orchestration.
