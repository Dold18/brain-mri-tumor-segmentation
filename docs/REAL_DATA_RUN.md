> **Configured for the external SSD `Storage`.** Use the three commands in `docs/STORAGE_SETUP.md`; this file explains the underlying workflow. The dataset, cache, manifest, and run outputs default to `/Volumes/Storage/brain-mri-tumor-segmentation/`. Older relative-path examples below are not the recommended invocation for this build.

# First real MRI training run (v4)

This document replaces the manual dataset steps for the **first exploratory run**.
The project implements binary **glioma-associated region segmentation**, not
brain-tumor diagnosis or healthy-vs-tumor classification.

## 1. Start in a new v4 project folder

On macOS, unzip the v4 archive into its own folder. Open Terminal **inside**
that folder (the folder containing `pyproject.toml`). Use Python 3.11 instead
of the macOS system Python 3.9:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,app]'
python -m pytest -q
python -m scripts.system_check
```

The system check prints available CUDA/MPS acceleration and remaining storage.
If using a large external SSD, replace `data` in the check with the desired
absolute SSD location, such as `/Volumes/MySSD/mri-data`. Make sure it is a
trusted volume with enough space and that it can be written by your account.

## 2. Download official data (one command)

The authoritative source is [Medical Segmentation Decathlon](https://medicaldecathlon.com/dataaws/),
Task01_BrainTumour, with a public AWS download managed by MONAI. Dataset
license: **CC BY-SA 4.0**. The source archive is approximately **7.4 GB**;
its extracted files take additional space. For a local full-cohort cache,
plan for **tens of gigabytes more**; for a quickstart, allow at least 18–20 GB
free before downloading. If storage is tight, put `--output-root` and later
`--cache-dir` on an external SSD.

```sh
python -m scripts.fetch_msd
```

- The script downloads `Task01_BrainTumour.tar` from the official S3 HTTPS URL.
- It can resume an interrupted transfer (`*.part`) when the server supports
  HTTP Range. It checks the official MONAI MD5 checksum before extraction.
- It safely extracts only regular files under `Task01_BrainTumour/` and verifies
  the complete training catalog before making the dataset directory visible.
- If the dataset already exists and the catalog is complete, it reuses it.

Already downloaded the official tar? Skip the download with:

```sh
python -m scripts.fetch_msd \
  --archive "/path/to/Task01_BrainTumour.tar"
```

The resulting directory should be `data/Task01_BrainTumour` with
`dataset.json`, `imagesTr`, and `labelsTr` inside it.

## 3. Run a real MRI quickstart (one command)

```sh
python -m scripts.real_data_quickstart --device auto
```

This command:
1. Creates or reuses the once-only 70/15/15 **patient-level** partition;
2. Opens an actual image/mask NIfTI pair to check label geometry;
3. Caches **eight training patients and two validation patients**;
4. Trains the smaller quickstart U-Net on those real MRI slices;
5. Saves a model checkpoint and labeled *exploratory* validation report;
6. Exports representative MRI/prediction/ground-truth images.

Artifacts created:

```
manifests/patient_splits.json          # stable and should not be recreated
cache/msd/                             # preprocessed .npy studies, Git ignored
runs/real_quickstart/best.pt           # genuinely trained exploratory checkpoint
runs/real_quickstart/history.csv       # real loss and validation history
runs/real_quickstart_validation/       # real validation metrics and comparison PNGs
```

The quickstart uses only **2 training epochs with 16 sampled slices per epoch**;
it is a pipeline validation, **not** a reliable segmentation benchmark. A low
Dice score at this stage is expected and is not a reason to alter the split.
To rerun training, choose `--output-dir runs/real_quickstart_02`. You can reuse
the split manifest and preprocessed cache.

## 4. Open the interactive application

```sh
streamlit run app/streamlit_app.py
```

In the sidebar use these paths for the quickstart:

- **Checkpoint:** `/Volumes/Storage/brain-mri-tumor-segmentation/runs/real_quickstart/best.pt`
- **Dataset:** `data/Task01_BrainTumour`
- **Split manifest:** `/Volumes/Storage/brain-mri-tumor-segmentation/manifests/patient_splits.json`
- **Preprocessed cache:** `cache/msd`

Select a cached validation patient and click **Run model inference**. The
segmentation overlay comes from the neural network checkpoint.

## 5. Full baseline and formal evaluation

Only after quickstart works, follow `docs/DEVELOPER_SETUP.md` for complete
training and validation. It requires caching the designated training and
validation patients and will consume much more time, RAM, and SSD space.

Do not run held-out test evaluation until baseline model/configuration and any
threshold selection are frozen. The public official challenge test images lack
public reference masks; the project's held-out set is the **locally defined**
15% of the 484 labeled cases.

**Local validation required:** This execution environment has no internet
access to retrieve the 7.4 GB dataset and cannot install NiBabel, so actual
NIfTI integration and real patient training must be verified on the user's
computer. The automated CPU tests are not clinical model results.
