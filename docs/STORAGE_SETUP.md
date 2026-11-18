# External SSD setup — `Storage` (macOS)

This build defaults all dataset files, preprocessing caches, patient-split manifests,
checkpoints, logs, and reports to `/Volumes/Storage/brain-mri-tumor-segmentation/`.
The project source code and `.venv` can remain on the Mac's internal disk.

## Confirm your drive is mounted

```bash
ls /Volumes
# Must show Storage
diskutil info /Volumes/Storage
```

If macOS instead mounted it as `Storage 1`, reconnect or unmount the duplicate
and resolve that first. The code **refuses to use a missing SSD** rather than
creating a folder that could be on the internal drive.

## Run from the extracted project source folder

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,app]'
python -m pytest -q
python -m scripts.system_check
```

Check that the system check's `requested_data_root` begins with
`/Volumes/Storage/brain-mri-tumor-segmentation` and that sufficient free space
is available. For a first full archive download, extraction and ten-patient
quickstart, allow **at least 20 GB** free; for the full patient cache allow
substantially more. The SSD must be writable, and should use APFS or exFAT;
a restrictive read-only format will not work.

## Download and train (no path flags required)

```bash
python -m scripts.fetch_msd
python -m scripts.real_data_quickstart --device auto
streamlit run app/streamlit_app.py
```

The viewer defaults to the same SSD dataset/cache/split/checkpoint paths.
You can still override any of the command-line paths explicitly if you later
move the dataset.

## Storage layout

```text
/Volumes/Storage/brain-mri-tumor-segmentation/
├── data/
│   ├── downloads/Task01_BrainTumour.tar
│   └── Task01_BrainTumour/
├── cache/msd/
├── manifests/patient_splits.json
└── runs/
    ├── real_quickstart/
    │   ├── best.pt
    │   └── history.csv
    └── real_quickstart_validation/
```

All paths in `brain_mri/storage_paths.py` can be changed in one place. Manual
advanced CLI tools also use these paths by default. `--config` remains relative
to the project source folder, so run commands there.

If you previously downloaded the archive **onto your Mac**, move the
`Task01_BrainTumour.tar` to the SSD and point `--archive` at the existing file:

```bash
python -m scripts.fetch_msd --archive '/path/to/Task01_BrainTumour.tar'
```

Do not reformat or erase your existing SSD. The workflow creates only its own
new project folder under `/Volumes/Storage`.
