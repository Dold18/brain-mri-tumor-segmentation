# Portable runtime configuration

The measured experiment used a macOS external SSD at `/Volumes/Storage` for its measured experiment.
**Those original paths are preserved when no environment override is set.**

For new computers, set one environment variable *before starting Python*:

```bash
export BRAIN_MRI_STORAGE_ROOT="$HOME/brain-mri-data"
```

For Linux, Windows shells, or other drive paths, set the variable to an absolute
path appropriate to the machine. It changes the **defaults** for the dataset,
preprocessing cache, patient-split manifest, checkpoint runs, and reports. All
individual CLI path flags remain available.

Then execute:

```bash
python -m scripts.system_check
python -m scripts.fetch_msd
python -m scripts.inspect_dataset --limit 3
python -m scripts.prepare_splits
python -m scripts.prepare_cache --config configs/baseline.yaml
python -m brain_mri.train --config configs/baseline.yaml --device auto
```

When using the original development Mac, **leave this variable unset** to
continue using `/Volumes/Storage/brain-mri-tumor-segmentation` without moving
anything. The disconnected-SSD guard remains active for any attempted writes to
`/Volumes/Storage` (including custom CLI paths there). The override does **not**
copy data, remap existing checkpoints, or change the original 338/73/73 split.
A fresh reproduction uses its own split manifest and should report it as a new
run; do not claim to have reproduced the existing benchmark unless the original
split and configuration are used and results are actually rerun.

## Model weights

Weights are intentionally not included in Git. Publication can provide the
verified checkpoint separately as a GitHub Release asset or model-hosting
artifact, subject to rights review. If distributed, provide its SHA-256, software
revision, configuration, split-manifest provenance, download instructions, and
research-only limitations.

Known verified checkpoint SHA-256:

```text
de19d735e98f610856151fe08ef8669f083649e100cce771be3c01d70ad8cf44
```


## Reproducing the published split

On the development Mac, export the validated patient identifiers into the source
repository using `python -m scripts.export_split_manifest`. The command checks
both the file SHA-256 and the split semantic hash against `results/release_audit.json`.
It must not regenerate the train/validation/test assignments. The export contains
public dataset case identifiers only, not scans, annotations, or patient medical records.

On another computer, after downloading the dataset and choosing your storage
root, copy `manifests/patient_splits.json` into the configured manifest location
*before* preparing the cache. Do not run `prepare_splits` again for the same
reported experiment. Check the documentation before comparing scores across
platforms; nondeterministic hardware effects can still cause differences.
