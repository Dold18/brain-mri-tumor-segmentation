# Release preparation

**Maintainer:** Jacob Dold. The Python source is MIT-licensed, while the MRI
dataset and dataset-derived figures are separately attributed as CC BY-SA 4.0.
No medical data or trained weights should be committed to Git.

## Before staging

From the development Mac, in the project root:

```bash
source .venv/bin/activate
python -m pytest -q
python -m scripts.audit_release
python -m scripts.export_split_manifest
```

The last command exports a **verified copy** of the exact split manifest from
the external SSD to `manifests/patient_splits.json` and rejects any hash
mismatch. It does not regenerate a split or copy MRI volumes. Review the
case identifiers before staging. This manifest is needed to replicate the
reported 338/73/73 assignment.

## Inspect and stage safely

```bash
git init
git add .
python -m scripts.release_preflight
git status --short
git diff --cached --stat
```

The preflight checker examines tracked paths and flags private data folders,
model files, secrets and files over 25 MiB. Manually inspect `git status` too.
Do **not** stage `data/`, `cache/`, `runs/`, `.venv/`, `.env`, `.nii.gz`, or `.pt`.

After all checks, commit the audited source and documentation. If publishing
to GitHub with GitHub CLI (`gh`), verify you are signed into the correct account
and confirm the destination repository does not already exist. Create/push the
repository only when you choose to make it public.

## Checkpoint access

The recorded checkpoint on the development SSD is:

```text
/Volumes/Storage/brain-mri-tumor-segmentation/runs/baseline/best.pt
```

SHA-256:

```text
de19d735e98f610856151fe08ef8669f083649e100cce771be3c01d70ad8cf44
```

Weights are *not* contained in this source patch. Before redistributing the
checkpoint, review dataset-derived model distribution terms. A GitHub Release
asset can host the `.pt` separately from tracked source files. Document its
license, hash, research-only scope, configuration, and tested install procedure.
For local verification:

```bash
shasum -a 256 "/Volumes/Storage/brain-mri-tumor-segmentation/runs/baseline/best.pt"
```

A model checkpoint alone is not a stand-alone demonstration: inference also
requires four correctly ordered and preprocessed MRI sequences. Do not imply
that raw arbitrary MRI uploads are supported or that the system diagnoses
patients.

## Release gate

- Code tests and CPU smoke test succeed on a clean install or CI.
- Release audit passes on the Mac with the frozen checkpoint.
- Split manifest has been exported and verified against the audit.
- The Streamlit viewer runs a real checkpoint with the intended data paths.
- README figures and dataset attribution are intact.
- Code and figure licenses remain clearly differentiated.
- No protected data, secrets, or model weights are committed to Git.
