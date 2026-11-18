# Final software/release audit

This is a focused check of the **already-trained** model and its evaluation artifacts.
No retraining, NIfTI conversion, or new test inference is performed.

## Update safely

Extract the finalization ZIP **inside** the existing project root, at:
`/Users/jdold/Desktop/brain-mri-tumor-segmentation`.
It adds `brain_mri/viewer.py`, `scripts/audit_release.py`, and new tests/docs;
it updates only the viewer, `brain_mri/visualization.py`, and `pyproject.toml`.
It does **not** contain or overwrite `best.pt`, patient data, caches, split
manifests, or evaluation reports on `/Volumes/Storage`.

## Verify on the Mac

```bash
cd /Users/jdold/Desktop/brain-mri-tumor-segmentation
source .venv/bin/activate
python -m pip install -e '.[dev,app]'
python -m pytest -q
python -m scripts.audit_release
streamlit run app/streamlit_app.py
```

The audit reads the following defaults on the `Storage` SSD:

- `runs/baseline/best.pt`
- `runs/baseline/validation_review/metrics.json`
- `runs/baseline/final_test_epoch24/metrics.json`
- `manifests/patient_splits.json`

Its output is `runs/baseline/release_audit.json` on the external SSD. This
contains file SHA-256 digests and verifies patient/split/config/metric consistency.

**Do not change** the checkpoint, selected threshold, or locked test results.
The Streamlit GUI continues to use real cached inference and only updates
presentation logic. Individual patient metrics in the viewer are for study
review, not additional tuning on the held-out test cohort.

## Final UI checklist

1. Select `BRATS_359` under **validation**, run inference, change the slice.
2. Check the enlarged original/prediction/reference panels.
3. Expand **Inspect prediction errors** and verify TP/FP/FN colors.
4. Select the already-evaluated **test** study `BRATS_419`, run inference and
   download the PNG comparison. It must reflect the actual model output.
5. Confirm the sidebar reports checkpoint epoch 24 and fixed threshold 0.50.
6. Review `docs/FINAL_EVALUATION.md` and `release_audit.json` before publication.

## What this patch cannot verify here

The packaging runtime has no access to your SSD checkpoint or full MRI dataset,
and does not have Streamlit/NiBabel installed. Synthetic tests and compilation
can pass here; the live app and audit must be executed once on the Mac.
