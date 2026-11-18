# Brain MRI Tumor Segmentation — Final Evaluation

This document summarizes the **actual experiment reports supplied by the project
operator** from the November 2026 full-data training run on macOS with MPS.
The test-set outcomes were **not recomputed in the source packaging environment**;
re-run `python -m scripts.audit_release` on the Mac to verify artifact consistency
and compute the local checkpoint SHA-256 before GitHub publication.

## Protocol

- Task: binary **whole-tumor-associated-region segmentation**, including edema.
- Dataset: Medical Segmentation Decathlon `Task01_BrainTumour` (glioma cases).
- Model: four-modality 2D U-Net implemented in PyTorch; training config `configs/baseline.yaml`.
- Patient-level partitions: **338 training / 73 validation / 73 held-out test**.
- Selected checkpoint: **epoch 24** from a 30-epoch training run, selected by validation Dice.
- Fixed decision threshold: **0.50**. The validation-only sensitivity sweep tested 0.20–0.80;
  0.45 produced a mean-Dice change of only +0.0000031, so 0.50 was retained.
- Evaluation: full patient-volume mask overlap, then macro-average across patients.
- The test partition was opened **after** checkpoint selection and threshold choice.
- The test cohort is an internal split of the **labeled** MSD cases. It is not the
  separate official challenge test set and does not represent external clinical validation.

## Reported metrics

| Patient-level mean | Validation (n=73) | Held-out test (n=73) |
|---|---:|---:|
| Dice | 0.874040 | **0.885570** |
| IoU | 0.788973 | **0.801410** |
| Precision | 0.886348 | 0.881665 |
| Recall | 0.880029 | **0.898015** |

Test Dice median: 0.912085. The reported test dataset contains 39/73 cases
with Dice at least 0.90, 66/73 at least 0.80, and 2/73 below 0.70.

### Representative cases, chosen from automated ranking

- `BRATS_226`, worst observed held-out Dice 0.6061, exhibits both missed
  annotated territory and extra predicted voxels (single shown slice).
- `BRATS_175`, Dice 0.8002, generally follows a small annotated region but
  disagrees on boundaries; it is a sampled lower-performing case, **not** the
  second-worst case.
- `BRATS_419`, Dice 0.9611, shows close prediction/reference agreement in the
  selected slice and high patient-level recall.

These slice visualizations are illustrations of cases scored on **whole volumes**.
No clinical diagnostic performance or healthy-vs-tumor accuracy is claimed.

## Limitations

1. All training and evaluation cases come from the same dataset; hospital,
   scanner, protocol, and population shifts have not been independently tested.
2. No healthy-vs-tumor diagnostic validation. The cohort consists of glioma studies.
3. Binary targets include edema and other annotated tumor-associated subregions.
4. The 2D U-Net processes each slice independently; it does not explicitly model
   relationships across adjacent axial slices.
5. Some patients have substantial false-positive or false-negative regions,
   including case `BRATS_226` in the held-out set.
6. Aggregate overlap metrics are **not** a measure of clinical safety or clearance.

## Evidence and verification

Primary results on the user's external SSD:

- `runs/baseline/validation_review/metrics.json`
- `runs/baseline/final_test_epoch24/metrics.json`
- `runs/baseline/final_test_epoch24/per_patient.csv`
- `runs/baseline/best.pt`
- `manifests/patient_splits.json`

Run `python -m scripts.audit_release` to check hashes, original reports,
per-case metrics, split integrity, checkpoint epoch, and baseline config. Keep
`release_audit.json` alongside the original evaluation artifacts. Do not
publish a model file or dataset excerpts until distribution rights and sizes
have been reviewed.
