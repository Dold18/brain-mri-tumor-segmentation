# Model card — Brain MRI Tumor Segmentation (baseline v1)

## Model

- Architecture: custom 2D U-Net, 4 input channels, binary segmentation output.
- Framework: PyTorch. Model hyperparameters and preprocessing are read from
  `configs/baseline.yaml` and embedded in `runs/baseline/best.pt`.
- Selected checkpoint: epoch 24 of the 30-epoch training run.
- Binary threshold: 0.50, retained after validation-only threshold analysis.
- Input: four co-registered MRI sequences from Medical Segmentation Decathlon
  `Task01_BrainTumour` in a consistent channel order.
- Output: voxel-wise probability of belonging to the union of annotated
  tumor-associated subregions, including edema. **Not a diagnosis.**

## Intended use

Research and educational demonstration of deep learning, medical-image
preprocessing, segmentation metrics, and interactive model inference. The
viewer is suitable for exploring the benchmark scans and comparing predictions
against their reference annotations.

## Not intended for

Clinical decision-making, tumor screening, determining whether a patient has
cancer, selecting treatment, interpreting scans of unknown modality or
preprocessing, or deployment to hospitals. No validation on healthy controls
or external hospitals has been performed.

## Training and evaluation

- Dataset: 484 labeled glioma cases from MSD Task01_BrainTumour.
- Patient-level partition: 338 train, 73 validation, 73 internally held-out test.
- Model selection by full-volume mean validation Dice.
- Training: AdamW, Dice + BCE loss, fixed seed and case manifest, balanced
  slice sampling, MPS-enabled training on macOS.
- Evaluation: each patient scored over a complete 3D volume assembled from
  independent 2D slice predictions; reported cohort scores are macro-means.

| Evaluation | Mean Dice | Mean IoU | Mean precision | Mean recall |
|---|---:|---:|---:|---:|
| Validation (n=73) | 0.8740 | 0.7890 | 0.8863 | 0.8800 |
| Internally held-out test (n=73) | **0.8856** | **0.8014** | **0.8817** | **0.8980** |

These results were supplied from an actual full-data run and should be
cross-checked by `python -m scripts.audit_release` against the original
checkpoint/report files before publication. They are **not** measurements on
the official unlabeled MSD challenge test cohort. No diagnostic sensitivity or
specificity can be inferred from them.

## Failure modes and caveats

- Overprediction and missed parts of the tumor-associated region are both
  observed in lower-performing cases, including BRATS_226 (test Dice 0.6061).
- Slice-by-slice inference does not enforce continuity across adjacent slices.
- Dependence on dataset-specific normalization, channel order, alignment,
  and scanner/population characteristics limits portability.
- Reference masks reflect the dataset annotation convention; the binary class
  includes edema rather than identifying only tumor core.
- Model metrics describe overlap with provided masks; they are not evidence of
  patient safety, clinical performance, or generalization beyond this dataset.

## Reproducibility/provenance

Store and retain: checkpoint SHA-256, original patient manifest and fingerprint,
configuration, evaluation JSON/CSV, training logs, environment versions, and
code revision. Raw patient NIfTI files are excluded from the repository.
Review the dataset's attribution/share-alike requirements before redistributing
adapted images or model weights. Project code is authored and maintained by
Jacob Dold; dataset and third-party attributions remain separately required.
