# Implementation and verification status (November 17, 2026)

## Implemented

Four-input 2D U-Net, NIfTI loading and preprocessing, deterministic patient
splits, memory-mapped MRI cache, balanced slice sampling, patient-volume
training/validation metrics, checkpointing/resumption, locked held-out test
evaluation, threshold analysis, Streamlit viewer with real inference, color
comparisons and downloadable slice images, and automated tests.

## Actual training/evaluation (performed on the user's Mac)

The operator completed a 30-epoch training run on macOS with MPS and uploaded
validation and held-out test metrics. Checkpoint epoch 24 was selected by
validation. A 0.50 threshold was retained. Reported held-out performance on
73 internal test patients: **mean Dice 0.885570, IoU 0.801410, precision
0.881665, recall 0.898015**. See `docs/FINAL_EVALUATION.md`.

The source packaging environment cannot access the user's SSD checkpoint or
full MRI dataset and has not independently reproduced those medical-model
predictions. The reported held-out `metrics.json` file *was* checked for
consistency between patient confusion counts and overall metrics.

## Local packaging tests

Python syntax compilation, CPU synthetic training/inference tests, UI diagnostic
helper tests, split/evaluation guardrail tests and release-report integrity
checks pass. One NIfTI-related test skips in the packaging environment because
NiBabel is not installed. Live Streamlit layout requires a final Mac browser
check; Streamlit is not installed here.

## Storage and safety

The Mac defaults to `/Volumes/Storage/brain-mri-tumor-segmentation`. All large
NIfTI data, caches, training artifacts and checkpoints remain on the SSD and
are excluded by `.gitignore`. The release audit reads the actual local files;
it does not change the model, threshold or reported evaluation data. Research
software only, not a validated clinical diagnostic device.
