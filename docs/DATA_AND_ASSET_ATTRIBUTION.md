# Dataset and example-figure attribution

## Source dataset

- Name: **Medical Segmentation Decathlon**, `Task01_BrainTumour` (glioma MRI studies).
- Source: https://medicaldecathlon.com/dataaws/ and https://registry.opendata.aws/msd/
- License: **Creative Commons Attribution–ShareAlike 4.0 International** (`CC BY-SA 4.0`): https://creativecommons.org/licenses/by-sa/4.0/
- Citation: Antonelli, M. et al., *The Medical Segmentation Decathlon*, Nature Communications **13**, 4128 (2022), https://doi.org/10.1038/s41467-022-30695-9
- A related original dataset description: Simpson, A. L. et al., *A large annotated medical image dataset for the development and evaluation of segmentation algorithms* (2019), https://arxiv.org/abs/1902.09063

## Changes to the figures

The images in `figures/heldout_*.png` were **adapted from MSD Task01_BrainTumour MRI scans** by selecting individual axial slices from the public cases, applying the project's display conversion, and adding overlays from the original ground-truth mask and the project's trained 2D U-Net prediction. Colored masks, text and layout were added by the application. The figures are not original unmodified source images.

- `figures/heldout_BRATS_226.png` — selected axial comparison from held-out case BRATS_226.
- `figures/heldout_BRATS_175.png` — selected axial comparison from held-out case BRATS_175.
- `figures/heldout_BRATS_419.png` — selected axial comparison from held-out case BRATS_419.
- `figures/app_screenshot.png` — screenshot of the operational Streamlit interface, showing an annotated MRI slice from **validation** case BRATS_421 together with the model prediction and patient-level metrics. The underlying MRI content is derived from MSD Task01_BrainTumour; interface layout is project-generated.

**Figure license:** The dataset-derived MRI figures in this directory are offered under **CC BY-SA 4.0**, subject to the original dataset attribution above. They are provided for research illustration, not medical use. The original dataset maintains its own licensing; these terms do not automatically apply to the project's source code.

The repo does **not** distribute raw NIfTI volumes, reference masks, a dataset mirror, or model weights. Before distributing trained weights or more derived patient imagery, confirm any additional upstream rights and use the correct attributions.

## Project source code

The project Python source code is copyright (c) 2026 Jacob Dold and released under the [MIT License](../LICENSE). This license does not replace the dataset license or the CC BY-SA 4.0 terms for the dataset-derived figure assets.
