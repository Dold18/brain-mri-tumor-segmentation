# Release checklist (public GitHub)

## Already evidenced

- [x] Full 338/73/73 patient-level split of the 484 labeled glioma cases.
- [x] Real 30-epoch baseline training on the development Mac, MPS acceleration.
- [x] Epoch-24 checkpoint selected using validation Dice, threshold 0.50 retained.
- [x] Full internally held-out 73-patient evaluation and per-case evidence.
- [x] Local artifact audit status **passed**; checkpoint SHA-256 recorded.
- [x] Synthetic test suite executed in packaging environment (46 passed, 1 skipped because NiBabel unavailable).
- [x] Dataset license/figure attribution documented.

## Still required before publication

- [ ] On development Mac: `python -m pytest -q` after applying this release patch.
- [ ] On development Mac: `python -m scripts.audit_release` still passes with existing checkpoint, splits, and evaluation files.
- [x] Actual browser screenshot provided, confirming prediction and patient-level metrics for validation case BRATS_421; saved as `figures/app_screenshot.png`.
- [ ] Confirm the expandable error overlay and PNG export work in the development browser (not visible in the supplied screenshot).
- [ ] Run `python -m scripts.release_preflight` and resolve reported tracked files/secrets/large binaries.
- [ ] Confirm there are no patient NIfTI volumes, caches, `.env` files, proprietary assets, or `.pt` checkpoints staged.
- [x] Source-code [MIT License](../LICENSE), copyright (c) 2026 Jacob Dold; MRI-derived figures retain CC BY-SA 4.0 attribution.
- [x] Current project authored and maintained by Jacob Dold; no institutional sponsorship claimed.
- [ ] Record the release Git commit; the audit fingerprints the training configuration, not the source-code commit.
- [ ] Check that all links/screenshots render correctly in GitHub and CI passes.

- [ ] Export the exact audited patient-level split manifest using `python -m scripts.export_split_manifest` and verify it appears under `manifests/patient_splits.json`.
- [ ] Decide whether to distribute the separately hosted model checkpoint; confirm applicable terms and its SHA-256. See `docs/RELEASE_PREPARATION.md`.

## Fixed experiment fingerprint

- Best checkpoint SHA-256: `de19d735e98f610856151fe08ef8669f083649e100cce771be3c01d70ad8cf44`
- Split-document semantic hash: `d9cadeb0e1d8d5415fe91356250939238c559f92d145198085cdb60a564d351d`
- Internal held-out test mean Dice: `0.8855699862259567`
- Patient-macro means are evaluated over the **whole 3D volume**, not per-slice averages.
- Refer to `results/release_audit.json` for all original file fingerprints.

## Reproduction notes

Do not recreate the fixed split manifest when comparing a reproduction to the original run. For a fresh training attempt, report its own seed, splits, configuration, code revision, and test results. The shared repository contains experiment summaries but does not claim a bitwise-reproducible GPU training trace on another machine.
