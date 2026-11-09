# ReadMEArya1 — Preprocessing and DataLoader Handoff

**Documentation date:** 2026-11-09

**Scope:** The image-input part of task **B1** on page 21 of the implementation roadmap, up to the WideResNet model.

This file documents Arya's implementation alongside the teammate's READMEs. The data and fold handoff is in [MOJI2.md](MOJI2.md); the detailed B1 usage guide is in [B1_PREPROCESSING.md](B1_PREPROCESSING.md).

## 1. Work completed

The path from reading an image to returning a batch ready for the model was implemented and checked:

```text
fold_manifest.csv
    -> Select the fold, role and eligible drawings
    -> Decode the PNG
    -> Composite on white and convert to RGB
    -> Preserve aspect ratio and pad to 224 x 224
    -> Convert to a tensor and normalize
    -> Return image batches, targets and identifiers
```

This builds on the teammate's data infrastructure. The A1–A3 label-generation rules, retained doctors and workbook assignments were preserved. Developing WideResNet, Deep RBF and research-model training experiments was outside this handoff.

## 2. File organization

The implementation follows the supplied project structure, using `config/`, `data/` and `utils/`:

| Path from the repository root | Responsibility |
| --- | --- |
| `config/data_config.py` | Data paths, input size, normalization constants and versioned preprocessing configuration |
| `config/settings.py` and `config/__init__.py` | Preserve access to existing settings and export the input configuration |
| `data/dataloader.py` | Image transforms, tensor validation, `SpiralDataset` and `make_loader` |
| `data/dataset.py` | Re-export the earlier public names for compatibility; implementation lives in `data/dataloader.py` |
| `data/splits.py` | Additional leakage checks using image hashes, alongside participant and duplicate-group checks |
| `utils/image_review.py` | Development-only before/after grid and source provenance |
| `utils/check_inputs.py` | Complete input and loader check without running a model |
| `train.py` | Connect training/validation transforms and save preprocessing metadata in checkpoints and history |
| `eval.py` | Restore and verify the same preprocessing configuration during evaluation |
| `tests/` | Transform, Dataset, DataLoader, leakage, grid and train/evaluation integration tests |
| `requirements-b1-lock.txt` | Exact package versions used in the verified environment |

The unused top-level `preprocessing/` folder was removed. Preprocessing runs inside `data/dataloader.py`. The separate `outputs/preprocessing/` directory contains generated reports and images, not implementation code.

## 3. Implemented preprocessing method

Pipeline version: **`spiral-full-frame-v1`**

1. Decode the PNG, composite alpha onto a white background and convert to RGB. RGB, RGBA, grayscale and palette images are supported.
2. Fit the complete image into a square using uniform scaling and Pillow bilinear interpolation. Center the image and fill the remaining space with white. A 640 x 480 input becomes 224 x 168, with 28 white rows above and below.
3. Convert to a `float32` tensor in `C x H x W` order and divide pixel values by 255.
4. Normalize the RGB channels using:

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

The default output for each image is **`float32[3, 224, 224]`**, containing finite values.

Following the roadmap's first input baseline, augmentation is disabled. Training and evaluation use identical deterministic image operations. Cropping, flipping, rotation, denoising, thresholding and colour changes are not applied, preserving the full image geometry and existing colours. No statistics are fitted using the full dataset or held-out test set.

Processing happens when an image is read. Raw files are not overwritten and no processed-image cache is created. The configuration, pipeline version and configuration hash are recorded for reproducibility.

## 4. Dataset and DataLoader contract

The active input is **`datasets/annotations/fold_manifest.csv`**. The older random-split `manifest.csv` is not an input to this loader.

- `fold_id`: an integer from 1 to 25, identifying the current development model fold.
- `split`: `train`, `val` or `test`, corresponding to the manifest roles `train`, `validation` and `test_locked`.
- `label_source="consensus"`: use expert labels as targets: `control=0`, `parkinson=1`, `unknown=2`.
- `label_source="real"`: use the recorded clinical cohort as the target; expert labels remain available in the metadata.

Only eligible drawings with the requested role are selected. Quarantined and unassigned drawings are excluded from model input. The loader checks for participant, duplicate-group and image-hash overlap between roles. Inconsistent partition or development-fold assignments also raise errors.

Important sample keys include `image`, `label`, `drawing_id`, `participant_id`, `role`, `fold_id`, `expert_label`, `expert_target` and `clinical_label`. Identifiers and clinical references remain separate from image tensors; they are not added to the input pixels. Missing or unfinished annotations are not automatically converted into the Unknown class.

Batch outputs:

```text
image: float32[B, 3, 224, 224]
label: int64[B]
```

Training samples are shuffled using a specified seed. Validation and test preserve manifest order. Workers receive reproducible seeds, and the final short batch is retained. A custom transform fully replaces the default transform and is not normalized a second time; its output must satisfy the image shape and data-type contract.

Example, run from the repository root:

```python
from config import PreprocessingConfig
from data.dataloader import make_loader, build_train_transform, build_eval_transform

config = PreprocessingConfig(image_size=224)

train_loader = make_loader(
    fold_id=1, split="train", batch_size=8, seed=42,
    transform=build_train_transform(config),
)
val_loader = make_loader(
    fold_id=1, split="val", batch_size=8,
    transform=build_eval_transform(config), allow_empty=True,
)

batch = next(iter(train_loader))
images, targets = batch["image"], batch["label"]
```

For a different input size, pass the matching `image_size` to both the transform configuration and the loader. On macOS, create worker-based loaders inside an `if __name__ == "__main__":` guard.

## 5. Local data and fold preparation

To make the default paths runnable in this checkout, the teammate's existing A1–A3 functions were run on the original sources, generating the current tables in `datasets/annotations/`. Original files beside the repository were preserved; no original PNG or TXT was deleted or replaced. Hash comparisons confirmed that all 294 local PNG/TXT files match the supplied dataset.

| Item | Recorded result |
| --- | ---: |
| Raw PNG / TXT files | 147 / 147 |
| Eligible development images | 103 |
| Eligible locked-test images | 36 |
| Quarantined duplicate images | 4 |
| Unassigned images | 4 |
| Manifest rows | 3,675: 147 drawings x 25 folds |

Workbook folds 1–25 form the development set for training and validation. Folds 26–34 form the fixed test set. In each development run, the current fold provides validation participants and the remaining development folds provide training participants.

Fold 22 has no validation images. The loader supports this with `allow_empty=True`; a validation score must not be fabricated for that fold.

Dataset files and generated outputs are ignored by Git. A fresh clone must prepare its data and current annotation tables using the A1–A3 handoff.

## 6. Run input checks without a model

Run these commands from the repository root:

```bash
source .venv/bin/activate
python -m utils.check_inputs --fold-id 1 --batch-size 8
python validate_data.py
python -m pytest -q
```

For a fresh checkout, the verified environment uses Python 3.11 on macOS ARM64:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-b1-lock.txt
```

For data stored elsewhere:

```bash
python -m utils.check_inputs --manifest /path/to/fold_manifest.csv --root /path/to/project
```

## 7. Verification results and outputs

The last code verification run passed **48 automated tests**. They cover full-frame geometry, alpha handling, normalization and its inverse, corrupt inputs, invalid tensors, determinism, worker loading, leakage rejection, raw-file preservation, compatibility with earlier imports and configuration handoff between training and evaluation.

Real-data checks for fold 1 found:

- Training: **99 images from 31 participants**.
- Validation: **4 images from 1 participant**.
- Combined: all **103 eligible development images**, with no samples dropped or repeated by the loader.
- Across all 25 folds, only fold 22 has empty validation. Its full input check also passed with 103 training images and zero validation images.
- Annotation, label, fold and image-path validation passed.

Generated files under `outputs/preprocessing/`:

| Output | Purpose |
| --- | --- |
| `development_review.png` | Original versus processed images; normalization is inverted for display |
| `review_selection.json` | Fixed drawing IDs for repeating the same grid |
| `review_grid.json` | Source hashes, selection reasons, dimensions, pipeline metadata and QC-only flags |
| `b1_report.json` | Counts, batch shapes, labels, source-integrity checks and manifest-integrity checks |
| `fold_checks.json` | Training/validation counts for each fold and raw-file hash comparison |
| `setup_report.json` | Local dataset setup provenance |

The grid uses development images only. Development duplicates are clearly marked quarantined/QC-only and remain excluded from the loader. The B1 input-check command does not open test or unassigned images for its batches or grid. The grid is ready for team review; its existence does not establish clinical approval.

## 8. Integration and remaining work

`train.py` records preprocessing settings and their hash in checkpoints and run history. `eval.py` restores and verifies those settings. Older checkpoints without this metadata are rejected. Integration was tested using a small stand-in model; that test is not a research-model training result.

Remaining work includes WideResNet development/verification, completion of the training and final evaluation tasks, embeddings and Deep RBF, model and participant voting, and verification of PNG-to-TXT pairing. No research model was trained and no held-out test performance was evaluated in this phase. Identity, diagnosis and confidence-policy questions from the data handoff were not resolved or changed by preprocessing.

The original implementation-roadmap PDF was not edited in this phase. This README records the current implementation status.
