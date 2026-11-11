# WideResNet + CE and WideResNet + focal: audit and run guide

Reviewed on 11 November 2026. These runs are for development comparison before
hyperparameter tuning. No research training or final test evaluation was run in
this review.

## Thesis results

Source: Maryam Dadkhah's University of Tehran thesis, Table 4.7, PDF page 101
(printed page 79). The discussion on PDF page 102 describes F1 for the target
Parkinson class; do not substitute macro F1 when comparing these numbers.

| Model | Accuracy (%) | Precision (Parkinson) | F1 (as reported) |
| --- | --- | --- | --- |
| WideResNet + cross-entropy | 54.3 +/- 2.4 | 0.67 +/- 0.05 | 0.57 +/- 0.06 |
| WideResNet + focal loss | 62.5 +/- 2.1 | 0.80 +/- 0.06 | 0.62 +/- 0.05 |

Preserve the reported +/- values; this implementation does not reconstruct their
uncertainty calculation. The focal accuracy is 8.2 percentage points above CE in
that table. These are reference results, not expected scores on our cleaned data.

## What agrees with the thesis

- ImageNet-pretrained WideResNet-50-2, RGB input, and expert three-class targets:
  `control=0`, `parkinson=1`, `unknown=2`.
- One linear 2048-to-3 output head. A 64-D metric projection belongs to the later
  representation stage, not these two baselines.
- Early stages frozen; later stages fine-tuned. The project's `early` choice
  freezes the stem, layer1 and layer2 and trains layer3/layer4 plus the head.
  Frozen-stage BatchNorm statistics also stay frozen.
- Independent models for the 25 prescribed development folds. No random
  image-level split and no training on locked test images.
- Ordinary mean CE on raw logits, and true-target multiclass focal loss
  `mean(-alpha[target] * (1 - p[target])**gamma * log(p[target]))`.
  The thesis's displayed focal equation omits the target indicator. Summing its
  term over every class literally would penalize the correct confident answer;
  the implementation uses the intended labelled-class focal objective.

The source sections are 4.2.1-4.2.2 (PDF page 65), 4.4.1 (pages 83-84), 4.4.3
(page 88), and 4.4.4.1 (pages 90-93).

## Findings and changes

1. **Missing thesis training operations:** the input baseline intentionally used
   no augmentation or oversampling. Added optional `--augmentation standard`
   (rotation, horizontal flipping, Gaussian noise) and
   `--sampling class_balanced` (inverse training-class frequency, replacement,
   one training-set-sized sample draw per epoch). The latter balances class
   probabilities in expectation, not exact counts in every batch. Both are
   training-only. Validation/test stay deterministic and use each image once.
   This follows the operations described on page 88, but not an undisclosed
   original recipe. No pixel-space SMOTE or generated synthetic spirals are
   being substituted for original samples.
2. **Incomplete/stale notebook:** it stopped after loss checks, specified full
   freezing and a 64-D embedding, and referenced a missing upload-preparation
   script. Replaced its duplicate setup/loss path with calls to the canonical
   trainer and added `prepare_colab_upload.py` to package current code and all
   103 eligible development images. The upload contains zero test images.
3. **Resume epoch misreporting:** requesting 15 epochs from a checkpoint already
   at epoch 30 could export epoch-30 predictions under a requested budget of 15.
   The trainer now rejects that request; aggregation rejects inconsistent actual
   and requested epochs as well.
4. **Large-seed crash:** the CLI accepted 63-bit seeds while NumPy's seed function
   only accepted 32-bit values. NumPy now receives `seed % 2**32`.
5. **Reproducibility:** pinned pretrained weights to ImageNet V2, the recipe the
   previous torchvision DEFAULT selected. Training operations and strengths are
   recorded in the configuration digest/checkpoint. Fold-run summaries now
   include a configuration digest so different tuning runs do not overwrite
   the same summary filename. OOF summaries include the training settings.

The architecture, CE objective, focal objective, gradient flow, and frozen
BatchNorm behavior did not need replacement.

## What the thesis does not specify

Exact optimizer/learning rates, batch size, epoch count, weight decay, torchvision
weight version, freeze boundary, focal gamma/alpha values, augmentation strengths,
and the final oversampling schedule are not recoverable from the relevant thesis
sections. The settings below are explicit initial project choices. They must not
be described as the thesis's exact hyperparameters.

| Setting | Initial value |
| --- | --- |
| Image size | 224 x 224; complete image, aspect fit and white padding |
| Pretrained weights | torchvision Wide_ResNet50_2_Weights.IMAGENET1K_V2 |
| Trainable stages | layer3, layer4 and 2048-to-3 classifier |
| Optimizer | AdamW |
| Head / backbone learning rates | 1e-3 / 1e-5 |
| Weight decay | 1e-4 |
| Epochs / batch size / seed | 15 / 8 / 42 |
| Rotation | uniform angle in [-10, +10] degrees; expand before fitting |
| Horizontal flip | probability 0.5 |
| Gaussian noise | standard deviation 0.01 in [0,1] RGB units, before normalization |
| Replacement sampling | probability per drawing proportional to 1 / training-class count |
| Focal gamma / alpha | 2 / [1,1,1] |
| Checkpoint selection | final fixed-budget epoch; no best-epoch selection |

Use the same sampling and augmentation for CE and focal to isolate the effect of
the loss. Initial focal alpha is uniform so replacement sampling handles class
reweighting once. `--focal-alpha-mode inverse_frequency` is available for a later,
separately named experiment; its weights use training-fold labels only. None of
these alpha choices is established as the thesis's original setting.

The existing PNGs are coloured spirals, but their exact time-colour mapping and
PNG/TXT pairing are not proven. The thesis separately reports a weaker
no-time-information focal model (46.9 +/- 2.5%); do not identify our inputs with
that ablation or claim verified equality to the thesis renderer. Its final
validation-weighted voting over 25 models on the test set is also not the pooled
development OOF report generated here. The cleaned cohort/exclusions may differ
from the thesis cohort. Exact reproduction needs the original training/rendering
code and settings; final ensemble work belongs after development tuning.

## Local terminal: first fold

Open the repository and its existing environment:

```bash
cd "/Users/arya/Desktop/Parkinson Prediction Research/ParkinsonAiResearch"
source .venv/bin/activate
python validate_data.py
```

For a fresh checkout, create a Python 3.11 virtual environment and install
`requirements.txt` first. The first pretrained run downloads the ImageNet weights.
`--device auto` selects CUDA, then Apple MPS, then CPU; use `--device cuda` on a
CUDA machine. Zero workers works in local terminals, notebooks and restricted
environments.

Run each experiment:

```bash
python train_all_folds.py --fold-start 1 --fold-end 1 \
  --loss cross_entropy --pretrained --freeze-policy early --label-source consensus \
  --epochs 15 --batch-size 8 --image-size 224 --lr 1e-3 --backbone-lr 1e-5 \
  --weight-decay 1e-4 --seed 42 --num-workers 0 --device auto \
  --augmentation standard --rotation-degrees 10 --hflip-prob 0.5 --noise-std 0.01 \
  --sampling class_balanced --checkpoint-dir outputs/baselines/cross_entropy/checkpoints --resume

python train_all_folds.py --fold-start 1 --fold-end 1 \
  --loss focal --focal-gamma 2 --focal-alpha-mode uniform \
  --pretrained --freeze-policy early --label-source consensus \
  --epochs 15 --batch-size 8 --image-size 224 --lr 1e-3 --backbone-lr 1e-5 \
  --weight-decay 1e-4 --seed 42 --num-workers 0 --device auto \
  --augmentation standard --rotation-degrees 10 --hflip-prob 0.5 --noise-std 0.01 \
  --sampling class_balanced --checkpoint-dir outputs/baselines/focal/checkpoints --resume
```

These are startup checks. Fold 1 has only four validation images; its score alone
cannot establish whether one loss is better. After both finish, change
`--fold-end 1` to `--fold-end 25` in both commands. Keep `--resume`: completed
folds are reused and interrupted folds continue from their last saved epoch.
Use a new output folder when changing hyperparameters. To reproduce the old input
baseline as an ablation, use `--augmentation none --sampling shuffle` in separate
output folders.

Each fold writes `*_last.pt` (model, optimizer, configuration and RNG state),
`*_history.json`, validation prediction CSV and validation report JSON. There is
no new best.pt selection. Checkpoints contain the whole backbone and AdamW state,
so 50 completed models can occupy many GB. Preserve checkpoints to resume.

## Aggregate after all 25 folds

```bash
python aggregate_validation.py --loss cross_entropy --freeze-policy early --seed 42 \
  --predictions-dir outputs/baselines/cross_entropy/validation_predictions \
  --output-dir outputs/baselines/cross_entropy/reports --require-complete

python aggregate_validation.py --loss focal --focal-alpha-mode uniform \
  --freeze-policy early --seed 42 \
  --predictions-dir outputs/baselines/focal/validation_predictions \
  --output-dir outputs/baselines/focal/reports --require-complete
```

Both create `oof_validation_summary.json`, `oof_validation_predictions.csv`,
`oof_validation_per_fold.csv`, and the fold-support audit. Expect 25 fold reports,
103 OOF images, and no score/predictions for empty fold 22. For pilot summaries,
omit `--require-complete`; the result will correctly be marked incomplete.

Read accuracy and `pooled_oof_metrics.per_class.parkinson` precision/recall/F1
alongside macro F1 and the confusion matrix. Single-class and tiny fold scores
should not be averaged blindly or treated as reliable multiclass estimates.
Send both summary JSONs, prediction CSVs and histories for the next tuning pass.
Keep final test data out of tuning.

## Colab / VS Code notebook

### Progress bars, storage and local resume

The updated notebook displays fold, epoch and batch progress, current loss,
elapsed time and estimated remaining time for the current loss. Terminal runs
also show epoch/batch bars. `--no-progress` disables them.

The notebook sets `CHECKPOINT_EVERY = 5`, saving at epochs 5, 10 and 15. The
final requested epoch is always saved. Up to four unsaved epochs can repeat
after interruption. Terminal commands retain the previous every-epoch default;
use `--checkpoint-every 5` for the notebook policy. This is an I/O setting and
does not change the model recipe or checkpoint identity, so old compatible
checkpoints remain resumable.

Before training, configuration checks a conservative budget of about 41.3 GB
free for 25 folds across both losses, including temporary writes and a safety
margin. It deliberately budgets for all selected folds, even if some already
exist. Every checkpoint save also checks the destination filesystem and keeps
the previous completed file if writing the new temporary file fails. Drive's
filesystem does not reliably expose account quota: on Colab enter the actual
`DRIVE_FREE_GB` from Google's Storage page. The configuration blocks a run that
does not fit. Do not rely on `/content` as permanent storage.

To continue on your Mac, stop the Colab training cell, preserve its files,
reload the updated notebook, select the local project Python kernel, and run
setup/configuration with `FOLD_START = 1`, `FOLD_END = 25`. Local checkpoints
stay under `outputs/baselines`; local fold 1 is reused. Colab checkpoints are
not automatically downloaded or mixed into this local run. Cross-machine
checkpoints need a separate compatibility check before reuse.

The progress update is packaged locally as
`outputs/colab_upload/parkinson_training_code_v2.zip`. The v2 archive is uploaded to
`My Drive/ParkinsonResearch`; make sufficient Drive space available before
starting the full comparison with a Colab kernel. Local training reads the updated
source directly and does not need that upload.

The notebook supports a Colab GPU kernel from either VS Code or Colab in the
browser. The notebook editor can stay on your Mac while Python, image loading
and training run on Google's machine. A remote kernel cannot read your Mac's
Desktop paths; the code and dataset must also be available on the remote machine.

### 1. Shared dataset and separate model code

The full reusable dataset has already been uploaded to the connected Google
account `aryakargar85@gmail.com` at
`My Drive/ParkinsonResearch/datasets/parkinson_dataset_v1.zip`. All model
notebooks can reuse this version without another dataset upload.

The updated notebook uses this shared dataset and a separate code archive.
Create a code snapshot locally after code changes:

```bash
cd "/Users/arya/Desktop/Parkinson Prediction Research/ParkinsonAiResearch"
python3 prepare_colab_code.py
```

The snapshot is `outputs/colab_upload/parkinson_training_code_v1.zip`. It has
been uploaded to `My Drive/ParkinsonResearch` with explicit user approval.
The training notebook can load both code and data directly from Drive.
The local notebook has the exact expected archive checksum. When preparing a
new code snapshot, update that checksum deliberately or use a new version.

Expected Drive layout before training:

```text
My Drive/
  ParkinsonResearch/
    datasets/parkinson_dataset_v1.zip
    parkinson_training_code_v1.zip
```

The shared dataset retains the original relative paths, finalized labels,
eligibility flags and prescribed fold assignments. It includes locked test data;
the training commands still select only development train/validation rows.
`DATA_ROOT` points to `/content/ParkinsonData`, `DATASET_DIR` to its `datasets`
subdirectory and `MANIFEST` to `datasets/annotations/fold_manifest.csv`.
`--root` and `--manifest` are already wired in the notebook. Keep local and Colab
outputs separate.

### 2. Connect the notebook to a GPU

**VS Code:** install the Microsoft **Python** and **Jupyter** extensions and the
official [Google Colab extension](https://marketplace.visualstudio.com/items?itemName=google.colab).
Open `notebooks/01_wideresnet50_2.ipynb`, click **Select Kernel > Colab > New Colab
Server**, sign in to Google and choose an available GPU machine, then its Python
kernel. The exact choices depend on your account and availability. See the
[official extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

**Browser:** open [Colab](https://colab.research.google.com/), choose **File >
Upload notebook** and select `notebooks/01_wideresnet50_2.ipynb`. Choose **Runtime
> Change runtime type > GPU**, then connect. Saving a copy to Drive keeps your
notebook edits there too.

### 3. Run the cells

Run the setup cell and authorize Drive access using the same account that holds
the dataset. It mounts `/content/drive`, verifies and extracts the shared data
to `/content/ParkinsonData` and the separate model code to
`/content/ParkinsonAiResearch`, installs requirements and prints the GPU name.
If you used a different Drive folder, edit `DRIVE_DIR` in that cell. If Drive
authorization does not open in VS Code, update the Colab extension and use
**Colab: Mount Google Drive to Server...** in the command palette, then rerun.

Training reads images from the extracted runtime disk. This follows Google's
[recommendation to copy an archive from Drive and extract it locally](https://research.google.com/colaboratory/faq.html),
which avoids repeated small image reads through Drive. Checkpoints, predictions,
reports and results ZIPs are written to Drive automatically:

```text
My Drive/ParkinsonResearch/runs/baselines_shared_dataset_v1/
  cross_entropy/checkpoints/
  cross_entropy/validation_predictions/
  cross_entropy/reports/
  focal/checkpoints/
  focal/validation_predictions/
  focal/reports/
  baseline_results.zip
```

Run with `FOLD_END = 1` first, then set `FOLD_END = 25` and rerun configuration and
training for the full development comparison. Both losses use the same settings
as the terminal commands.

The final cell saves a small JSON/CSV results ZIP. It excludes checkpoints. In
VS Code, download this ZIP from Drive instead of relying on the browser-specific
`files.download()` widget. Full model/optimizer checkpoints can occupy many GB,
so check Drive space before running all folds for both losses.

### 4. Continue after a runtime ends

Colab's `/content` disk is temporary. Reconnect to a GPU and rerun setup,
configuration and the desired training cells with the same ZIP, settings and
Drive output folder. Existing `--resume` behavior restores the last saved epoch;
an interrupted epoch is repeated. Completed folds are reused. Use a new output
folder when changing the training recipe. Code and dataset archives can be
versioned separately. Generated archives are ignored by Git.

## Verification

- Data validation passes on the active cleaned annotations and fold manifest.
- The full suite passes: **121 tests**. Checks cover loss behavior, frozen-stage statistics/gradients,
  train/evaluation reload, sampling isolation, seeded augmentation, both-loss
  CPU resume equivalence, epoch-budget guards and OOF aggregation.
- A full WideResNet forward/backward check uses a real training batch at
  224 x 224 for each loss, with no pretrained download or optimizer update.
- The Colab package is inspected and loaded independently across all 25 folds.
- No research accuracy has been measured by these software checks. GPU/Colab
  execution and pretrained-weight downloading still happen when you run them.
