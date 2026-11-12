# CV04: WideResNet + contrastive learning + KNN

CV04 is implemented as a complete development workflow: one-fold training,
resume, multiple-fold experiments, pooled development selection, frozen selected
checkpoints, independent evaluation and unlabeled image inference. The notebook
is `notebooks/CV04_VSCode_ColabDisk_Recovery.ipynb`. No new Python dependency is required.

## What follows the thesis

Source: `University_of_Tehran_Thesis_Maryam_Dadkhah.pdf` in the parent research
folder, PDF pp. 77, 94-95 and Table 4.7 on p. 101 (printed pp. 55, 72-73, 79).

1. Train a three-class WideResNet-50-2 using focal loss, inside each training fold.
2. Remove its classifier from the computation path and extract pooled 2048-D features.
3. Freeze this feature extractor and train a separate fully connected 64-D projection.
4. Learn from expert Control/Parkinson/Unknown labels using contrastive learning.
5. Classify projected query features with a bank of projected **training drawings only**.

The CV section explicitly cites **reference 13, Khosla et al., Supervised
Contrastive Learning**. Some existing extracted thesis text reverses the digits
and displays this as 31; the rendered page and bibliography establish 13. Its
[Eq. 2](https://papers.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf)
is the default `--contrastive-loss supcon`: normalize embeddings, compare dot
products at temperature tau, exclude self-comparisons, average log probabilities
over each anchor's same-class positives, then average over anchors with positives.
Singleton anchors remain in other anchors' denominators. Their count is logged.

There is a real source ambiguity: thesis p. 77 prints the pairwise margin loss:

```text
d_ij = ||z_i - z_j||_2
L = (1 / (2 * N_pairs)) * sum[same_ij * d_ij^2
                            + (1-same_ij) * max(0, margin-d_ij)^2]
```

Use `--contrastive-loss thesis_pairwise` to run that literal objective. It works
for all three classes by assigning positive pairs to equal labels. The old
balanced positive/negative group-mean comparator remains available as
`--contrastive-loss balanced`; it is a separate ablation, not the printed loss.
The old `train_representation.py --branch metric` defaults and checkpoints remain
compatible with that comparator.

## Choices the thesis does not recover

Defaults are **project settings**, not recovered thesis hyperparameters:

| Setting | CV04 default |
| --- | --- |
| Projection | Linear(2048,64), bias, no activation |
| Loss | Single-view supervised contrastive |
| Embedding normalization | L2; also used for KNN |
| Temperature | 0.1 |
| Pairwise margin when requested | 1.0 |
| Pairs | All unordered training-image pairs; no self-pairs |
| Optimizer / epochs | AdamW, lr=0.001, weight decay=0, 30 updates |
| Training batch | Entire training fold; one update per epoch |
| KNN | k=3, L2, uniform voting before development selection |
| Vote tie | Unknown |
| Checkpoint | Final epoch; saved after every projection update |

Each training drawing supplies one deterministic view. This implements the
SupCon objective on frozen features, not the paper's full two-augmentation
encoder-training pipeline. No paired augmentation recipe is specified for this
thesis stage. The focal baseline keeps its existing recorded augmentation and
sampling policy. Feature extraction reuses its evaluation preprocessing exactly.
`--no-normalize`, `--pair-policy cross_participant`, other embedding sizes and
explicit non-focal baselines are labeled ablations. With cross-participant pairs,
same-participant positives are omitted only when the class has another participant;
the one-participant fallback is logged. SupCon denominators still include all
other drawings.

The existing small cohort permits a full-fold objective; pair memory is O(N²).
No hard mining or extra sampling is silently added. Unknown is an expert class,
not an RBF distance rejection rule; KNN vote fractions are not calibrated
probabilities. Clinical labels never form training pairs or influence inference.

## One fold

Run from the repository root using the project environment. Finish the focal
baseline first and use its actual final checkpoint. Its fold, manifest hash,
three-class map and 2048-D feature metadata must match. A CE source is refused
unless `--allow-non-focal-baseline` explicitly records the ablation.

```bash
python train_contrastive_knn.py \
  --baseline-checkpoint /path/to/fold01_focal_last.pt --fold-id 1 \
  --output-dir outputs/cv04/supcon_temperature0.1/fold01
```

Add `--resume` with the same settings to continue; `--epochs` is the total budget.
A fresh run refuses a nonempty output directory. The baseline file stays intact.
For the earlier printed loss without normalization, use a separate output folder:

```bash
python train_contrastive_knn.py \
  --baseline-checkpoint /path/to/fold01_focal_last.pt --fold-id 1 \
  --contrastive-loss thesis_pairwise --margin 1 --no-normalize \
  --output-dir outputs/cv04/pairwise_margin1/fold01
```

## All folds and parameter selection

Use the focal checkpoint directory from the baseline notebook. The runner checks
for exactly one matching final baseline per requested fold before launching.
Use `--baseline-glob` to distinguish multiple source experiments; it never guesses
which duplicate checkpoint you intended. Each fold starts a new projection and
constructs its own bank. Use `--dry-run` to inspect commands, or begin with
`--fold-end 1` for a pilot.

```bash
python train_contrastive_knn_all_folds.py \
  --baseline-dir outputs/baselines/focal/checkpoints \
  --temperatures 0.07 0.1 0.2 --output-root outputs/cv04 --resume

python tune_contrastive_knn.py --runs-dir outputs/cv04 \
  --k-values 1 3 5 7 --distances l2 --weightings uniform distance \
  --output-dir outputs/cv04_selection
```

For pairwise experiments, the runner accepts `--contrastive-loss thesis_pairwise
--margins 0.5 1 2 --no-normalize`. Use a separate output root. Supply multiple
roots to `--runs-dir` to compare objectives on the **same** source baselines.
For external data locations, pass the same `--manifest` and `--root` to each
command. Selection does not relocate source baseline files; evaluate a relocated
selected checkpoint with `--baseline-checkpoint` pointing to the identical file.

Selection chooses one shared configuration across the requested folds by pooled
three-class image expert macro-F1. Ties use accuracy, then smaller k, then stable
distance/weighting/experiment order. A k larger than any training bank is skipped
and recorded. Every candidate must contain every requested completed fold.
Missing, duplicate or mislabeled OOF drawings cause failure. Fold 22 contributes
an explicit empty validation report and no fictitious score. With all 25 folds,
the output records `complete_development_schedule=true`; pilot selection is
clearly marked incomplete. Test pixels are never opened during training or
selection. An interrupted candidate is refused until resumed to its total budget.

`selection.json` records every candidate's settings, scores, input checkpoint
hashes and selected checkpoint hashes. `foldXX/selected.pt` freezes the chosen
projection, bank and KNN settings; original `last.pt` training runs are preserved.
`oof_predictions.csv` and `oof_cv04_report.json` contain pooled image results,
clinical-reference results and participant results. The participant rule is PD
only if every drawing is PD, Control if at least half are Control, otherwise
Unknown. Image Unknowns remain in the denominator. No single expert participant
reference is invented from differing drawing-level expert labels.

These OOF numbers are **development selection scores**, optimistic after tuning;
they are not nested-CV estimates or independent test results. No confidence
interval matching the thesis's unexplained +/- notation is manufactured.

## Independent evaluation and new images

```bash
python eval_representation.py --checkpoint outputs/cv04_selection/fold01/selected.pt \
  --output-dir outputs/cv04_selection/reloaded_fold01

python predict_contrastive_knn.py --checkpoint outputs/cv04_selection/fold01/selected.pt \
  --images /path/to/new_spiral.png --output-dir outputs/cv04_inference
```

Evaluation restores the exact baseline, preprocessing, projection and bank.
`--split test --confirm-heldout-test` is available only for a frozen final
protocol. A single fold's test evaluation is a single-model result; a weighted
25-model thesis ensemble is a separate, still unspecified protocol.
Unlabeled prediction accepts image files directly and requires no clinical or
expert query labels. It writes neighbor vote fractions and source hashes.

## Colab and verification

The current CV04 notebook is
`notebooks/CV04_VSCode_ColabDisk_Recovery.ipynb`. Older CV04 notebook variants
were removed to keep one entrypoint. Open it in VS Code with the existing remote
Colab GPU server; the Mac is the editor and download destination, while all
training executes on Colab's GPU.

### Setup and saved work

The notebook reads the checksum-pinned
`My Drive/ParkinsonResearch/parkinson_training_code_v7_cv04_storage_fix.zip`
and `My Drive/ParkinsonResearch/datasets/parkinson_dataset_v1.zip`. No new code
upload is required when using those existing files. Normal readable Python
sources are extracted on Colab; there is no encoded archive inside notebook cells.

It copies and verifies an existing run from Drive without changing the originals,
then rebases projection references to hash-identical copied backbones. If a local
run already exists, it is retained instead of being overwritten by older Drive
files. Checkpoints, reports, logs and ZIP exports go only under
`/content/ParkinsonRuns`, `/content/ParkinsonLogs` and `/content/ParkinsonExports`.
The notebook measures the Colab VM's available disk space automatically.

### Run All and partial training

`FOLDS_PER_RUN = 25` is the default: Run All skips completed folds, trains every
remaining requested fold, creates a verified ZIP after each, and then performs
KNN selection, independent reload and final report export. Download completed
ZIPs while training continues; files still only on `/content` are temporary.

Set `FOLDS_PER_RUN = 1` to pause after each unfinished fold for a download. In
that mode, selection safely waits until every requested projection is complete;
reload and final report export wait for selection. A missing fold is never
hashed or passed to the selector. Rerun the training cell to continue, then rerun
selection and later cells once the requested development schedule is finished.
Progress bars show completed folds, current stage epochs, batches, loss and time.

### Compact saving and recovery

Only the active focal fold retains optimizer/RNG state in one rolling last
checkpoint, saved each epoch on Colab disk. At completion the full-precision
backbone is saved separately and compared tensor-for-tensor, then the redundant
active optimizer checkpoint is removed from the local run. Completed backbones
retain fold, class mapping, preprocessing and manifest identity, and can be reused
for CV07. Small contrastive/KNN checkpoints retain their own resume state.

Each completed-fold ZIP contains one backbone, its small projection/KNN checkpoint,
recipe, history and reports. ZIPs are lossless and checksum-verified; dense float
weights may compress only modestly. The final reports ZIP adds selected projections
and reports without duplicating backbones. Download all these ZIPs to retain them
across Colab VM deletion. The optional recovery cell restores downloaded archives
on a fresh VM before the settings/import cell. Completed-fold ZIPs do not contain
an unfinished fold's optimizer state.

The active checkpoint is checked before training. On a PyTorch archive-reading
failure, a readable matching Drive copy is tried first. If no readable active
copy survives, the local unfinished work folder is preserved under
`/content/ParkinsonQuarantine` and only that fold restarts. Completed backbones
and Drive files are not removed by recovery. Metadata mismatches and unrelated
errors stop instead of being treated as corruption. The log retains full output;
the notebook displays a short excerpt of the actual error to avoid VS Code's
long-output truncation hiding its cause.

Measured backbone tensor storage is 267.63 MB/fold, or 6.69 GB for 25 folds;
an active resumable checkpoint is approximately 769.35 MB before serialization
overhead. The planner allows temporary saves, selection files and a safety margin,
plus room for ZIP exports. This concerns the Colab VM disk, not free Drive quota.

Run `python -m pytest -q tests/test_cv04_colab_disk.py tests/test_cv04_compact.py
 tests/test_training_progress.py` for saving, corruption recovery, lossless
backup/restore, progress and complete/incomplete notebook workflow checks. These
software tests use synthetic inputs, not full research-cohort GPU training.

Run `python -m pytest -q tests/test_cv04.py tests/test_representation_models.py
tests/test_representation_pipeline.py` for the targeted checks. They cover an
independent SupCon equation, finite-difference gradients, the printed pairwise
reduction, synthetic learning and unseen-vector KNN, deterministic resume,
independent reload, unlabeled inference, fold discovery, OOF coverage and
development selection without test pixels. Full cohort training has not been
performed by these software tests. Table 4.7's 69.1 +/- 1.7% is a source result,
not a result of this implementation.
