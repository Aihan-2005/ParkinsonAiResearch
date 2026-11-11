# Available-data validation across predetermined seeds

This workflow keeps the thesis workbook assignments and the separate locked test
set. It evaluates the available participants without pretending that every
prescribed two-subject pair has usable images. It does not implement or evaluate
the thesis's final validation-weighted test ensemble.

## Evaluation procedure

1. Choose the model settings, fixed epoch budget and random seeds before running.
   The initial seed list is `42 43 44`; these are project choices.
2. For each seed, start an independent model for every one of the 25 workbook
   development folds. All eligible images of its assigned subjects stay outside
   that model's training set. Other development subjects provide training data.
3. Keep the final-epoch checkpoint. Small validation sets are not used to select
   a best epoch. Each fold saves its held-out predictions and a report, including
   an explicit empty report when no validation images exist.
4. Aggregate each eligible development drawing exactly once per seed. Do not
   average the small fold F1 scores, do not add predictions for missing images,
   and do not evaluate a subject with models trained on that subject.
5. Calculate an image-weighted confusion matrix and metrics, and subject-balanced
   image-label metrics. The latter assigns image weight `1 / drawings_from_that_subject`,
   so every subject contributes total weight one. This does not invent one
   clinical diagnosis for a subject whose drawings have different expert labels.
6. Repeat the complete schedule for the predetermined seeds. Summarize one pooled
   evaluation per seed with a mean and sample standard deviation; never select the
   best seed or count repeated predictions as additional independent subjects.

Subject-balanced macro F1 over all configured classes is the primary repeated
summary. Image-weighted metrics, Parkinson precision/recall/F1, class support,
confusion matrices, per-subject image accuracy and individual-run diagnostics
remain available. Pooled metrics summarize the available held-out predictions;
they are not an unbiased estimate after unrestricted hyperparameter searches and
are not the accuracy of a final ensemble on independent test data.

## Missing subjects and completion

The cleaned local dataset currently has 8 complete pairs, 16 incomplete pairs and
1 empty pair. Its available development cohort is 103 images from 32 participants.
The nominal workbook development roster has 50 participants. These counts are
computed from the supplied manifest and assignments, not hardcoded into scoring.

| Pair status | Behavior |
| --- | --- |
| `complete_pair` | Both assigned subjects have eligible labelled images; include their held-out predictions. |
| `incomplete_pair` | Only one assigned subject has eligible labelled images; retain that person's held-out predictions and identify the unavailable partner. |
| `empty_pair` | No held-out images; no score or predictions. Keep the explicit fold report. |

`prediction_coverage_complete=true` means all 25 reports and all expected
available-image predictions are present. It does **not** mean all prescribed
subject pairs were complete. `all_subject_pairs_complete` and `pair_status_counts`
report that separately. `--require-complete` enforces prediction coverage.

The complete-pair-only result is a secondary sensitivity analysis. It evaluates a
smaller, potentially different cohort and must not replace the all-available-data
result without an explicit protocol decision. A fully complete thesis-style
pair protocol requires recovering missing eligible recordings or formally
changing the assignments.

## Run from the repository root

Preview the actual cohort and the training schedule without training:

```bash
python evaluate_development.py --seeds 42 43 44 \
  --loss cross_entropy --epochs 15 --augmentation standard --sampling class_balanced \
  --output-dir outputs/repeated_validation/cross_entropy --dry-run
```

Run the same plan:

```bash
python evaluate_development.py --seeds 42 43 44 \
  --loss cross_entropy --epochs 15 --augmentation standard --sampling class_balanced \
  --output-dir outputs/repeated_validation/cross_entropy --resume
```

This requests **75 model runs**, not three models. The pretrained first run may
download ImageNet weights. It can require substantial compute and checkpoint
storage; keep the output folder on persistent storage when using Colab. Extra
training options such as `--device cuda`, learning rates and batch size are
forwarded to `train_all_folds.py`. See that script's `--help` for those options.

Use `--loss focal` and a different output folder for the focal experiment, keeping
the other settings and seeds fixed to compare losses. The augmentation and
sampling settings above follow the existing baseline guide; their numerical
strengths are documented project choices, not recovered thesis settings.

If predictions already exist in this workflow's seed folders, repeat the same
command with `--aggregate-only` to rebuild reports without training. Keep its
training options unchanged. To summarize seed summaries stored elsewhere:

```bash
python summarize_validation_seeds.py --seeds 42 43 44 \
  --summaries /path/seed42/reports/oof_validation_summary.json \
              /path/seed43/reports/oof_validation_summary.json \
              /path/seed44/reports/oof_validation_summary.json \
  --output-dir outputs/repeated_validation/combined
```

Older single-seed summaries must first be regenerated by the current
`aggregate_validation.py`. No retraining is required when the original
final-epoch prediction CSVs and reports are present and consistent.

## Outputs and checks

The output folder contains:

- `evaluation_plan.json`: fixed seeds, settings, cohort/assignment hashes and
  expected coverage. A changed plan requires a new output folder.
- `pair_coverage.json`: the actual available and unavailable subjects for each pair.
- `seed_<seed>/checkpoints/`: independent fold checkpoints.
- `seed_<seed>/validation_predictions/`: original per-fold prediction CSVs/reports.
- `seed_<seed>/reports/`: pooled image-weighted and subject-balanced metrics,
  per-subject results, complete-pair-only sensitivity results, class/pair coverage,
  and prediction/roster hashes.
- `combined/Report.md`, `combined/repeated_validation_summary.json` and
  `combined/metrics_by_seed.csv`: the repeated evaluation, with no best-seed selection.
- `evaluation_run.json`: execution plan and completion/failure state.

Aggregation checks workbook roles independently from the manifest's claimed
partition, cross-fold label/identity consistency, final-epoch/configuration
provenance, and exactly one held-out prediction per expected drawing. Repeated
summarization rejects missing/duplicate seeds, changed prediction files, and
different cohorts, assignments, settings or epoch budgets. The workflow stops
on failed training or incomplete prediction coverage instead of silently
omitting a bad seed.

An empty fold may be explicitly reported while the available-data evaluation is
complete. An interrupted run or a missing available drawing makes it incomplete.

## Interpretation and limits

The standard deviation across seeds describes training randomness. It is **not a
population confidence interval** and does not solve the small participant count.
Different images of one person and overlapping CV training runs are dependent.
No additional independent participants appear by repeating training.

No ensemble weights are inferred from tiny pair scores or from the pooled score.
If a final test ensemble is later implemented, uniform voting is a possible
separately documented adaptation; this evaluator does not run that stage. The
locked test data stay out of development fitting and scoring.

Hyperparameter searches, epoch selection and rejection-threshold tuning require
inner participant-grouped splits if the outer CV result is to evaluate the full
selection procedure. This implementation freezes the supplied settings; it does
not silently tune them on outer held-out predictions or implement nested CV.

## Verification

The full suite passes **129 tests**. The end-to-end evaluation test trains 50
independent small synthetic models across two seeds, exports real predictions,
and runs the same aggregation and repeated-summary paths. It checks model resets,
complete/incomplete/empty pairs, subject weighting and exclusion of a deliberately
unavailable test image. Other checks reject changed recipes, partial coverage,
missing seeds and altered prediction files.

The real cleaned-data plan and its fresh development-only Colab export both
confirm 103 images, 32 subjects and the 8/16/1 pair breakdown. No full research
training or test-performance evaluation was launched during implementation.
