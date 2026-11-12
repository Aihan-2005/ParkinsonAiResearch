# CV07: WideResNet + Triplet + KNN

Only the Triplet + KNN comparison from roadmap CV07 is implemented here.
OSM and GMML are outside this implementation. No new dependency is required.
Use `notebooks/03_triplet_knn.ipynb` locally or in Colab.

## Thesis method and exact objective

Source: the parent folder's `University_of_Tehran_Thesis_Maryam_Dadkhah.pdf`,
PDF p. 77 (printed p. 55), pp. 94-95 (printed pp. 72-73), and Table 4.7 on
p. 101 (printed p. 79). The rendered equation has **squared norms**, not an
unsquared Euclidean margin:

```text
L = (1 / N_triplets) * sum max(0, ||z_anchor - z_positive||_2^2
                                - ||z_anchor - z_negative||_2^2 + margin)
```

1. Reuse a completed, fold-specific three-class WideResNet-50-2 trained with focal loss.
2. Bypass its classifier; freeze and evaluate its 2048-D pooled feature extractor.
3. Train a separate fully connected projection, defaulting to the thesis's 64 dimensions.
4. Form anchor/positive/negative relationships using expert Control/Parkinson/Unknown
   labels. Positive means same expert class and a different drawing; negative means
   a different expert class. Clinical labels do not define triplets or predictions.
5. Build KNN's reference bank from projected **training drawings only**, then classify
   projected validation or new drawings. Backbone and projection use the exact source
   baseline's evaluation preprocessing.

Unknown is a learned third class and the result of a KNN vote tie. It is not a
distance rejection threshold. Neighbor vote fractions are not calibrated probabilities.

## Recorded project choices

The thesis does not recover the original margin, mining, normalization, optimizer,
epoch budget, exact activation stack, or KNN settings. These defaults are explicitly
recorded in every checkpoint; they must not be described as recovered thesis values.

| Setting | Default |
| --- | --- |
| Projection | Linear(2048, 64), with bias and no activation |
| Normalization | None; `--normalize` is an optional ablation |
| Triplet objective | Printed squared-L2 hinge, margin 1.0 |
| Mining | All valid ordered triplets, including zero-loss terms in the denominator |
| Positive policy | All different drawings sharing an expert label |
| Optimization | AdamW, lr 0.001, weight decay 0, 30 epochs |
| Update batch | Entire training fold; one projection update per epoch |
| KNN before selection | k=3, L2 distance, uniform voting; ties become Unknown |
| Checkpoint | Final epoch, with atomic saves after each update |

The default does not introduce hard or semi-hard mining. `--triplet-mining batch_hard`
is an explicit ablation: choose the farthest positive and nearest negative for each
valid anchor, then average the hinge over valid anchors including inactive terms.
`--positive-policy cross_participant` prefers positives from another participant;
when a class has only one participant, its same-participant fallback is logged.
Anchors without positives cannot contribute triplets but can remain negatives and
KNN bank entries. Their counts and missing classes are recorded. A fold without
any valid triplet is refused. There are no self-positive pairs or manufactured views.

All-triplet computation uses pair distances plus sorted-negative prefix sums. It
matches the complete triplet mean without allocating a cubic triplet tensor;
working storage is O(N^2), appropriate for this cohort. CPU/CUDA use direct distances;
MPS uses an equivalent centered Gram calculation and CPU search indices. Device
roundoff can differ; GPU/MPS execution has not been verified by the CPU tests.

## Train and resume one fold

Run from the repository root after completing the matching focal baseline:

```bash
python train_triplet_knn.py \
  --baseline-checkpoint /path/to/fold01_focal_last.pt --fold-id 1 \
  --output-dir outputs/cv07/triplet_all_margin1/fold01
```

Append `--resume` with the same recipe; `--epochs` is the total epoch budget.
Changed settings, source hashes, manifests or membership are refused on resume.
A fresh run refuses a nonempty destination. Baseline weights are never updated.
The source checkpoint must match the fold, manifest, class mapping and preprocessing.
`--allow-non-focal-baseline` records a deliberate source ablation.

## All folds, margin experiments and KNN selection

```bash
python train_triplet_knn_all_folds.py \
  --baseline-dir outputs/baselines/focal/checkpoints \
  --margins 0.5 1 2 --output-root outputs/cv07/triplet_v1 --resume

python tune_triplet_knn.py --runs-dir outputs/cv07/triplet_v1 \
  --k-values 1 3 5 7 --distances l2 --weightings uniform distance \
  --output-dir outputs/cv07/triplet_v1_selection
```

Use `--fold-end 1` on **both** commands for a pilot. The runner validates exactly
one source checkpoint per requested fold before launching any training; narrow
`--baseline-glob` when multiple baseline experiments exist. `--dry-run` prints the
plan. Set a new output root when changing the recipe. With external data, pass
the same `--manifest` and `--root` to every command.

Selection compares completed CV07 checkpoints only. Every candidate must contain
all requested folds and use identical source baselines within each fold. It chooses
one shared margin/projection/KNN configuration by pooled three-class expert macro-F1,
then accuracy, smaller k, and stable distance/weighting/candidate order. Oversized k
values are skipped and logged. Duplicate, missing or mislabeled OOF drawings fail
coverage checks. Fold 22 emits an explicit empty-validation report.

Outputs include `selection.json`, `oof_predictions.csv`, `oof_cv07_report.json`,
participant predictions, and frozen `foldXX/selected.pt` checkpoints. The participant
rule is PD only if all drawings are PD, Control if at least half are Control,
otherwise Unknown. Full 25-fold completion is recorded separately from pilot scores.
Selected inference checkpoints cannot be resumed as training runs.

These scores are **development selection scores**, optimistic after tuning; they
are not independent test estimates. Training and selection never load test pixels.
Table 4.7's 62 +/- 1.9% is a thesis result, not measured accuracy of this implementation.

## Independent evaluation and new drawings

```bash
python eval_representation.py \
  --checkpoint outputs/cv07/triplet_v1_selection/fold01/selected.pt \
  --output-dir outputs/cv07/reloaded_fold01

python predict_triplet_knn.py \
  --checkpoint outputs/cv07/triplet_v1_selection/fold01/selected.pt \
  --images /path/to/new_spiral.png --output-dir outputs/cv07/new_predictions
```

The source baseline file must remain available. A relocated identical copy can be
supplied with `--baseline-checkpoint`. Reload validates manifest/source hashes and
bank membership without opening training images. New-image inference takes no query
labels and writes predictions, vote fractions and provenance. The shared evaluator's
test split requires `--split test --confirm-heldout-test` after freezing the final
protocol. This is single-model evaluation; a thesis-style 25-model weighted ensemble
needs a separately defined weighting protocol.

## Colab and verification

Copy `outputs/colab_upload/parkinson_training_code_v4_cv07.zip` to
`My Drive/ParkinsonResearch/`, then open `notebooks/03_triplet_knn.ipynb`.
The notebook checks the code archive checksum and uses the existing shared dataset
and focal-baseline Drive paths. It defaults to a one-fold pilot. Existing notebooks
and code archives retain their pinned versions. After code changes, regenerate the
v4 archive and update the new notebook's checksum together.

```bash
python -m pytest -q tests/test_cv07.py tests/test_cv04.py \
  tests/test_representation_models.py tests/test_representation_pipeline.py
```

Checks cover the independent triplet equation and finite-difference gradients,
singleton/absent classes, inactive hinges, synthetic learning and unseen-vector KNN,
deterministic resume, source preservation, independent reload, unlabeled inference,
fold command construction, OOF selection, bank isolation and absent test pixels.
These software tests do not train the research cohort or establish clinical accuracy.
