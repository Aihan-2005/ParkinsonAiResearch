# Metric learning and learned-centre RBF

**Implementation scope:** the contrastive + KNN comparator is implemented; the
RBF branch is a restricted learned-centre prototype. Neither is a completed
thesis reproduction. See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) for
the source comparison, remaining methods and verification limits.

For the implementation inventory and verified handoff status, see
[ReadMEArya2](ReadMEArya2.md).

B4 implements two **separate** modelling branches after a fold-specific CV
baseline. It does not change A1–A3 labels, participant rosters or splits.

1. Frozen WideResNet pooled features (2048-D) → linear 64-D projection →
   contrastive loss → training-only KNN bank, with three expert classes.
2. Frozen WideResNet pooled features (2048-D) → independent linear 64-D
   projection → two learned RBF centres → Control/Parkinson or rejection.

The baseline's classification head is not reused or modified. The backbone is
frozen, in evaluation mode, throughout B4. Feature extraction is scoped to one
checkpoint, manifest and development fold; there is no shared all-data cache.
This is a proposed implementation of roadmap pages 15–17. The thesis's exact
projection/mining, modified RBF loss and stage schedule were not supplied.

## Prerequisite and environment

Use the existing project environment (`requirements-lock.txt`); B4 adds no
dependencies. Run commands from the repository root with that environment active.

You need a **real fold-specific three-class baseline checkpoint** from `train.py`,
using `label_source=consensus`, the same manifest and the current separated
2048-D WideResNet baseline. The roadmap's reported stage order starts with focal
classification; CE checkpoints are accepted for separately identified ablations.
The source loss and epoch are recorded. A baseline trained on evaluation
participants must never be substituted, even if its weights have the right shape.

For example, first finish the baseline stage:

```bash
python train.py --fold-id 1 --loss focal --epochs 15 --freeze-policy early
```

Use the **exact final checkpoint path printed by that run** below. The default
baseline requests ImageNet weights. Earlier data-validator/roster inconsistencies
must be reconciled through the A-phase workflow; B4 does not discard source
ratings, remove empty participants or regenerate manifests to bypass them.

## Contrastive + KNN

Replace the example baseline path with your actual fold-1 checkpoint:

```bash
BASELINE="/absolute/path/to/fold01_focal_last.pt"
python train_representation.py --branch metric --fold-id 1 \
  --baseline-checkpoint "$BASELINE" \
  --epochs 30 --embedding-dim 64 --margin 1.0 \
  --knn-k 3 --distance l2 --knn-weighting uniform \
  --output-dir outputs/representations/metric_fold01
```

- Expert classes remain `control=0`, `parkinson=1`, `unknown=2`.
- All unordered training-image pairs are considered, with Euclidean distance.
  For a class with multiple participants, positive pairs use different
  participants. Otherwise same-participant positives are permitted and logged.
  Different-label pairs are negatives, including different drawings from the
  same participant. Missing classes and per-class participant support are logged.
- Positive loss is `distance²`; negative loss is `relu(margin-distance)²`.
  Average each nonempty pair group, then average those group means. At least
  one positive and one negative pair are required; no fake pairs are created.
- One epoch is one full-training-fold update. The current dataset has at most
  103 eligible development images, so this avoids batches without valid pairs.
  This implementation is intended for this small cohort; pair memory is O(N²).
- Projection defaults to one `Linear(2048,64)`, with no activation or L2
  normalization. `--normalize` is a distinct logged experiment.
- KNN references are projected **training images only**, in drawing-ID order.
  The query never joins the bank. Tied class votes become Unknown. Distance
  weighting uses only exact-zero neighbors when any occur in the selected k.
  Equal-distance neighbor selection is stable in drawing-ID order.
- Select k, weighting, margin, normalization and distance through development
  comparisons. KNN's Unknown is a third class; it is not an RBF rejection radius.

## RBF L1 / L2

```bash
python train_representation.py --branch rbf --fold-id 1 \
  --baseline-checkpoint "$BASELINE" \
  --distance l1 --epochs 30 --embedding-dim 64 \
  --known-margin 1.0 --reject-margin 1.0 --beta 1.0 \
  --distance-threshold 1.0 --gap-threshold 0.1 \
  --output-dir outputs/representations/rbf_l1_fold01
```

Those values are **illustrative development settings**, not recovered thesis
hyperparameters or validated thresholds. L1 and L2 have different distance
units and need separate development selection. For L2, use `--distance l2`
and a separate output directory and selection record.

The projection is initialized independently of the contrastive branch. Both
known classes must be present in the training fold. Initialize their centres
from their projected training-class means; then learn the projection and centres:

```text
d_c = sum(abs(z - centre_c))                       # L1
d_c = Euclidean norm(z - centre_c)                 # L2, not squared
known = mean_known(d_y + relu(known_margin - d_other))
reject_g = mean_group(sum_c relu(reject_margin - d_c))
loss = known + beta * mean(nonempty reject_g groups)
```

Reject groups are expert-Unknown/recorded-clinical-Control and
expert-Unknown/recorded-clinical-Parkinson. Their sizes and missing groups are
logged. Empty groups contribute no mean and produce no NaNs; unknown examples
with missing/invalid clinical origin cause an error. Clinical origin is used
**only in this training objective**, never as an inference input. Known-class
targets always use the expert label, including expert/clinical disagreements.

The head implements restricted learned centres, not the more general affine
Deep-RBF architecture. Focal loss belongs to the source classification stage;
it is not multiplied into this margin objective.

Rejection uses this order:

```text
min(d0,d1) > distance_threshold     -> Unknown: far_from_known_classes
d0 == d1                          -> Unknown: equal_distance_tie
abs(d0-d1) < gap_threshold         -> Unknown: ambiguous_distance_gap
otherwise                         -> nearest known class
```

Radius equality is accepted; gap equality is accepted unless distances tie.
The gap rule is global. Exact ties reject even when gap threshold is zero;
this completes the roadmap's zero-gap edge case without an arbitrary class bias.
Thresholds are explicit CLI inputs and frozen into the checkpoint. The script
does not optimize them or estimate a test threshold automatically.

## Resume and independent evaluation

Repeat the **same training command**, adding `--resume`, to continue `last.pt`.
`--epochs` means the total epoch budget and can increase on resume. Configuration,
source-baseline hash, manifest hash and training membership must match. Optimizer,
model, history and Torch RNG state are restored. A nonempty output directory
cannot be overwritten by a fresh run.

```bash
python eval_representation.py --checkpoint outputs/representations/metric_fold01/last.pt \
  --split val --output-dir outputs/representations/metric_fold01/reloaded_val
```

For an RBF run, supply its own `last.pt`. Evaluation restores the exact stage and
baseline, preprocessing and KNN bank. If the baseline file moved, use
`--baseline-checkpoint /new/path/to/the/same/file.pt`; its SHA-256 must match.
Use `--manifest` and `--root` consistently for nondefault data locations.

Training never requests test features. `eval_representation.py --split test` requires
`--confirm-heldout-test`; use this only after the final protocol is frozen.
There is no mismatch bypass and no inference-time threshold override. Fold 22
can train but exports empty validation predictions and null metrics, not a score.

## Outputs and provenance

Each branch/fold output directory contains:

| File | Contents |
|---|---|
| `last.pt` | Stage/config/version, epoch, model/optimizer states, RNG, loss history, baseline path/hash, manifest hash, preprocessing, training identities and diagnostics; metric branch also saves its training embedding bank and labels |
| `val_predictions.csv` | One row per held-out development image; identities, expert reference, decision, fold/stage/config provenance, stage-checkpoint SHA-256, KNN vote fractions or RBF distances and rejection reason |
| `val_report.json` | Three-class image metrics, coverage, rejection fraction, accepted-case accuracy, all-case PD recall, configuration, training support and fidelity note |

Undefined precision/recall denominators are null; accepted accuracy is null if
all cases reject. Existing all-class macro-F1 uses zero contribution for absent
classes, matching the baseline's convention; the present-class variant is also
reported. Rejected true-PD cases stay in the all-case recall denominator.

Evaluation writes analogous files in the selected output directory. Vote
fractions are not calibrated probabilities. B4 files are separate from the
baseline OOF aggregator's schema and are not silently mixed into its reports.
Keep the source baseline checkpoint alongside representation checkpoints: stage files are
small and reference, rather than duplicate, its weights.

Epoch selection uses the fixed final budget. Learning rates, margins, thresholds,
k, normalization and epoch budget are development protocol choices. No automatic
hyperparameter search is included. Logs are image-level reports; ensemble and
participant voting, clinical-reference reports and pooled B4 experiment reports
remain A4/B3 work. Triplet, OSM and GMML remain later comparisons.

## Verification and review

```bash
python -m pytest -q tests/test_representation_models.py tests/test_representation_pipeline.py
```

Checks cover loss equations and gradient directions, missing groups, exact
threshold boundaries, KNN ties/zero distances, source/manifest mismatches,
training-bank membership, frozen source preservation, independent reload,
resume versus continuous training, empty validation and locked test access.
Lifecycle tests use synthetic images and a small stand-in backbone. They verify
the pipeline, not model quality or reproduction of thesis results. A real
fold-specific baseline and development experiments are still required for
scientific validation. No held-out research test performance is claimed.

Implementation review: automated checks plus a local source review. Team/professor
review of the proposed loss, training stage and development settings is pending.

## Descriptive filenames and checkpoint compatibility

Commands are now `train_representation.py` and `eval_representation.py`; shared
code is `utils/representation_pipeline.py`; the head is `model/rbf.py` and its
class is `CentreRBF`. The environment lock is `requirements-lock.txt`.
Task labels such as B4 remain roadmap references, not executable filenames.

The checkpoint's original internal format identifier and parameter names remain
stable, so existing saved stage state dictionaries remain readable. Older
output directories are not moved. To resume an older run, use the new command
name with its original `--output-dir` and unchanged options. Old command module
names are removed; update notebooks/scripts to import the descriptive modules.
