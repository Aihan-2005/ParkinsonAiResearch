# Representation learning and RBF implementation audit

**Date:** 2026-11-11

**Conclusion:** the repository implements a contrastive-learning/KNN comparator
and a learned-centre RBF prototype. It does not implement every metric-learning
method or establish exact reproduction of the thesis's final Deep-RBF model.
The earlier description of the core implementation referred to the proposed
roadmap comparator, not full method coverage. Current names and documentation
make that distinction explicit.

## 1. Why the RBF model file is small

The head is one part of the system. The existing WideResNet supplies frozen
features. The other responsibilities are separated as follows:

| Module | Responsibility |
| --- | --- |
| `model/embedding.py` | Linear projection and optional normalization |
| `model/rbf.py` | `CentreRBF` head, centre initialization and rejection rules |
| `utils/rbf_loss.py` | Known-class and rejected-origin loss masks and reductions |
| `utils/metric_learning.py` | Contrastive pairs/objective and KNN |
| `utils/representation_pipeline.py` | Frozen features, provenance, inference and reports |
| `train_representation.py` | Optimization, stage checkpoints and resume |
| `eval_representation.py` | Independent reload and evaluation |

File length does not establish correctness. Equations, learning behavior,
source fidelity and real held-out results must be assessed separately.

## 2. RBF equation audit

Primary source: Habib Zadeh, Hosseini and Sra,
[Deep-RBF Networks Revisited](https://arxiv.org/pdf/1812.03190),
equations 2.3, 3.1 and 3.2. Local sources inspected were the thesis PDF pages
94–100 and presentation PDF page 27. The presentation formulas were inspected
visually because they are embedded as artwork.

| Component | Finding |
| --- | --- |
| Current head | `norm(projection(features) - centre_class, p)`: a shared projection and two centres |
| Paper head | `norm(A_class.T @ features + b_class, p) ** p`: class-specific affine units, not implemented here |
| L1 | Sum of absolute differences, matching the comparator's stated units |
| L2 | Unsquared Euclidean norm; differs from the paper's powered L2 unit |
| Known loss | Correct-class distance plus incorrect-class hinge penalties; same per-example term structure as slide 27 / paper equation 3.1 |
| Negative loss | Hinge penalties for every known class; same per-example term structure as slide 27 / paper equation 3.2 |
| Rejected groups | Expert-Unknown divided by recorded clinical origin, matching the described grouping |
| Aggregation | Known-example mean plus beta times mean of nonempty reject-group means: proposed balancing, not a recovered thesis equation |
| Rejection | Far-first threshold, exact tie, global distance-gap check; gap scope and boundary completion are documented project choices |

An exponential Gaussian activation is not required by the cited paper's
distance-output formulation. Adding one would change loss and threshold units.
The original thesis's complete modified loss, balancing and stage settings are
not supplied. The class is therefore named **`CentreRBF`**, not presented as the
general affine paper head. No new head or distance-power variant was silently
substituted during this audit.

## 3. Metric-learning coverage

The contrastive term comparison uses Hadsell, Chopra and LeCun,
[Dimensionality Reduction by Learning an Invariant Mapping](https://www.cs.toronto.edu/~hinton/csc2535/readings/hadsell-chopra-lecun-06-1.pdf),
equation 4.

| Component | Status |
| --- | --- |
| Linear 2048-to-64 projection; optional normalization | Implemented |
| Positive squared distance; negative squared-hinge contrastive term | Implemented |
| Cross-participant positives and logged within-participant fallback | Implemented |
| All unordered training pairs | Implemented for the small current cohort |
| Equal positive/negative group-mean weighting | Implemented adaptation; not the original uniform-pair reduction |
| Training-only KNN with L1/L2 and uniform/distance voting | Implemented |
| Fold/source checks, resume, reload and image reports | Implemented |
| Triplet, OSM, GMML | Not implemented |
| Hard/semi-hard mining or scalable balanced minibatch sampler | Not implemented; the current method uses all eligible pairs |
| Automatic k/margin/threshold selection | Not implemented; settings are supplied explicitly |
| Representation-stage fold runner and pooled OOF report | Not implemented |
| Backbone fine-tuning in this stage | Not implemented; the source backbone stays frozen |
| Research-cohort training and final independent test results | Not established by software tests |

Unknown is a third class for contrastive + KNN and a rejected training example
for RBF. Clinical origin is used only to form rejected training groups. Neither
branch's decisions depend on the expert/clinical labels attached to evaluation
features.

## 4. Audit changes

- Replaced task-numbered filenames, imports and commands with descriptive names.
- Named the head `CentreRBF` and documented its restrictions in the model and
  emitted fidelity notes.
- Added finite-value/shape guards to direct distance calls and validate all
  initialization labels before changing either centre.
- Added finite-difference gradient checks for contrastive loss, RBF loss and the
  L1/L2 head through its input features.
- Added synthetic optimization checks: contrastive loss decreases and KNN
  classifies unseen toy vectors; both RBF variants learn known classes and
  rejection of both synthetic clinical-origin groups.
- Added checks that changing evaluation labels cannot change decisions.
- Repaired the outdated baseline integration test to use a compatible small
  model, final checkpoint and real optimizer/epoch/prediction/reload paths.
- Corrected stale preprocessing links and renamed the environment lock and
  newly generated input-check report.

## 5. Filename migration and compatibility

| Former path | Current path |
| --- | --- |
| `train_b4.py` | `train_representation.py` |
| `eval_b4.py` | `eval_representation.py` |
| `utils/b4_pipeline.py` | `utils/representation_pipeline.py` |
| `model/deep_rbf.py` | `model/rbf.py` |
| `tests/test_b4_models.py` | `tests/test_representation_models.py` |
| `tests/test_b4_pipeline.py` | `tests/test_representation_pipeline.py` |
| `tests/test_b1_check.py` | `tests/test_input_checks.py` |
| `README/B4_METRIC_RBF.md` | `README/METRIC_LEARNING_AND_RBF.md` |
| `requirements-b1-lock.txt` | `requirements-lock.txt` |

Old names above are migration history, not executable files. New input checks
write `input_report.json`. Existing ignored output artifacts are preserved.
The original internal checkpoint format identifier and parameter names remain
stable, so saved configuration digests and state dictionaries remain readable.
Use the new command with the original `--output-dir` to resume an existing run.
New default outputs use `outputs/representations/`.

## 6. Verification and limits

The full audit run passed **102 tests**. Of these, **34** cover representation
learning/RBF models and lifecycle. Run the subset with:

```bash
python -m pytest -q tests/test_representation_models.py tests/test_representation_pipeline.py
```

These checks support correctness of the specified comparator. They do not prove
clinical accuracy, exact thesis fidelity, optimal thresholds, GPU reproducibility
or performance on the research cohort. The annotation/roster validator mismatches
remain separate data-integration work; the green software suite does not settle
them. A real fold-specific baseline and agreed development experiments are still
required.

Next scientific gate: one real fold through baseline, representation training,
resume, independent reload and held-out development evaluation; then development
selection and the fold schedule. Exact original-method fidelity requires the
professor's code or confirmation of the unresolved head/loss/stage choices.
