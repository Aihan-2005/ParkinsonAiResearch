# ParkinsonAiResearch

Repeated development evaluation: [pooled held-out predictions, missing-subject reporting and summaries across seeds](README/DEVELOPMENT_EVALUATION.md).

WideResNet + CE / focal: [thesis comparison, implementation audit and run commands](README/WIDERESNET_BASELINES.md).
The complete [baseline notebook](notebooks/01_wideresnet50_2.ipynb) calls the same training scripts.

Metric learning and RBF implementation: [64-D contrastive + KNN and RBF L1/L2](README/METRIC_LEARNING_AND_RBF.md).
Run `train_representation.py` after a fold-specific baseline, then reload with `eval_representation.py`.

The RBF branch is a learned-centre prototype; metric learning currently covers
contrastive + KNN. See the [implementation audit](README/IMPLEMENTATION_AUDIT.md)
for source fidelity, test results and remaining methods.

Arya's metric-learning and RBF handoff: [ReadMEArya2 — implementation, verification and remaining work](README/ReadMEArya2.md).

Current input pipeline: [preprocessing and data loading](README/ReadMEArya1.md).

Arya's implementation handoff: [ReadMEArya1](README/ReadMEArya1.md), alongside the teammate's READMEs.

It preserves the complete spiral image, pads to 224 × 224, normalizes RGB channels, and loads only eligible images for the requested workbook fold. Start with a model-free input check:

```bash
source .venv/bin/activate
python -m utils.check_inputs --fold-id 1
python validate_data.py
python -m pytest -q
```

Preprocessing and loading live in `data/dataloader.py`, with settings in `config/data_config.py` and review utilities in `utils/`, following the supplied project layout.

See the preprocessing handoff for environment setup, data preparation requirements, sample/batch contracts, review outputs and known limitations. The [A-phase handoff](README/MOJI2.md) documents the original label/fold work; its descriptions of unfinished preprocessing are superseded by the preprocessing handoff. [MOJI1](README/MOJI1.MD) describes the historical random-split version.
