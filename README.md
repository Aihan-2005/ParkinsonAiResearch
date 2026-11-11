# ParkinsonAiResearch

B4 implementation: [64-D contrastive + KNN and RBF L1/L2](README/B4_METRIC_RBF.md).
Run `train_b4.py` after a fold-specific baseline, then reload with `eval_b4.py`.

Arya's B4 handoff: [ReadMEArya2 — implementation, verification and remaining work](README/ReadMEArya2.md).

Current B1 input pipeline: [preprocessing and data loading](README/B1_PREPROCESSING.md).

Arya's implementation handoff: [ReadMEArya1](README/ReadMEArya1.md), alongside the teammate's READMEs.

It preserves the complete spiral image, pads to 224 × 224, normalizes RGB channels, and loads only eligible images for the requested workbook fold. Start with a model-free input check:

```bash
source .venv/bin/activate
python -m utils.check_inputs --fold-id 1
python validate_data.py
python -m pytest -q
```

Preprocessing and loading live in `data/dataloader.py`, with settings in `config/data_config.py` and review utilities in `utils/`, following the supplied project layout.

See the B1 guide for environment setup, data preparation requirements, sample/batch contracts, review outputs and known limitations. The [A-phase handoff](README/MOJI2.md) documents the original label/fold work; its descriptions of unfinished preprocessing are superseded by the B1 guide. [MOJI1](README/MOJI1.MD) describes the historical random-split version.
