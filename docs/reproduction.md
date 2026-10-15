# Reproduction

Install with `python -m pip install -e '.[dev]'` from the repository root.
Use `python scripts/run.py --help` to discover commands. The self-contained new
base experiment is documented in [Kaggle setup](kaggle.md).

All Hydra runs save resolved configuration. New base runs also save model and
optimizer state, stable splits, training histories, complete trajectories,
validation/test predictions, metadata and a factual report. Checkpoints use
minimum noisy-validation loss; test labels are not used for selection.

Legacy and historical evaluation commands need local artifacts. This repository
does not distribute datasets, weights or generated reports. Paths in historical
configs are examples of local runs, not downloadable artifacts. Old 32x32
ResNet-34 and new 224x224 pretrained ResNet-50 checkpoints are different formats.
Do not interchange them or apply old meta heads to the new base as if calibrated.

Run offline checks without training or dataset downloads:

```bash
python -m pytest -q -m 'not integration'
python -m compileall -q src scripts
python scripts/check_file_sizes.py --history
```

Tests marked `integration` require actual data and may trigger downloads. The
Kaggle configuration can be inspected without a GPU:

```bash
python scripts/run.py train-kaggle-base --cfg job --resolve
```
