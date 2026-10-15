# Room for Doubt

Distilling training dynamics for inference-time uncertainty estimation.

The project records a classifier's per-example margin history, predicts that
history from an image and candidate class, and uses the predicted trajectory to
score reliability at inference. Downstream actions include Top-2 correction and
uncertainty-weighted test-time augmentation.

## Current runnable experiment

The Kaggle entry point trains a stronger **CIFAR-100N** base classifier:

- ResNet-50 with torchvision ImageNet-1K V2 initialization, 224-pixel inputs;
- 50 epochs, cross-entropy warmup then generalized cross entropy;
- official observed CIFAR-100N labels; fixed 45k/5k train/validation split;
- best checkpoint selected by minimum noisy-validation cross entropy;
- clean test evaluated after selection, once at the end of the run;
- complete 50-epoch probability-margin trajectories for later distillation;
- mixed precision, both GPUs through DataParallel, resumable epoch checkpoints.

This is a new pretrained base-model experiment. It does not establish an
improvement from dynamics distillation, and **70% accuracy is a target, not a
guarantee**. The new base requires its own meta heads and evaluation adapter;
historical ResNet-34 checkpoints and preprocessing are not interchangeable.

See [Kaggle setup and launch](docs/kaggle.md).

## Local installation

Python 3.10+:

```bash
python -m pip install -e '.[dev]'
python scripts/run.py --help
python scripts/run.py train-kaggle-base --cfg job --resolve
```

Inspecting configuration does not download data or start training. A local GPU run:

```bash
python scripts/run.py train-kaggle-base output_dir=outputs data_dir=data
```

The default Kaggle configuration downloads CIFAR images, official noisy labels,
and ImageNet weights automatically. No private checkpoints are required.

## Research code

```text
configs/                 Hydra experiment configurations
src/room_for_doubt/
  data/                  Dataset loading, identities and splits
  models/                Base classifiers and trajectory predictors
  training/              Base and meta-model training
  dynamics/              Trajectory formats
  inference/             Flow sampling
  evaluation/            Metrics and inference-time policies
  analysis/              Matched comparisons and bounded studies
  reporting/             Per-run configuration, metrics and reports
  plotting/              Scientific plots
  legacy/                Historical exploratory commands
scripts/                 Launchers and repository size guard
tests/                  Lightweight offline checks
```

Historical commands may require locally stored data, trajectories and weights.
Their presence in the package does not make them compatible with the new base.
[Reproduction](docs/reproduction.md) and [layout](docs/repository_layout.md).

Generated reports, checkpoints, datasets, raw predictions and agent notes remain
local. Every new run still creates `config.yaml`, `metrics.json`,
`run_metadata.json`, `report.md`, tables and applicable checkpoints/artifacts.

## Protocol

Use noisy training labels for training and noisy validation for checkpoint and
policy selection. Use clean test labels only for final evaluation. Any clean
training-label detector supervision must be labeled **oracle**. Preserve stable
sample IDs, disjoint splits, explicit epoch indices, and record incomplete runs
rather than filling missing trajectories with zero.

Multiple independent base seeds and separate evaluation are needed for final
claims. Current exploratory results are not a universal effectiveness guarantee.
ARIS is an optional local aid, not a package dependency.

## Checks

```bash
python -m pytest -q -m 'not integration'
python scripts/check_file_sizes.py --history
```

No publication license has been selected by the project owner.
