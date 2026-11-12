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

## External CIFAR evaluations and manuscript

From the repository root, run `python paper/build_evidence.py` to generate
manuscript tables and figures from compact evidence without training or data
downloads. See [PDF build instructions](../paper/README.md). Some historical
appendix values are transcribed from recorded run summaries.

For checkpoint-level reproduction, follow the frozen protocols for
[CIFAR-10.1](planning/confirmation_protocol_v1.md) and
[CIFAR-10.2](planning/temporal_teacher_protocol_v1.md).
`python scripts/build_confirmation_s4.py --help` lists the notebook builders.
The separate fit and evaluation entry points are
`python scripts/confirmation_s4.py --help` and
`python scripts/temporal_teacher_fit.py --help`. These stages require the
datasets and original checkpoint outputs described in the protocols.

| Stage | Kaggle output version IDs |
| --- | --- |
| CIFAR-10N Worse base and frozen heads (seeds 42, 43, 44) | 368169294, 368169672, 368364704 |
| CIFAR-10.1 v6 evaluation and T4 runtime | 368534180 |
| CIFAR-10.2 matched controls fit on noisy data | 368561680 |
| CIFAR-10.2 evaluation | 368598938 |

With the local prediction archives and matching run summaries, rerun the
metric calculations and paired comparisons:

```bash
PYTHONPATH=src python scripts/replay_confirmation_s4.py outputs/s4_kaggle/evaluation_368534180
PYTHONPATH=src python scripts/replay_teacher_cifar102.py outputs/s4_kaggle/teacher_evaluation_368598938
```

These replays use saved probabilities. They do not regenerate model outputs
or benchmark latency. Compact evidence and private checkpoint provenance have
different access requirements.
