# CIFAR-10.1 v6 evaluation

Source: private Kaggle output, version 368534180.
The prespecified protocol is `docs/planning/confirmation_protocol_v1.md`.

The compact CSV and contrasts were recomputed from the three saved prediction
arrays. All 12 accuracy values, the primary mean and between-base spread, the
teacher gap, and both published paired intervals match the saved output.
Latency comes from the run summary. Prediction replay does not repeat timing
or regenerate network outputs. Direct-distillation contrasts are secondary.
`own_error_auroc` uses each predictor's own errors. `fixed_base_error_auroc` uses the
corresponding base model's error indicator for every predictor.
`corrected` and `harmed`
count changes in correctness relative to the corresponding base.

With the local prediction archives and observed summary present, run:

```sh
PYTHONPATH=src python scripts/replay_confirmation_s4.py outputs/s4_kaggle/evaluation_368534180
```
