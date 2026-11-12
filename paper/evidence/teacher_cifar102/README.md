# CIFAR-10.2 teacher comparison

Source: private Kaggle output, version 368598938.

The compact CSV and JSON were recomputed from the three saved prediction
archives. The primary seed deltas and interval matched the Kaggle summary.
Accuracy means weight seeds equally. The paired intervals resample the
same test images across all fitted bases and do not quantify uncertainty
over new base training runs. The teacher comparison is secondary.
The archive checksums identify local NPZs repacked from the downloaded
predictions. The CSV contains replayed accuracy, ECE, NLL and Brier score.

Run `python3 paper/build_teacher_cifar102.py` from the repository root to
regenerate the accuracy table and verify its means and contrasts against
the replay. No training or test dataset download is required.

To verify the 18 prediction rows and five paired contrasts from the local
archives and recompute the error diagnostics:

```sh
PYTHONPATH=src python scripts/replay_teacher_cifar102.py outputs/s4_kaggle/teacher_evaluation_368598938
```

`diagnostics.csv` includes expected tie-aware AURC and both error AUROCs.
Own-error AUROC uses each predictor's errors. Fixed-base-error AUROC uses
the errors of its matching base for every predictor. Corrected and harmed
counts are relative to that base.
