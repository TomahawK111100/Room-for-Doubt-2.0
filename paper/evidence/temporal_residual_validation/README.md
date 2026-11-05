# Validation-only residual pilot

One new CIFAR100N base42 and one head42. Base50epochs/head25epochs. All accuracies
refer to5,000 noisy validation labels, not clean test. Source snapshot archive:
2523c77e20ce00178e28eced2f1b9910b2e8e7c808c4f2ac1fd34cb1614acd9c.

Full compact originals and checks: outputs/temporal_pilot_import_20261027/.
Rebuild import/plot with scripts/analyze_temporal_summary.py --source <summary-folder>.
Weights and per-image probabilities are absent from this compact export.
No raw replay or significance interval is claimed.
