# Mini-WebVision transfer pilot, version20261022_v1

Status (25 October 2026): bases42/43/44 completed all50epochs and all scalar,
paired and augmentation-control stages. Base42 resumed after40epochs;43/44 after43.
Three nested heads per base; aggregate bases equally after head means. Runtime
workers0 did not prevent the replication memory interruptions; checkpoint recovery
completed the remaining work. Saved history prefixes match the interrupted runs.
Scientific settings and fixed split42 agree across all bases. Compact evidence
checked; raw-prediction replay remains pending. Data and original results preserved
in `outputs/miniwebvision_replications_import_20261025/` and the earlier seed42 import.
Configuration: `configs/mini_webvision_pilot.yaml`; actual per-run config and runtime
amendments are preserved with source outputs.
Animal-10N is closed for the bounded three-base stage, including recorded-cache
replay. Its small advantage over selected TTA and tie with History-50 motivate
retaining all controls here, rather than promoting only the favorable target.

## Dataset and provenance

Use WebVision **1.0**, Google training images with original class IDs0..49,
selected by `info/train_filelist_google.txt` in original file-list order. Use
all corresponding official `info/val_filelist.txt` entries (50 images per class,
2500 total) as the final evaluation partition. It is the official validation
partition, used as a held-out benchmark test here; it never selects our models.
Do not substitute WebVision2.0, ImageNet, a folder-sorted remap or an arbitrary
61k/66k third-party subset. No missing listed image is silently removed.

Official metadata-only preflight confirmed65944 training rows and2500 evaluation
rows before fitting. Exact original-list hashes are preserved locally under
`data/miniwebvision_metadata/`; selected-image byte hashes are recorded during
real image preflight. The adapter rejects truncated65944-row training membership. Image membership is frozen by
the supplied original lists, not by model predictions or observed test accuracy.
The adapter lazily decodes RGB and resizes each image to a256x256 square using
PIL bilinear interpolation. Supplied JPEG bytes from the official **resized**
small archives remain unchanged (not full-resolution originals); decoded pixels
therefore differ from the original variable-resolution image. IDs are offsets
in the selected original lists, with separate train/evaluation namespaces.

References:
- Dataset: https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html
- Distribution/terms: https://data.vision.ee.ethz.ch/cvl/webvision/download.html
- Original-label reader: https://github.com/LiJunnan1992/DivideMix/blob/master/dataloader_webvision.py

The dataset terms prohibit redistribution and assign responsibility to the user.
Downloading image archives requires the owner's explicit acceptance. Do not
publish/reupload images as a Kaggle Dataset or save them in notebook Outputs.
For Kaggle, use ephemeral `/kaggle/temp/` storage for downloaded images. Public
repository/notebook artifacts contain code, configs and numerical summaries only.

## Supervision and selection

Reserve10% of selected noisy Google training rows with split seed42 for noisy
validation; train on the remaining90%. Noisy validation chooses the minimum-CE
base checkpoint and each method's view budget/temperature. Scalar heads use
their separate reserved meta-validation partition and minimum regression MSE;
this partition comes only from base-training rows, not from clean evaluation.
No clean
training-label vector exists in this interface. Official clean validation labels
are used only for final accuracy/calibration/base-error ranking; parsing the
official membership/class list for defining the first50-class benchmark is not
model fitting or policy selection. Meta train/head-validation/calibration/test
reservations retain the existing label-free split seed20261015 and proportions.

## Fixed recipe and comparisons

ImageNet-initialized ResNet50, input224px,50epochs, SGD lr.01/momentum.9/weight
decay.0005;5epoch CE warmup then GCE q.7; batch128 on two GPUs. Training uses
RandomCrop256 with32px constant padding and horizontal flip, then resize224 and
ImageNet normalization. Shared inference views use the unchanged relative
one-eighth reflection-padding crop recipe at256px, with unmodified first view.
This is a declared adaptation of the project recipe, not the published DivideMix
training recipe (architecture, resolution and selection differ).

Compare base, selected uniform TTA, MSP/gap/entropy weighting, learned Early-5,
History-50 and selected-checkpoint static targets on frozen ResNet18 and base
features;25epoch scalar heads, three head seeds42/43/44, train-only PCA512.
Include base-feature augmentation-matched static control and direct paired
contrasts. Retain view grid1/5/10 and the existing temperature grid. No window,
backbone, target, data subset or loss tuning based on the new test results.

Primary: Early-5 on base features versus validation-selected uniform TTA.
Report per-head and within-base means, paired accuracy intervals, original-view
base-error AUROC/AUPRC/AURC and ECE/NLL/Brier, all negative findings included.
Bootstrap2000draws with seed20261019, conditional on the fitted model/split.
ImageNet initialization overlaps the WebVision category vocabulary; no strict
unseen-category or pretraining-free transfer claim is made. No SOTA comparison
is inferred from a different validation-selection protocol.

## Run order and compute

1. Offline synthetic adapter/split/base-resume checks; same-family code review.
2. Real data membership, full byte-identity manifest and two-GPU AMP sanity.
3. One independent base42 plus all heads/controls in one bounded launcher.
4. Inspect completion, memory/disk/time and fixed-protocol outcomes.
5. Replications43/44 are reserved as a separate compute allocation after the
   technical pilot, independent of whether its accuracy gain is positive.

Before fitting,23October quota amendment: Kaggle reported6 GPU-hours remaining.
Admission budget4.5h after data preparation; base3.5h, study3h, augmentation2h, each adjusted to remaining
session time. These are cooperative stage/epoch stop/resume checks, not hard
process timeouts or measured runtime predictions; an operation can overrun them.
No automatic third/fourth notebook, extra tuning or paid compute. Two L40 GPUs
can run the same recipe if available; no speedup factor is claimed before a run.
Full checkpoints/features remain in Outputs, images stay ephemeral. Compact
summaries are returned even on a blocked or incomplete stage.

## Reproduction

With an already permitted original-layout dataset:

```sh
PYTHONPATH=src python scripts/mini_webvision_pilot.py --input-root /path/to/release --output outputs/miniwebvision_seed42 --inspect-only
PYTHONPATH=src python scripts/mini_webvision_pilot.py --input-root /path/to/release --output outputs/miniwebvision_seed42
```

No training should be marked complete from an input preflight or synthetic test.
