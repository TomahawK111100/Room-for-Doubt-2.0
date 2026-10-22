# Frozen transfer protocol: Animal-10N pilot

Version: `animal10n_transfer_20261022_v1`, fixed on 22 October 2026 before
any Animal-10N model fitting or test-result inspection in this project.
CIFAR reference evidence: commit `552c0ac`. This is a prospective local protocol
freeze for a pilot, not an external preregistration or a completed experiment.
The machine-readable source is [configs/animal10n_pilot.yaml](../configs/animal10n_pilot.yaml).
Every run copies that file and records its SHA256, the implementation commit,
resolved stage configurations, input-file hashes, devices and checkpoint identity.

## Question and fixed comparisons

Does the existing Early-5 estimator retain an advantage on another natural
human-label-noise dataset, without tuning its target/window, architecture,
training recipe or policy grid on the new test set?

Primary estimator: selected-base ResNet50 features, train-only PCA512,
class-conditioned scalar MLP256 with class embedding32, predicting the mean
observed-class probability margin in the first five online training epochs.
The head fits observed classes and inference conditions on each view's predicted
class. Its badness score is a heuristic, not a calibrated noise probability.

Primary endpoint: test accuracy difference from noisy-validation-selected uniform
TTA. Key target control: accuracy difference from augmented-static using five new
training-distribution draws at the same fixed checkpoint. Report both whether or
not they favor Early-5. Also retain original-view base, MSP/gap/entropy weighting,
static checkpoint-margin regression, History-50 mean, and the independent frozen
ResNet18 feature branch. These are the same standard 18-head study and three new
augmented-static heads used in CIFAR, with three head seeds per target/encoder.

Secondary endpoints: original-view base-error AUROC/AUPRC; aggregated ECE, NLL
and Brier score; per-base head variation; accuracy/AUROC paired differences versus
static and History-50. No train-label-noise AUROC is computed: complete clean
training annotations are unavailable. A full-history mean is only one temporal
summary; this comparison does not reject all sequence models.

## Data and identity

The [official KAIST description](https://dm.kaist.ac.kr/datasets/animal-10n/)
and [author implementation](https://github.com/kaist-dmlab/SELFIE)
report 50,000 training and 5,000 test RGB64x64 images, ten classes, and an
estimated natural noise rate around8%. That estimate is descriptive, not a
known clean/noisy flag for individual training images and not an input parameter.
The authors construct the benchmark test as a cleaner partition by matching
annotations to image search keywords; this is not independently reannotation-
verified ground truth for every possible claim. Cite Song, Kim and Lee, SELFIE,
ICML2019 when reporting Animal-10N results.

The official download currently points to a request form. This adapter supports
the documented binary release: `data_batch_*.bin` and `test_batch.bin` together
in one directory. Each record is big-endian int32 source ID, int32 label,
and channel-first uint8 image bytes (3x64x64). The byte order follows the
[author byte reader](https://github.com/kaist-dmlab/SELFIE/blob/master/SELFIE/reader/batch_patcher.py). It does not guess labels from an
arbitrary mirror or silently download substitute data. The official source
publishes no reference checksums on the inspected page: recorded SHA256 values
establish the identity of supplied files, not independently verified provenance.
Obtain the official archive and retain its source information with the run.

Binary records are mapped read-only. Internal sample IDs follow naturally sorted
batch filenames and row offsets; original record IDs are checked for uniqueness
within each official partition. Train/test IDs have separate namespaces.
Preflight checks record sizes/counts, observed training-label range and class
counts, and hashes opaque train/test files. It does not evaluate a model or use
test-label values for selection. Missing, ambiguous or malformed data stop before
GPU work. No synthetic noise, label cleanup, or sample exclusion is introduced.

A fixed seed42 shuffle splits the official training pool into45,000 observed-label
training and5,000 observed-label validation images. Official test5,000 is kept
separate. Meta partitions use the existing seed20261015 reservation inside the
base-training subset. PCA fits only the reserved meta-training image features.
All subsets and row alignment are saved.

## Base and geometry

- ImageNet-1K V2 ResNet50; input224x224; all weights finetuned.
- Fifty epochs; SGD lr0.01, momentum0.9, weight decay0.0005; epoch cosine schedule.
- Cross entropy epochs1–5, then GCE q0.7. Record every observed-label margin at
  its augmented minibatch forward before the parameter update, with no missing rows.
- Global batch128, two visible CUDA GPUs, DataParallel and AMP, matching the CIFAR
  training topology. Select minimum observed-validation cross entropy, never
  the last epoch or test accuracy. Save best and resumable last checkpoints.
- Animal native training crop64x64 with constant padding8 and horizontal flip,
  then resize224/normalize. This preserves the CIFAR relative crop/padding rule
  (32 with padding4) rather than throwing away three quarters of each Animal image.
- Inference has the unmodified original view first, then deterministic reflected-
  padding crops/flips at native64 with padding8; all methods share view prefixes.
  The original CIFAR32 view stream is byte-for-byte preserved by a regression test.
- Augmented-static uses the actual training transform at the selected checkpoint,
  five draws, seed20261019. It does not replay old RNG draws or evolving weights.

Two L40s may replace two T4s without changing global/per-device batch sizes,
precision policy or scientific settings. Record the exact GPU/driver/software
versions; numerical equivalence across hardware is not assumed. One GPU changes
BatchNorm's per-device minibatch, so it is outside this frozen base topology.
A single-GPU variant would require a separately named protocol before fitting.

## Heads, policies and statistics

Use three head seeds42/43/44,25 Adam epochs at lr0.001, batch256; same meta-validation
MSE checkpoint rule, PCA seed20261016 and512 projected dimensions. The frozen
ResNet18 comparison remains a secondary representation control.

Select each method's views from[1,5,10] and temperature from
[0.02,0.05,0.1,0.25,0.5,1,2,5,10,1000000] plus exact uniform on observed-label
validation. Existing tie rules prefer fewer views and exact uniform. Freeze all
policies before loading test labels in the corresponding evaluation stage.
Base training performs its final AMP test evaluation after checkpoint selection;
downstream comparisons consistently use FP32 and do not mix the two accuracies.
All later targets/hyperparameters are already fixed, despite that earlier base
metric being present in the saved Output.

The pilot is base seed42. Planned independent replications are43/44 with the same
split and settings; add them after technical completion/data checks regardless
of the sign of the pilot gain. Do not replace a negative seed or extend epochs,
change the early window, select temperatures from test, or drop a weak metric.
Any later methodological change receives a new protocol version and an explicit
exploratory label. No predetermined accuracy threshold defines pilot success.

Average heads within a base, then bases equally. Three heads on one base do not
constitute three independent base replications. Report all per-head estimates
and nominal95% paired test-example intervals (2,000 draws, seed20261019;
stratified paired draws for AUROC differences). They condition on fitted models
and selected policies, lack multiplicity correction and do not measure
base-training uncertainty. One pilot cannot establish general effectiveness.

## Execution and output

Audit on CPU, before reserving a training session:

```sh
python scripts/animal10n_pilot.py --input-root /path/to/extracted/parent --output outputs/animal10n_seed42 --inspect-only
```

Then run the same command without `--inspect-only` on two visible CUDA GPUs.
For a server with more GPUs, expose exactly two with `CUDA_VISIBLE_DEVICES=0,1`.
`--base-only` is a separate output scope and must use its own output directory.
For later replications use `--seed 43`/`--seed 44` and a separate output directory.
No old Kaggle notebooks, CIFAR weights, or historical caches are required.

One launcher runs base → matched feature study → direct paired contrasts →
augmented-static control in separate processes. It saves progress, logs,
resolved configs, metrics, run metadata, reports, compact CSVs and
`animal10n_summary.zip` even on failure. Preserve the **full Output** too:
summary ZIPs do not contain weights, trajectories, raw predictions or cached
features. A paused training stage resumes from last.pt in the same directory;
a compact-only import cannot resume training. Admission budget is8h; base4.5h,
study3h and control2h caps are reduced to remaining wall time. These are pause
limits, not promises that all jobs finish within them.

Planning estimate, **not a measured Animal-10N benchmark**, excluding download:
2xT4 about2.5–4h for the base and3.5–5.5h for a complete pilot;
2xL40 about0.7–1.5h for the base and1.2–2.5h for the complete pilot.
The prior CIFAR-10N seed44 full run took3.57h, of which2.63h was base training.
Data decoding, CPU transforms, PCA, disk and single-GPU feature extraction limit
end-to-end speedup; GPU specification ratios do not predict wall time directly.
Refine the estimate from actual epoch times after the first three epochs.

## Current validation boundary

The adapter, observed-only CPU training/resume,64px shared views, matched-head
study and augmented-static path are tested with tiny synthetic binary data and
networks. Missing/corrupt inputs fail explicitly. No Animal-10N archive is
currently available locally; no real Animal training or accuracy is claimed.
The first real run still needs the official binary data and CUDA environment.


## ARIS preparation record

The local ARIS `experiment-plan` and `experiment-bridge` workflows record the
claim map, run order, falsifiers, compute estimates and honest execution states
in `refine-logs/EXPERIMENT_PLAN.md`, `EXPERIMENT_TRACKER.md`,
`EXPERIMENT_CODE_REVIEW.md`, and `EXPERIMENT_RESULTS.md`. The research contract
is `idea-stage/docs/research_contract.md`; compute specifications and ledger
are local under `.aris/compute/`. These working files remain outside public Git.

A fresh Codex reviewer reads the primary source files for the predeployment
code review. Its same-family verdict is provisional; it does not constitute
cross-model experiment-integrity or paper-claim audit acceptance. Final result
and claim audits require actual run outputs. Do not mark the real-data/GPU
milestones complete based on synthetic tests or a prepared notebook.


The primary selected-uniform reference is explicit in new result columns
`gain_pp_vs_tta_selected`, `ci_low_pp_vs_tta_selected`, and
`ci_high_pp_vs_tta_selected`; `selected_tta_reference_views` records its frozen
budget. Existing `*_vs_tta` columns retain their historical **same-view-budget**
meaning and are not substituted for the primary comparison. Animal study
intervals explicitly use bootstrap seed20261019; old CIFAR runs retain their
historical default. Root `test_evaluated` records persisted use even if a later
stage fails or pauses, including when completed results are reused.
