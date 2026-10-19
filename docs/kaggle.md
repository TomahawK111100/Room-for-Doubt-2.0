# Kaggle: stronger CIFAR-100N base

Use **GPU T4 x2** and enable **Internet** in Notebook settings. The launcher uses
both visible GPUs through PyTorch DataParallel. Batch size 128 is global (normally
64 per GPU). Actual GPU memory use and runtime must be measured on Kaggle; local
checks do not substitute for a two-T4 run.

Run in one Python cell:

```python
import os, subprocess, sys
from pathlib import Path

REPO = "https://github.com/TomahawK111100/Room-for-Doubt-2.0.git"
REVISION = "main"  # Replace with a published commit SHA for a pinned run.
ROOT = Path("/kaggle/working/room-for-doubt")
if not ROOT.exists():
    subprocess.run(["git", "clone", REPO, str(ROOT)], check=True)
subprocess.run(["git", "fetch", "origin", REVISION], cwd=ROOT, check=True)
subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=ROOT, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(ROOT)], check=True)
env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
           MPLCONFIGDIR="/kaggle/working/matplotlib")
subprocess.run([sys.executable, str(ROOT / "scripts/kaggle_run.py")],
               cwd=ROOT, env=env, check=True)
```

First run requires no uploaded data, private model or token. The repository is
public. The runner downloads official CIFAR images and noisy annotations, and
public torchvision ImageNet-1K weights. Run ID and settings are in
`configs/kaggle_base.yaml`.

## What runs

ResNet-50 keeps the ImageNet stem and uses 224x224 normalized images. Training
augmentation is CIFAR random crop and horizontal flip before resize. The loss is
cross entropy for epochs 0–4, then generalized cross entropy with q=0.7. SGD,
cosine learning rate, one base seed, 50 epochs. This fixed recipe is a pilot, not
a guarantee of reaching 70%.

The checkpoint selection criterion is always cross entropy on the **observed
noisy validation labels**, even when the training objective changes to GCE.
Clean training labels are never exposed by the training dataset wrapper.
The clean test is evaluated only after completing training and selecting the
checkpoint. Training logs show noisy-validation accuracy, not clean test accuracy.

This strengthens the base with external ImageNet supervision. Do not attribute
its gain to Room for Doubt. To measure the latter, retrain meta heads on the new
trajectories and compare policies on this same base. The historical 32x32
ResNet-34 loader cannot load this ResNet-50 checkpoint.

## Saved output and continuation

All outputs are in `/kaggle/working/rfd_outputs/cifar100n_resnet50_gce_seed42/`:

- `checkpoints/best.pt`: minimum noisy-validation CE model;
- `checkpoints/last.pt`: atomic epoch state, optimizer, scaler, RNG, best model,
  history and all collected margins;
- `artifacts/meta_targets.npz`: completed trajectories and reserved meta splits;
- `artifacts/split_ids.npz`, `val_predictions.npz`, `test_predictions.npz`;
- resolved config, metrics, metadata, environment, history, report and tables.

Use **Save Version with outputs** to preserve artifacts. If the process stops,
rerunning in the same session resumes automatically from the last complete epoch.
The runner pauses after an epoch when its 10.5-hour time budget has elapsed; it
never pretends a partial run is complete. In a new session, attach the previous
notebook output as input and run the same cell: the launcher finds that run's
`checkpoints/last.pt` and copies its directory to working storage. Retain the
original output directory structure. Only attach one matching prior run.

A restart may repeat the interrupted epoch. `last.pt` also contains the selected
best state, allowing recovery if writing `best.pt` and `last.pt` was interrupted.
Keep scientific configuration and code revision fixed across resumes. There is
no need to upload checkpoints to GitHub.

The exported margin is p(observed class) minus the largest competing probability,
recorded on each sample's augmented training forward before its update. Epochs
are explicit. Every training sample appears once per epoch, including the final
partial batch. These are training-time dynamics, not an epoch-end evaluation pass.

## References

- [Torchvision ResNet-50 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- [Generalized cross entropy, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html)
- [Kaggle P100 retirement and T4x2 availability](https://www.kaggle.com/product-announcements/735239)

## After the base training completes

The `post-kaggle` command prepares frozen image features, trains **new mean and
full-trajectory heads**, and compares the methods on a shared view cache. It
requires the complete base-run folder containing:

```text
config.yaml
metrics.json
history.csv
checkpoints/best.pt
artifacts/meta_targets.npz
artifacts/split_ids.npz
```

`last.pt` is needed to resume base training, but not for this downstream command.
A partial base run is rejected. The importer verifies epochs 0–49, all sample IDs,
official observed annotations, reserved splits, checkpoint configuration and
minimum noisy-validation CE in the complete history. Image size and class count
come from the source config; weights are restored without downloading another
base initialization.

From the repository root, with the completed folder available locally or as
Kaggle input:

```bash
python scripts/run.py post-kaggle \
  source_run=/path/to/cifar100n_resnet50_gce_seed42 \
  data_dir=/kaggle/working/rfd_data \
  output_dir=/kaggle/working/rfd_post \
  device=cuda
```

For local execution replace the three paths and use `device=auto`. The default
run requires at least 4 GiB free output storage. Image encoding and 20-view
inference are substantial; running this stage on Kaggle is preferable when local
storage is limited. `min_free_gb` is configurable.

The same command supports explicit stages by adding `stage=prepare`,
`stage=train`, or `stage=evaluate`. Use the same output directory, run ID and
scientific configuration across stages. Cached split inference resumes from a
flushed row boundary. Completed heads are verified and reused; partial heads
resume from their last checkpoint. A completed head cannot be substituted from
an older base: source digest, dataset, target mode, seed, partitions and frozen
encoder weights are checked. Keep partially trained head paths unchanged when
resuming; completed artifacts may be relocated with the whole pipeline directory.

The comparison contains original-view base, uniform TTA, MSP-weighted TTA,
mean-margin UW-TTA and full-trajectory UW-TTA. Mean/full use three meta seeds by
default. All methods share the **same views and base probabilities**. Meta
checkpoint selection uses reserved meta validation; downstream temperatures use
only noisy base validation, with uniform TTA included as a conservative option.
`artifacts/all_policies_before_test.json` is written before loading the test
images in this stage. Validation/test view scores and full sweeps are retained.

Primary comparison outputs:

- `tables/comparison.csv`: all method/seed rows;
- `tables/aggregate.csv`: means and meta-seed standard deviations;
- `artifacts/test_*.npz`: paired predictions and original-view badness;
- `config.yaml`, `metrics.json`, `run_metadata.json`, `report.md`.

Accuracy and ECE describe each final predictor; reported error AUROC/AUPRC always
use the same original-view base errors, so they compare uncertainty scores on a
common target. AURC is omitted because of known tie-order ambiguity. Pairwise
intervals are conditional on fitted models and validation choices. One base seed
and a previously inspected source test do not establish independent confirmatory
performance or robustness across base seeds.

The pipeline was tested offline with tiny synthetic images/models and full
source-artifact contracts. This verifies wiring, checkpoint/split checks and
selection ordering. It is not a measurement of the current Kaggle model and does
not replace a real GPU run.


## Unattended diagnostic battery after post-kaggle

`python scripts/build_kaggle_diagnostic_cell.py` creates the standalone launcher
`outputs/kaggle_diagnostic_cell.py`. Attach both the completed base notebook
output and completed post-kaggle output. Existing inputs remain read-only. The
launcher auto-detects these runs and reuses the features and 20-view caches.

The fixed battery fits nine small scalar heads: best-checkpoint observed-class
margin, last recorded margin, and the mean of the first five epochs, each with
seeds 42/43/44. Hidden layers, input features, splits and training budgets match
the existing scalar history-mean head. It also includes class-only history and
MSP/entropy/gap controls, fixed 1/5/10/20-view prefixes, reconstruction diagnostics,
NLL/Brier/ECE, paired intervals against the base/TTA/MSP, final train-noise
evaluation, and AMP/FP32 forward-pass reconciliation. Clean train labels are used
only for final diagnostic evaluation, after policies have been frozen.

This is explicitly exploratory because the previous test results have already
been inspected. Every predeclared comparison is exported; the script does not
select a winner by test metrics. A static original-image checkpoint target and
an augmented train-time target also differ in acquisition, which is disclosed.

Outputs: `/kaggle/working/rfd_diagnostics/diagnostic_resnet50_seed42/`. Read
`morning_summary.txt` and download `summary_bundle.zip` for compact analysis.
Configs, metrics, metadata, reports, model checkpoints and diagnostic arrays are
also preserved. Re-running in the same working directory resumes completed
controls and cached forward passes. The eight-hour limit is checked between
stages/heads, not a strict timeout during a head or forward pass. A paused or
failed run is labeled accordingly and exports its partial summary. Use Kaggle
Save Version with Save & Run All to preserve results without keeping the browser
open. No base model is retrained.


## Matched classifier-feature study

Build: `python scripts/build_kaggle_diagnostic_cell.py --study features`.
This produces `outputs/kaggle_feature_study_cell.py` and an equivalent one-cell
`.ipynb`. Use the same two inputs as the diagnostic run: completed base training
and the original completed post-kaggle output. The diagnostic output is optional
and not required. The launcher autodetects the base/post pair.

The fixed study crosses two frozen encoders with three scalar targets and three
meta seeds: external ImageNet ResNet18; selected, noisy-label-finetuned ResNet50;
mean first five epochs; mean all fifty epochs; original-image best-checkpoint
margin. Each feature space uses an unwhitened PCA fitted on meta-train only, with
512 outputs. Both branches therefore have the exact same learnable head size and
initialization for a given seed. These are newly fitted projected heads; previous
unprojected ResNet18 heads are not reused as a matched control.

The base-model features are captured immediately before its classification layer
in the same FP32 forward pass. Recomputed probabilities must agree with the
shared source cache within a 1e-3 maximum absolute tolerance; downstream methods
all use the original shared cached probabilities. Native and projected feature
caches are stored separately and checkpointed by completed input rows. Both
networks stay frozen. In an eventual ResNet50-feature inference adapter, its
features can be reused from the classifier forward pass.

View budgets are fixed to 1, 5 and 10. Both budget and temperature are selected
on noisy validation, with ties preferring fewer views and uniform weights. All
selections are saved before constructing the test dataset. Clean training labels
are then used only for a final diagnostic epoch-by-epoch analysis of real-history
noise detection on the reserved detector_eval partition (base-seen images).

Outputs: `/kaggle/working/rfd_feature_study/base_feature_study_seed42/`.
Download `summary_bundle.zip`. `test_validation_selected.csv` and `aggregate.csv`
are the primary summaries; `test_all_budgets.csv` reports every predeclared budget
without selecting a winner by test. Three meta seeds on one base do not establish
base-seed robustness. This is exploratory after earlier CIFAR test inspection.
The 8-hour inter-stage budget, partial reports and same-directory resume follow
the diagnostic workflow. Allow at least 4 GiB free working space; native and
projected features consume substantially more storage than the feature-only heads.


## Two-notebook batch (A and B)

Build A with `python scripts/build_kaggle_diagnostic_cell.py --study extended`;
build B with `--study campaign`. Both cells embed the exact unpublished source
updates over the public base commit; no GitHub authentication is required. Use
separate notebooks with the same completed base/post inputs, GPU T4 x2 and
Internet On. Save & Run All preserves outputs independently of the browser.

A uses the completed seed42 base. It fits 42 matched feature heads: two encoders,
three meta seeds and seven fixed targets (first2, first5, first10, first20, all50,
last5, static best margin). It reuses input view caches and keeps fitting and
policy selection separate from final label-correctness evaluation. Time budget:
1.5 hours checked between stages/heads. Output:
`rfd_feature_study/expanded_feature_study_seed42/summary_bundle.zip`.

B trains CIFAR100N seed43, CIFAR10N Worse seed42, then optionally CIFAR100N seed44.
The original 50-epoch base recipe is retained. A distinct `base_split_seed=42`
keeps the training/validation membership fixed while changing initialization and
minibatch/augmentation randomness. Old checkpoints lacking this field retain
their original behavior (`base_split_seed=seed`). The existing meta split seed is
also preserved. Each base is followed automatically by the standard 18-head
feature study, which can now prepare its own reference features and shared views
without needing an earlier post-kaggle run. This avoids training unused legacy
meta heads or creating another notebook between stages.

B reserves 7.5 session hours, admits a fresh job only when the remaining budget
is sufficient, and marks deferred/paused/failed jobs explicitly. An independent
job may proceed after another job fails; no failure is silently labeled complete.
It restores a previous full campaign output when attached as input and otherwise
resumes in place. Working storage must have at least 6 GiB free before admitting
a new job. Base checkpoints, feature caches, heads and compact reports remain
available. A 2/3-model ensemble baseline uses the fixed CIFAR100N seed prefixes
42/43 and 42/43/44 when available; it is never selected by test performance.

B output: `rfd_campaign/independent_runs/campaign_summary.zip`, including compact
base reports, all downstream summary bundles and the ensemble table. The A/B
limits total 9 session hours, excluding small startup/export overhead and any
inter-stage overrun. They are not a live reading of the account GPU quota. Actual
quota availability and GPU admission are controlled by Kaggle. Training in B uses
both T4s; feature/head stages currently use one.

Analysis must preserve all predefined targets and report core (first5/all50/static)
comparisons across independent bases. Do not choose a favorable window or seed by
the already inspected CIFAR test. CIFAR10N has only one new base seed in this batch;
CIFAR100N reaches three total base seeds only if the optional seed44 job completes.

## Follow-up: direct target contrasts and an augmentation-matched static control

These commands are prepared for completed feature-study Outputs. They do not
change an already-running notebook or retrain its base. A compact summary ZIP
alone is insufficient: preserve the full Outputs with heads, native/projected
features, shared caches and per-view score arrays.

### Direct paired comparisons (CPU; no head or base training)

```bash
python scripts/run.py contrast-feature-targets \
  study_run=/path/to/full/feature_study \
  post_run=/path/to/shared_cache_run \
  output_dir=outputs run_id=paired_c100_base42
```

`post_run` is the folder containing `cache/`; omit it when the complete shared
cache is at `study_run/shared/`. For the initial feature study with an external
post-Kaggle Input, attach that Input and specify its path explicitly.

The command compares Early-5 with Static and History-50 separately for each
encoder and head seed, keeping each method's saved validation-selected view count
and temperature. No policy is retuned. The accuracy interval is a 2,000-draw
paired test-example bootstrap. The AUROC difference uses a paired stratified
bootstrap with the same resampled examples for both scores and fixed counts of
original-view base errors/correct examples. These are conditional, exploratory
intervals without multiplicity correction; they do not estimate variation across
base training runs. IDs, selected predictions, copied source scores and frozen
policies are retained for audit. `summary_bundle.zip` contains compact tables and
run metadata, not all resumable artifacts.

### Static target under training augmentations (GPU recommended)

```bash
python scripts/run.py study-augmentation-control \
  source_run=/path/to/completed/base \
  study_run=/path/to/full/feature_study \
  post_run=/path/to/shared_cache_run \
  data_dir=/path/to/existing/cifar_data device=cuda \
  output_dir=outputs run_id=augmented_static_c100_base42
```

The new target averages five margins from the **fixed best-validation base
checkpoint**, evaluated on new deterministic draws from the actual base-training
transform: `RandomCrop(32,padding=4)` with constant padding, then random horizontal
flip, resize and ImageNet normalization. The ordinary TTA stream uses reflected
padding and is deliberately not used to construct this target. Original
training augmentation RNG draws were not logged; this matches their distribution,
not their exact realizations. Only observed labels enter target construction.

Only base ResNet50 features are used in this bounded control. Existing train-only
PCA, projected features, and Early-5/History-50/Static heads are reused read-only
or copied as small head artifacts. With three source head seeds, **nine old heads
are reused and only three augmented-static heads are fitted** with the original
training budget. The cache builder caps each expanded forward at 64 views and
supports resumption. Targets and new head checkpoints carry their augmentation
protocol and target-file digest. New policies freeze before test labels are
loaded; every reused method must reproduce its original validation-selected
policy. The completed run also produces direct paired target contrasts.

Outputs include resolved `config.yaml`, metrics, metadata, factual report,
training targets, head checkpoints, reconstruction/selection/contrast tables and
`summary_bundle.zip`. Keep full Outputs for recovery. A time-budget pause is an
incomplete run; resume with the same paths and scientific settings.

This control addresses augmentation acquisition, while differences in changing
online weights, checkpoint timing and the CE/GCE schedule remain. No improvement
is assumed in advance. Local synthetic tests verify the execution contract;
CIFAR results require the full saved Kaggle artifacts.

## Matched inference-cost measurements

```bash
python scripts/run.py benchmark-runtime \
  source_run=/path/to/completed/base study_run=/path/to/full/feature_study \
  data_dir=/path/to/existing/cifar_data device=cuda \
  output_dir=outputs run_id=cost_c100_base42
```

This command measures final-probability consensus for the base, uniform TTA and
Early-5 with fixed head seed42. View counts1/5/10 use saved validation policies,
including each count's saved temperature. It does not choose a policy by timing
or accuracy. Stable noisy-validation image IDs supply deterministic view prefixes;
no clean train/test labels are accessed. The benchmark does not train models.

The FP32 reference implementation shares the selected base's penultimate features
with PCA and the scalar head on the same device. Its weighted probabilities are
checked against the existing cached NumPy weighting formula in offline tests.
It measures this fused inference path, not the wall time of the research cache
extraction and offline scoring pipeline. Single-view / uniform policies skip an
unnecessary head; this is not the cost of separately returning an error-ranking
score for every original image.

Two scopes are saved: `model_only` uses prepared device tensors;
`with_preprocessing` includes CPU view generation, normalization and host-to-device
transfer. Disk/model loading is outside both. Batch sizes1/4, warmup5 and30 timed
repetitions are recorded; expanded batches cannot exceed64 views. CUDA and MPS
are synchronized around each measurement. Report batch median/p95 separately
from amortized milliseconds per image; batch p95 is not single-image p95.
CUDA rows report PyTorch allocated resident/peak memory, not total GPU usage.
CPU/MPS peak memory is left unavailable rather than approximated.

Use an otherwise idle device, record hardware and CPU threads, and keep Mac and
Kaggle measurements separate. These comparisons do not establish an advantage
over a two-base ensemble, which needs its own matched timing experiment. Each
independent timing run requires a fresh run ID. Outputs contain resolved config,
metrics, hardware/precision metadata, a factual report, a compact timing CSV and
raw timing repetitions. Real GPU measurements remain pending.

## Follow-ups on the two completed replications (19 October)

Attach the full Outputs of both replication notebooks as Notebook Inputs:
CIFAR-100N base44 and CIFAR-10N Worse base43. Notebook names can be ambiguous:
the launcher identifies these exact models by checkpoint SHA256 and checks for
complete feature arrays, heads, shared caches and per-example scores. A compact
campaign summary is not sufficient. Raw CIFAR images and official annotations
are linked read-only from the attached Outputs; large feature caches are not copied.

With a GPU and Internet enabled, from a pinned checkout, run:

```bash
python scripts/kaggle_followups.py
```

The bounded default batch runs direct paired comparisons for both bases, then
matched runtime measurements on both, then an augmentation-matched static
control on both. Each stage runs in a separate process to release allocations.
No base is retrained; only three new static heads per base are fitted. Existing
heads/PCA/features are reused. All methods and intervals remain exploratory.

Run this batch alone on its GPU during timings. T4 x2 is compatible, but these
stages use one GPU; they do not require two GPUs for comparable base training.
The overall six-hour budget is checked between stages; the augmentation control
has an additional two-hour inter-stage budget per base. These are admission/check
budgets, not a hard kill during an active fit or bootstrap. No measured wall-time
estimate is available yet for the combined batch.

Results are under `/kaggle/working/rfd_followups/`. Download
`followup_summary.zip` for compact analysis and preserve the full Output for
resumption. A non-complete stage is recorded explicitly; the notebook reports
an incomplete batch rather than silently succeeding. Re-running in the same
working directory skips completed stages after verifying the saved batch
contract. To recover in a fresh notebook, preserve the prior full follow-up
Output and restore it to the same writable output path before running; attaching
a compact ZIP does not restore the new heads or target arrays.
