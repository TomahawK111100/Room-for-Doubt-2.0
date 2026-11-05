# Manuscript workspace

[Download the current working manuscript PDF](revision/Room_for_Doubt_manuscript.pdf) (5 November 2026).

`revision/main.tex` is the active, anonymous English working draft. The ICOMP
layout is retained for editing; it is not a current submission or an accepted
paper. The header identifies it as a working draft. Conference choice remains open.

- `revision/sections/`: method, protocol, results, discussion and appendix.
- `evidence/`: small CSV snapshots underlying the current tables; base seeds are
  kept separate and meta seeds are averaged only within each base.
- `evidence/sources.csv`: paths to the complete local imported experiment reports.
- `build_tables.py`: validates selected policies and regenerates manuscript tables.
- `build_figures.py`: checks snapshot hashes and generates vector PDF / PNG figures.
- `revision/figures/`: accuracy gains, exploratory history windows and calibration.
- `archive/icomp2026/`: original user-supplied files, preserved unchanged and
  excluded from Git, including the original ZIP and private OpenReview PDF.
- `notes/`: local review mapping and writing notes, excluded from Git.

The command-line build requires TeX Live with the standard `helvetic`, `times`,
`courier`, and `booktabs` packages. The original style files are unchanged.

From the repository root:

```sh
python paper/build_tables.py
python paper/build_figures.py
python paper/build_followups.py
cd paper/revision
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively open `revision/` in the Codex LaTeX editor. The tables can be
regenerated from the compact tracked evidence without downloading model weights.
A full training reproduction still requires the training configs, checkpoints,
and view/prediction caches documented in `docs/kaggle.md`.

## Current scope

The draft covers Early-5 scalar regression on frozen base features, uniform and
static-score TTA, matched static/history scalar targets, three CIFAR100 base seeds (42/43/44)
and three CIFAR10 Worse base seeds (42/43/44). It does not merge historical ResNet34/Top-2/CFM
numbers with the new ResNet50 evaluation. The augmentation controls cover all six bases; the CIFAR10 base42 study was
rebuilt on its verified original checkpoint with the earlier policies reproduced.

Nominal intervals are conditional paired test-example bootstrap intervals, not
uncertainty across base training runs. Current test evidence is exploratory.
Raw per-example predictions and full checkpoints are not included here.

## Figure interpretation

The accuracy figure shows differences from each corresponding base. Individual
head dots and sample standard deviations quantify head initialization, not base
training variance. Every policy is selected on noisy validation; view budgets may
differ between methods. The window figure is exploratory on C100 base seed 42;
its categorical targets are not equally spaced epoch times. Static-target lines
use the same feature source. Calibration figures show changes in ECE and NLL
from the same base, averaged over heads. Positive values mean deterioration.

Small public CSVs are sufficient to regenerate these figures. Full run configs,
checkpoints and caches are required for training-level reproduction, as explained
in the Kaggle guide. Private reviews, original submission files and writing notes
are excluded from Git.

Public CSV snapshots use LF line endings; numerical values are retained from
the imported outputs. `evidence/file_manifest.csv` records their public-file hashes.

## First follow-up batch (21 October import)

The evidence/followups folder preserves direct comparisons, augmentation-control selection/results,
and full compact T4 timing tables for C100 base44 and C10 Worse base43. Source paths,
base digests and run commit are in its sources.csv; public bytes are covered by the
main evidence hash manifest. The build_followups.py script regenerates four tables and the
paired augmentation-control figure. Intervals condition on each fitted base/head;
that first batch covers one base per dataset; later controls are recorded below. Runtime is one fixed-order session,
not an independently replicated performance guarantee. Full checkpoints, raw predictions
and raw timing repetitions remain in Kaggle Outputs, outside the compact evidence.

## Additional CIFAR-100N controls (21 October)

Both added base42/base43 paired and augmentation stages completed; the combined
campaign was partial at that import because C10 base42 lacked inputs; this is resolved in the 22 October import below. The new evidence consists of ten stage tables plus two cache-recovery CSVs,
preserved under evidence/followups/. Full imports remain local
under outputs/kaggle_available_controls_import_20261021/. The review verifies
42 contrast point estimates, existing selected policies/accuracies, and 93
reported native-cache recovery accuracy checks. It cannot recompute bootstrap
intervals or inspect actual weights from the compact export.

At the 21 October import, the augmentation-control figure and tables covered C100 bases42/43/44 and
C10 Worse base43. Seven of nine C100 head intervals exclude zero; two on base44
include zero. Mean Early-5-minus-augmented-static gain across the three C100
bases is 0.76 percentage points (sample SD 0.24), with heads averaged within base.
Cache recovery uses saved native FP32 arrays and explicitly does not claim
bit-identical original shared probabilities. The single-session runtime evidence
is unchanged and still covers only C10044 and C10 43.

## Completed CIFAR-10N closeout (22 October)

The control42 and new base44 tasks both completed at source commit 5df2c8f.
Full compact imports are preserved locally under
outputs/kaggle_cifar10_closeout_import_20261022/{control42,seed44}; the review
under outputs/cifar10_closeout_review_20261022 records source hashes and checks
42 direct contrast point estimates and 76 frozen policy entries. The new public
c10_s44 snapshot and follow-up tables extend all main figures and target controls
to three independent bases per dataset. Control42 rebuilds 18 heads on its verified
original base, reproducing all 23 prior policies/accuracies; the original main
snapshot is retained. Rebuilding is not another independent base run.

Nine of nine C10 augmented-static conditional accuracy intervals are positive;
seven of nine C100 intervals are positive. All 18 original-view AUROC difference
intervals are positive. Raw predictions and the new base44 checkpoint are absent
from the compact export; verification checks metadata and arithmetic, not the
raw bootstrap or the new checkpoint bytes. Runtime measurements are unchanged.

## Prospective Animal-10N pilot (22 October 2026)

One base (seed42) completed the frozen 50-epoch recipe, standard18-head study, paired comparisons and three augmented-static heads in2.70h on two T4 GPUs. Compact imported evidence is under `evidence/animal10n_s42/`; `python paper/build_animal_pilot.py` regenerates its table and figure. The section `revision/sections/animal10n_pilot.tex` preserves the single-base limitation, mixed conditional intervals versus augmentation/static-history controls, worsened ECE and mirror provenance. Full original import and byte manifest are local in `outputs/animal10n_seed42_import_20261022/`. Seeds43/44 are not completed.

## Animal-10N replications (22 October 2026)

All three preplanned bases42/43/44 now complete. `evidence/animal10n_replications/` records per-base means, all selected heads, paired contrasts, measured wall times and summary-audit status. `python paper/build_animal_replications.py` rebuilds the three-base table and figure. Base42 snapshots and `build_animal_pilot.py` remain preserved as historical pilot evidence; the active manuscript uses the three-base section/table/figure. Mean Early-5 gain over TTA is0.2489pp; accuracy is virtually tied with History-50 and error AUROC is below MSP on bases43/44. See local `outputs/animal10n_replications_import_20261022/` for source imports and byte manifest. No raw predictions/checkpoints are supplied in the compact summaries.


## Animal recorded-cache replay (22 October)

Kaggle version361233698 completed CPU replay for bases42/43/44. Each base passed
21 scalar-head replays,40 selected-row checks and paired metric/interval checks.
Elapsed audit time199.34s excludes environment setup. Exact source snapshots and
reports are preserved locally in outputs/animal10n_raw_audit_import_20261022/.
Base forward checks cover eight fixed validation images per base; full feature
extraction/PCA were not rerun. This is deterministic replay, not cross-family
semantic audit acceptance.

## Mini-WebVision pilot, 24 October 2026

One completed base42 with three nested head seeds. Compact evidence is in
`evidence/miniwebvision_s42/`; regenerate its table and figure with
`python paper/build_miniwebvision_pilot.py` from the repository root.
The section records the memory interruption/resume, fixed noisy-label selection,
AMP/FP32 difference, conditional intervals and pending independent replications.
No raw-prediction replay has been performed for this dataset.

## Mini-WebVision replications, 25 October 2026

The manuscript now uses all three completed bases42/43/44. The previous pilot
files remain historical. Rebuild the updated figure/table/compact aggregate with
`python paper/build_miniwebvision_replications.py`. Evidence:
`evidence/miniwebvision_replications/`; source imports:
`outputs/miniwebvision_replications_import_20261025/` and the earlier seed42 import.
Means use equal base weights after nested-head averaging. Raw prediction replay
remains pending; conditional intervals are not between-base uncertainty.


## Separate temporal compression pilot (27 October2026 import)

`revision/sections/temporal_residual_validation.tex` records the separate pilot's
validation selection and subsequent frozen clean-test evaluation. One CIFAR100N
base42 completed 50 epochs and one residual head42 completed 25 epochs.
Test accuracy: base69.79%, student71.83%, five-snapshot mean73.15%; student gain
+2.04pp [1.46,2.65] conditional nominal paired interval. ECE worsens; NLL/Brier
improve. Raw replay passes46 checks. Evidence: evidence/temporal_residual_test/;
preserved import: outputs/temporal_test_import_20261027/. Independent-base
replication and matched controls remain outstanding. Scalar results are unchanged.

## Temporal follow-up imports

Independent base seeds43/44 imported with155 raw-replay checks. Three-base mean student gain+2.01pp (sample SD0.52pp). Fixed epoch1 controls only on seeds43/44: temporal minus control+0.32pp and+0.93pp; first nominal conditional interval includes zero. ECE deteriorates on all three bases. See evidence/temporal_residual_replications/ and outputs/temporal_replications_import_20261027/. Full temporal attribution and equal-compute ensemble advantage remain unproven.

Selected-single exploratory control added: epochs5/3 chosen on noisy validation for43/44. Temporal head advantage0.48/0.30pp (second conditional interval includes zero). Two-pass interpolation slightly exceeds temporal heads; no latency measurement.81 raw checks. Evidence: evidence/temporal_selected_control/. Source: outputs/temporal_selected_control_import_20261027/. Full temporal attribution is unsupported.


## S2 post-hoc controls, 28 October 2026

A/B imports and503 raw replay checks: outputs/posthoc_controls_import_20261028_verified/.
Rebuild the compact table/figure with `python paper/build_posthoc_controls.py`.
Three-base mean accuracies: base69.60%, Early-5 residual71.61%, selected single71.24%,
Early-2 71.55%, direct soft-target71.53%, independent-two student71.87%.
Residual-specific novelty is unsupported; independent teachers cost extra training.
Runtime and the missing full seed42 epoch-selection grid remain pending. All tests
are exploratory reuse. Supplemental AURC uses expected ordering within exact ties;
historical metric definitions remain unchanged. Local replay does not refit heads.


## S2 closeout update, 28 October 2026

Original seed42 selection_grid.csv recovered from full Kaggle A Output version362957682.
All20 rows ranked by the frozen rule; epoch3/alpha0.5 exactly matches the policy.
Source hash and scope: outputs/posthoc_controls_import_20261028_verified/recovered_original/verification.json.
Runtime-only notebook submitted: https://www.kaggle.com/code/tomahawkkz/cifar100n-s2-runtime.
Active Events showed Running once; no log polling. No new fitting or test evaluation.
Actual runtime results remain pending; S2 quality/cost gate stays open.

## PDF snapshot: 5 November 2026

The compiled working draft includes all four scalar-study datasets and the three-base
CIFAR-100N posterior-compression controls. It is not an accepted paper. Runtime
measurements for the latter branch have not been imported or verified; the draft
does not claim a measured deployment speedup. The private CV is not published.
