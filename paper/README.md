# Manuscript workspace

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
and two CIFAR10 Worse base seeds (42/43). It does not merge historical ResNet34/Top-2/CFM
numbers with the new ResNet50 evaluation. Further replication is pending; no
unseen run is represented by a placeholder result.

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
