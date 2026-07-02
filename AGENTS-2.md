# AGENTS.md

Guidelines for Codex, AI agents, and student contributors.

For the scientific idea, methodology, and current project scope, read `README.md`.

---

## 1. General Rule

Do not duplicate or redesign the methodology here.

Use this file only for stable working rules:

* reproducibility;
* clean repository structure;
* correct dataset usage;
* experiment outputs;
* compact reports;
* basic validation.

When in doubt, follow `README.md`.

---

## 2. Preferred Stack

Use the default research stack unless changed by the project owner:

* Python;
* PyTorch;
* PyTorch Lightning;
* Hydra;
* NumPy;
* pandas or polars;
* scikit-learn;
* matplotlib;
* local experiment outputs.

TensorBoard is optional.

---

## 3. Repository Style

Keep code modular.

Separate at least:

* data loading;
* models;
* training;
* evaluation;
* logging;
* analysis;
* plotting;
* reporting.

Do not put the whole project into one notebook or one large script.

Codex may propose the exact repository structure.

---

## 4. Configuration

Use Hydra for experiment configuration.

Every meaningful run must save the resolved config:

```text
config.yaml
```

A result without a saved config is not reproducible.

Important configurable fields:

* dataset;
* split;
* model;
* optimizer;
* learning rate;
* batch size;
* number of epochs;
* seed;
* checkpoint path;
* output directory.

---

## 5. Dataset Rules

Default rule:

* noisy train subset is used for training;
* noisy validation subset is used for checkpoint selection;
* clean test subset is used only for final evaluation.

Do not use clean labels for:

* training loss;
* validation loss;
* checkpoint selection;
* descriptor construction;
* model selection.

If clean labels are used outside evaluation, mark the experiment as an oracle ablation.

---

## 6. Checkpoints

Training must save checkpoints.

Default inference checkpoint:

```text
checkpoint with minimum validation loss
```

Save metadata:

* best checkpoint path;
* best epoch;
* best validation loss;
* seed;
* dataset split;
* run ID.

Do not silently use the last checkpoint if a best-validation checkpoint exists.

---

## 7. Experiment Outputs

Each meaningful run should create one output folder.

Required files:

```text
config.yaml
metrics.json
run_metadata.json
report.md
```

Recommended folders:

```text
tables/
plots/
checkpoints/
artifacts/
```

The exact structure can change, but the run must be understandable without rerunning it.

---

## 8. Compact Tables

Generate compact tables for later ChatGPT analysis.

Compact means:

* essential columns only;
* short method names;
* rounded values;
* no unnecessary per-sample rows.

Use 3-4 meaningful digits in compact summaries.

Full-precision files may be stored separately.

Example:

```csv
method,split,seed,auroc,auprc,aurc,ece,acc
msp,worse,0,0.712,0.481,0.184,0.092,0.821
aum,worse,0,0.803,0.586,0.142,0.071,0.821
pred_desc,worse,0,0.819,0.604,0.136,0.068,0.821
```

---

## 9. Reports

Every important experiment must generate `report.md`.

The report should be factual, not advisory.

It should collect information from the saved config, metrics, tables, and generated plots.

Minimal structure:

```markdown
# Experiment Report: <run_id>

## Configuration

## Dataset

## Checkpoint

## Metrics

## Compact Results

## Generated Artifacts

## Notes
```

The report should not invent conclusions or next steps.

It may include a short factual `Notes` section if the run produced warnings, failures, missing artifacts, or unusual metric values.

Interpretation and research decisions can be written later by the human researcher or during separate analysis.

---

## 10. Tests

Add lightweight tests for fragile logic.

Prioritize:

* dataset loading;
* stable sample IDs;
* train/val/test split construction;
* checkpoint selection;
* metric computation;
* descriptor computation;
* report generation.

Do not overbuild tests before the baseline pipeline works.

---

## 11. Reproducibility Metadata

For important runs, save:

* seed;
* dataset;
* split;
* model;
* optimizer;
* learning rate;
* batch size;
* number of epochs;
* checkpoint path;
* git commit if available;
* device information if available.

Use multiple seeds for final claims when compute allows.

---

## 12. Agent Workflow

When using Codex:

1. inspect the repository;
2. read `README.md` and `AGENTS.md`;
3. propose a short plan;
4. implement one task;
5. run or add basic checks;
6. save outputs in the expected format.

Good tasks:

* create repository skeleton;
* implement dataset loading;
* implement training;
* implement checkpointing;
* implement logging;
* implement analysis scripts;
* implement metrics;
* implement plots;
* implement report generation.

Avoid tasks like:

* “implement everything”;
* “make it SOTA”;
* “add all possible methods”;
* “rewrite the whole repository.”

---

## 13. Definition of Done

A task is complete when it has:

* working code;
* saved config if relevant;
* saved outputs if relevant;
* compact table if relevant;
* `report.md` for meaningful experiments;
* basic validation for fragile logic.

A training run is not complete until its outputs can be inspected.
