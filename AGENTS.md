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


## 14. Economical agent workflow and disabled ARIS

Project-owner instructions, 28 October and 4 November 2026: ARIS is disabled.
This section supersedes older ARIS routing and reviewer requirements.

* Do not invoke ARIS orchestration, model routers, audit chains, automatic reviewer
  passes or archived skill instructions. Do not read `.antigravity/` during normal
  work. Its files are historical references only; keep them preserved and ignored.
* Work directly by default. Use source inspection, deterministic calculations and
  focused tests before delegating. Routine edits, imports, tables, reports, plotting,
  file searches and training submission do not require an agent or a model upgrade.
* Delegate only a concrete independent question with a defined deliverable. Normally
  use one subagent; at most two ordinary subagents concurrently when their work is
  independent and parallel work offers a clear benefit. No automatic reviewer chains.
* Use an available ordinary non-Astra model with low or medium reasoning for routine
  delegated work. Inherit settings only when they are already economical and non-Astra;
  otherwise explicitly select an available ordinary model. Do not invent model IDs.
* Astra is an exceptional escalation, authorized conditionally by the owner on
  4 November: a consequential scientific/mathematical question, unresolved evaluation
  integrity issue, or difficult root-cause problem must remain after a focused ordinary
  attempt or deterministic check. Name the unresolved question and explain why another
  ordinary pass is unlikely to settle it. A generic desire for higher quality is not
  sufficient. A bounded independent review of a central scientific claim may qualify
  when it addresses a specific documented uncertainty, not routine approval.
* Before an Astra call, briefly state its purpose and why the escalation is justified.
  Use one Astra subagent at a time, low/medium reasoning initially, for one bounded
  question. High reasoning requires a specific justification; Ultra/max/maximum
  requires a separate explicit owner request. Never escalate automatically on a retry.
* Give agents only relevant files, facts and the exact question; prefer a fresh context
  rather than forking the entire chat. Request concise findings with evidence and a
  clear stopping condition. Do not let a subagent spawn further agents unless explicitly
  authorized for that task. Stop irrelevant or stalled agents promptly.
* Do not repeat reviews, broaden searches or rerun tests after adequate evidence is
  available unless a new change, failure or unresolved question justifies doing so.
  Do not monitor training logs repeatedly: verify submission once, then inspect completed
  artifacts when supplied or requested. Do not start background monitors automatically.
* These are project execution rules, not a technical account spending cap. Never claim
  exact token savings, a quota guarantee, or an app-wide model restriction without
  observing such enforcement. Do not change global Codex settings as part of this rule.

The optional `.antigravity/` ARIS files remain archived locally and ignored by Git;
they are not deleted and are not required to install or run the research package.
Their upstream commit was `26b95cfa0d8747078e9e43b42e20952709e561b8`.

Repository commands and migration mappings are documented in
`docs/reproduction.md` and `docs/repository_layout.md`. Use the offline test suite
for structural changes; do not rerun recorded experiments just to validate moves.
