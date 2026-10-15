# Repository layout

The installable package is `src/room_for_doubt/`. Models, data, training,
evaluation, analysis, reporting, plotting, trajectory formats and utilities are
separate modules. `scripts/run.py` dispatches the public CLI; `configs/` holds
Hydra configurations. `scripts/kaggle_run.py` manages the standalone Kaggle base
experiment and restoration from an attached prior notebook output.

Earlier top-level `src/train*.py`, `src/evaluate*.py` and plotting scripts now
live under their corresponding package subdirectories. Exploratory methods with
historical protocol limitations remain in `src/room_for_doubt/legacy/`.

Local-only locations: `data/`, `checkpoints/`, `artifacts/`, `outputs/`, `results/`,
`plots/`, `refine-logs/`, `.antigravity/`, `.aris/`. They are not needed for the
fresh Kaggle run. Existing local research evidence is retained, but not published.
