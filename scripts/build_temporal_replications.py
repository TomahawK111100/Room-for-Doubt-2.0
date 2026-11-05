"""Build both preregistered follow-ups without changing historical pilot archives."""

from pathlib import Path
from build_temporal_residual_notebook import build

if __name__ == "__main__":
    root = (
        Path(__file__).resolve().parents[1]
        / "outputs/temporal_replications_launch_20261027"
    )
    for seed in (43, 44):
        build(root / f"seed{seed}", replication_seed=seed)
