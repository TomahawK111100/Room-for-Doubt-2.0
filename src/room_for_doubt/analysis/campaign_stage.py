"""One campaign stage per process, so large allocations cannot cross stages."""

import sys
from pathlib import Path
from omegaconf import OmegaConf


def main():
    stage, config, output = sys.argv[1:]
    cfg = OmegaConf.load(config)
    if stage == "base":
        from room_for_doubt.training.kaggle_base import train
        train(cfg, Path(output))
    elif stage == "study":
        from room_for_doubt.analysis.feature_study import run_study
        run_study(cfg, Path(output))
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
