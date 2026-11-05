"""Bounded Mini-WebVision pilot, using the shared transfer-stage launcher."""

from pathlib import Path
from animal10n_pilot import main

if __name__ == "__main__":
    main(
        Path(__file__).resolve().parents[1] / "configs/mini_webvision_pilot.yaml",
        "/kaggle/working/rfd_miniwebvision_seed42",
    )
