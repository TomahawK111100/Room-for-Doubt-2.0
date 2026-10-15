"""Locate Hydra configs in a source checkout or a wheel installation."""

from pathlib import Path
import sys


def config_directory() -> Path:
    checkout = Path(__file__).resolve().parents[3] / "configs"
    installed = Path(sys.prefix) / "share" / "room-for-doubt" / "configs"
    for candidate in (checkout, installed):
        if (candidate / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Hydra configs are missing. Install the package or run from its checkout."
    )
