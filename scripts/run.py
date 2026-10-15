#!/usr/bin/env python3
"""Run checkout commands before or after installing the package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from room_for_doubt.cli import main

if __name__ == "__main__":
    main()
