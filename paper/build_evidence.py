"""Regenerate the manuscript tables and figures from compact evidence."""

from pathlib import Path
import subprocess
import sys

PAPER = Path(__file__).resolve().parent
BUILDERS = (
    "build_tables", "build_figures", "build_followups",
    "build_animal_pilot", "build_animal_replications",
    "build_miniwebvision_pilot", "build_miniwebvision_replications",
    "build_temporal_figures", "build_posthoc_controls", "build_teacher_cifar102",
    "build_external_diagnostics",
)


if __name__ == "__main__":
    for builder in BUILDERS:
        subprocess.run([sys.executable, str(PAPER / f"{builder}.py")], cwd=PAPER.parent, check=True)
