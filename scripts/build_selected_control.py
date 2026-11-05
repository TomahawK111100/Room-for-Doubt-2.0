from pathlib import Path
from build_temporal_residual_notebook import build

if __name__ == "__main__":
    build(
        Path(__file__).resolve().parents[1]
        / "outputs/temporal_selected_control_launch_20261027",
        selected_control=True,
    )
