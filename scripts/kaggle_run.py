"""Run from a Kaggle cell after cloning/installing this repository."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

RUN_ID = "cifar100n_resnet50_gce_seed42"


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Kaggle Settings -> Accelerator -> GPU T4 x2; Internet -> On"
        )
    print(
        "GPUs:",
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        flush=True,
    )
    repo = Path(__file__).resolve().parents[1]
    output = Path("/kaggle/working/rfd_outputs") / RUN_ID
    # Restoring a previous notebook output is optional; first runs download everything.
    if not output.exists():
        candidates = [
            p.parent.parent
            for p in Path("/kaggle/input").rglob("last.pt")
            if p.parent.name == "checkpoints" and p.parent.parent.name == RUN_ID
        ]
        if len(candidates) > 1:
            raise RuntimeError(
                "Several attached resumes found; attach only the intended previous output."
            )
        if candidates:
            shutil.copytree(candidates[0], output)
            print("Restored prior run:", candidates[0], flush=True)
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        MPLCONFIGDIR="/kaggle/working/matplotlib",
        PYTHONUNBUFFERED="1",
    )
    subprocess.run(
        [sys.executable, str(repo / "scripts/run.py"), "train-kaggle-base"],
        cwd=repo,
        env=env,
        check=True,
    )
    print("\nOUTPUT:", output, flush=True)
    print((output / "metrics.json").read_text(), flush=True)
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics["status"] != "complete":
        print(
            "Run paused with checkpoints saved. Save notebook output, attach it as input to a new session, and run the same cell.",
            flush=True,
        )
    else:
        print(
            "Finished. Save Version with outputs to preserve checkpoints, predictions and trajectories.",
            flush=True,
        )


if __name__ == "__main__":
    main()
