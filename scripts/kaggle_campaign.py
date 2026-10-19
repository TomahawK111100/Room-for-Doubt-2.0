"""Start or restore the fixed independent-training campaign from one cell."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from kaggle_diagnose import find_inputs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["kaggle_campaign", "campaign_cifar100", "campaign_cifar10"], default="kaggle_campaign")
    profile = parser.parse_args().profile
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Включи GPU T4 x2 и Internet On.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "Для сопоставимого обучения базы в пакете B выбери GPU T4 x2."
        )
    source, post = find_inputs(["/kaggle/input"])
    root = Path(__file__).resolve().parents[1]
    planned = OmegaConf.load(root / "configs" / f"{profile}.yaml")
    output = Path(planned.output_dir) / planned.run_id
    data = Path("/kaggle/working/rfd_campaign_data")
    if not output.exists():
        candidates = [
            p.parent
            for p in Path("/kaggle/input").rglob("config.yaml")
            if p.parent.name == planned.run_id
            and (p.parent / "metrics.json").exists()
        ]
        if len(candidates) > 1:
            raise RuntimeError(
                "Подключи один предыдущий Output этой campaign для продолжения."
            )
        if candidates:
            print("Restoring campaign:", candidates[0], flush=True)
            shutil.copytree(candidates[0], output)
    data.mkdir(parents=True, exist_ok=True)
    original = OmegaConf.load(post / "config.yaml")
    prior_data = [Path(original.data_dir), source.parent.parent / "rfd_data"]
    for name in (
        "cifar-100-python",
        "cifar-10-batches-py",
        "CIFAR-100_human.pt",
        "CIFAR-10_human.pt",
    ):
        if (data / name).exists():
            continue
        found = next((p / name for p in prior_data if (p / name).exists()), None)
        if found is not None:
            shutil.copytree(found, data / name) if found.is_dir() else shutil.copy2(
                found, data / name
            )
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR="/kaggle/working/matplotlib",
    )
    print(
        f"Fixed campaign: {OmegaConf.to_container(planned.jobs)}",
        flush=True,
    )
    print(
        f"Each base: 50 epochs, then standard matched feature study. Budget: {planned.max_hours} hours.",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run.py"),
            "run-kaggle-campaign",
            f"--config-name={profile}",
            f"source_run={source}",
            f"post_run={post}",
            f"data_dir={data}",
        ],
        cwd=root,
        env=env,
        check=True,
    )
    print("Download:", output / "campaign_summary.zip", flush=True)


if __name__ == "__main__":
    main()
