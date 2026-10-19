"""Locate the attached completed base run and launch its downstream comparison."""

import json
import os
from pathlib import Path
import subprocess
import sys


def find_source(input_root):
    candidates = []
    for path in Path(input_root).rglob("meta_targets.npz"):
        if path.parent.name != "artifacts":
            continue
        source = path.parent.parent
        if (
            not (source / "checkpoints/best.pt").is_file()
            or not (source / "metrics.json").is_file()
        ):
            continue
        status = json.loads((source / "metrics.json").read_text())
        if status.get("status") == "complete" and status.get("completed_epochs") == 50:
            candidates.append(source)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            "Нужен ровно один подключённый результат завершённого обучения базы. "
            + f"Найдено: {len(candidates)}. Проверь Add Input → Notebook Outputs. "
            + str(candidates)
        )
    return candidates[0]


def main():
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Включи Accelerator → GPU T4 x2 и Internet → On.")
    source = find_source("/kaggle/input")
    config = OmegaConf.load(source / "config.yaml")
    name = config.dataset.name
    annotation = "CIFAR-100_human.pt" if name == "cifar100n" else "CIFAR-10_human.pt"
    image_folder = "cifar-100-python" if name == "cifar100n" else "cifar-10-batches-py"
    previous_data = source.parent.parent / "rfd_data"
    data_dir = (
        previous_data
        if (previous_data / annotation).is_file()
        and (previous_data / image_folder).is_dir()
        else Path("/kaggle/working/rfd_data")
    )
    repo = Path(__file__).resolve().parents[1]
    output = Path("/kaggle/working/rfd_post")
    print("Источник:", source, flush=True)
    print("Данные:", data_dir, flush=True)
    print(
        "GPU:",
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        flush=True,
    )
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR="/kaggle/working/matplotlib",
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run.py"),
            "post-kaggle",
            f"source_run={source}",
            f"data_dir={data_dir}",
            f"output_dir={output}",
            "device=cuda",
            "stage=all",
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    result = output / "post_kaggle_resnet50_seed42"
    print("\nГотово. Итоговая таблица:", flush=True)
    print((result / "tables/aggregate.csv").read_text(), flush=True)
    print("\nСохрани Notebook Version с outputs. Результат:", result, flush=True)


if __name__ == "__main__":
    main()
