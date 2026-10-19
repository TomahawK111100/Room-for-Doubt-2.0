"""Find the two completed notebook outputs, then run the unattended diagnostics."""

import json
import os
import subprocess
import sys
from pathlib import Path


def find_inputs(roots):
    posts = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for marker in root.rglob("all_policies_before_test.json"):
            post = marker.parent.parent
            if (post / "cache/test_complete.json").exists() and (
                post / "features/backbone.pt"
            ).exists():
                if (
                    json.loads((post / "metrics.json").read_text()).get("status")
                    == "complete"
                ):
                    posts.add(post.resolve())
    if len(posts) != 1:
        raise RuntimeError(
            "Нужен один Output завершённого post-kaggle запуска (rfd_post с cache, features, heads). Через Add Input → Notebook Outputs подключи notebook с таблицами aggregate/comparison. Найдено: "
            + str(sorted(map(str, posts)))
        )
    post = posts.pop()
    from omegaconf import OmegaConf

    config = OmegaConf.load(post / "config.yaml")
    original = Path(config.source_run)
    if (original / "checkpoints/best.pt").exists():
        return original, post
    sources = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for marker in root.rglob("meta_targets.npz"):
            source = marker.parent.parent
            metrics = source / "metrics.json"
            if (
                marker.parent.name == "artifacts"
                and (source / "checkpoints/best.pt").exists()
                and metrics.exists()
            ):
                status = json.loads(metrics.read_text())
                if (
                    status.get("status") == "complete"
                    and status.get("completed_epochs") == 50
                ):
                    sources.add(source.resolve())
    if len(sources) != 1:
        raise RuntimeError(
            "Также нужен Output исходного обучения базы room-for-doubttt: checkpoints/best.pt и artifacts/meta_targets.npz. Подключи его через Add Input → Notebook Outputs. Найдено: "
            + str(sorted(map(str, sources)))
        )
    return sources.pop(), post


def main(
    command="diagnose-kaggle",
    output_dir="/kaggle/working/rfd_diagnostics",
    profile="standard",
):
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Включи GPU T4 x2 и Internet On.")
    source, post = find_inputs(["/kaggle/input", "/kaggle/working/rfd_post"])
    cfg = OmegaConf.load(post / "config.yaml")
    annotation = (
        "CIFAR-100_human.pt" if cfg.dataset.name == "cifar100n" else "CIFAR-10_human.pt"
    )
    candidates = [
        Path(cfg.data_dir),
        source.parent.parent / "rfd_data",
        Path("/kaggle/working/rfd_data"),
    ]
    data_dir = next(
        (p for p in candidates if (p / annotation).exists()), candidates[-1]
    )
    if not (data_dir / annotation).exists():
        from room_for_doubt.data.cifar_noisy import load_observed_labels
        from torchvision.datasets import CIFAR10, CIFAR100

        load_observed_labels(data_dir, cfg.dataset.name, cfg.dataset.noisy_split)
        {"cifar10n": CIFAR10, "cifar100n": CIFAR100}[cfg.dataset.name](
            str(data_dir), train=True, download=True
        )
    root = Path(__file__).resolve().parents[1]
    print("Base:", source, "\nPost:", post, "\nData:", data_dir, flush=True)
    print(
        "GPU:",
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        flush=True,
    )
    print(
        "База не переобучается. "
        + (
            "Сравнение двух энкодеров и трёх целей: 18 малых голов, PCA и forward-проходы."
            if command == "study-base-features" and profile == "standard"
            else "Расширенная проверка окон истории: 42 малых головы."
            if command == "study-base-features"
            else "Девять малых голов и диагностические forward-проходы."
        )
        + " Используется одна GPU.",
        flush=True,
    )
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR="/kaggle/working/matplotlib",
    )
    args = [
        sys.executable,
        str(root / "scripts/run.py"),
        command,
        f"source_run={source}",
        f"post_run={post}",
        f"data_dir={data_dir}",
        f"output_dir={output_dir}",
        "device=cuda",
    ]
    if profile == "extended":
        args.append("--config-name=feature_study_extended")
    subprocess.run(
        args,
        cwd=root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command",
        choices=["diagnose-kaggle", "study-base-features"],
        default="diagnose-kaggle",
    )
    parser.add_argument("--output-dir", default="/kaggle/working/rfd_diagnostics")
    parser.add_argument(
        "--profile", choices=["standard", "extended"], default="standard"
    )
    args = parser.parse_args()
    main(args.command, args.output_dir, args.profile)
