"""Local/Kaggle entry point for the frozen S4 protocol."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from room_for_doubt.data.confirmation import SEEDS, verify_frozen, write_json
from room_for_doubt.training.temporal_snapshots import file_digest


def training(args, protocol_hash):
    from omegaconf import OmegaConf

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen.json").exists():
        verify_frozen(out, args.seed, protocol_hash)
        return {"status": "already_frozen"}
    deadline = args.deadline or monotonic() + 6 * 3600
    config = dict(
        seed=args.seed,
        protocol_sha256=protocol_hash,
        data_dir=str(args.data),
        test_accessed=False,
        base_epochs=50,
        max_session_hours=6,
    )
    path = out / "config.yaml"
    if path.exists() and OmegaConf.to_container(OmegaConf.load(path)) != config:
        raise ValueError("Existing S4 run configuration differs")
    OmegaConf.save(OmegaConf.create(config), path)
    base_metrics = out / "base/metrics.json"
    try:
        if (
            not base_metrics.exists()
            or json.loads(base_metrics.read_text())["status"] != "complete"
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run.py"),
                    "train-kaggle-base",
                    "--config-name",
                    "confirmation_s4_base",
                    f"seed={args.seed}",
                    f"output_dir={out}",
                    f"data_dir={args.data.resolve()}",
                    "max_hours=4.5",
                ],
                cwd=ROOT,
                timeout=max(1, deadline - monotonic()),
                check=True,
            )
        if json.loads(base_metrics.read_text())["status"] != "complete":
            metrics = {"status": "deferred_base_budget", "test_accessed": False}
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "freeze",
                    "--seed",
                    str(args.seed),
                    "--data",
                    str(args.data),
                    "--output",
                    str(out),
                ],
                cwd=ROOT,
                timeout=max(1, deadline - monotonic()),
                check=True,
            )
            metrics = json.loads((out / "metrics.json").read_text())
    except subprocess.TimeoutExpired:
        metrics = {"status": "deferred_session_budget", "test_accessed": False}
    write_json(out / "metrics.json", metrics)
    write_json(out / "run_metadata.json", dict(config=config, status=metrics["status"]))
    (out / "report.md").write_text(
        "# S4 fit and freeze\n\nCIFAR-10N Worse. Noisy training and validation.\n\n"
        f"Status: {metrics['status']}. External evaluation runs separately.\n"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "freeze", "evaluate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--external", type=Path)
    parser.add_argument(
        "--frozen", nargs=3, type=Path, metavar=("SEED42", "SEED43", "SEED44")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol_hash = file_digest(ROOT / "docs/planning/confirmation_protocol_v1.md")
    if args.stage == "evaluate":
        if args.frozen is None or args.external is None:
            parser.error("evaluation requires --frozen and --external")
        from room_for_doubt.evaluation.confirmation import evaluate

        metrics = evaluate(
            dict(zip(SEEDS, args.frozen)),
            args.data,
            args.external,
            args.output,
            protocol_hash,
            args.device,
        )
    else:
        if args.seed is None or args.external is not None or args.frozen is not None:
            parser.error("fit/freeze requires --seed and accepts only training data")
        import torch

        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(
            args.device
        ):
            raise RuntimeError("Select a Tesla T4 GPU for the frozen S4 benchmark")
        if args.stage == "fit":
            metrics = training(args, protocol_hash)
        else:
            from room_for_doubt.analysis.confirmation_fit import fit_and_freeze

            metrics = fit_and_freeze(
                args.output, args.data, args.seed, protocol_hash, args.device
            )
            write_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
