"""Check the public package, output reports and Git publishing boundary."""

import importlib
import importlib.util
import json
from pathlib import Path
import pkgutil
import subprocess

import room_for_doubt
from room_for_doubt.reporting.artifacts import generate_report

ROOT = Path(__file__).resolve().parents[1]


def test_every_package_module_imports_without_running_experiments(monkeypatch):
    import torchvision
    import torch
    import pytorch_lightning as pl

    def forbidden(*args, **kwargs):
        raise AssertionError("Import attempted to load data, weights, or run training")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torchvision.datasets.CIFAR10, "__init__", forbidden)
    monkeypatch.setattr(torchvision.datasets.CIFAR100, "__init__", forbidden)
    monkeypatch.setattr(pl.Trainer, "fit", forbidden)
    for module in pkgutil.walk_packages(
        room_for_doubt.__path__, room_for_doubt.__name__ + "."
    ):
        importlib.import_module(module.name)


def test_report_lists_only_existing_artifacts(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"acc": 0.7891234}))
    generate_report(tmp_path, "offline", {"dataset": {}}, {"acc": 0.7891234})
    report = (tmp_path / "report.md").read_text()
    assert "## Compact Results" in report
    assert "0.7891" in report
    assert "`checkpoints/`" not in report
    assert "`config.yaml`" not in report
    assert (tmp_path / "tables/metrics.csv").read_text().splitlines() == [
        "metric,value",
        "acc,0.7891",
    ]


def test_package_sources_are_not_ignored():
    for path in (ROOT / "src/room_for_doubt").rglob("*.py"):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", str(path)], cwd=ROOT
        )
        assert result.returncode == 1, path


def test_size_guard_checks_public_files_and_committed_blobs(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "check_file_sizes", ROOT / "scripts/check_file_sizes.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("private.bin\n")
    (tmp_path / "private.bin").write_bytes(b"x" * 100)
    (tmp_path / "public.bin").write_bytes(b"x" * 20)
    assert list(guard.oversized_files(tmp_path, 20)) == [("public.bin", 20)]
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    (tmp_path / "public.bin").unlink()
    assert ("public.bin", 20) in list(guard.oversized_blobs(tmp_path, 20))
