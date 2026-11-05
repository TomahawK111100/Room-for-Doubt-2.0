"""Launch packaging and stage boundaries, without network or GPU work."""

import importlib.util
import io
import json
from pathlib import Path
import zipfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_source_notebook_contains_current_code_no_data_and_no_final_eval(tmp_path):
    builder = module("build_temporal_residual_notebook")
    first, manifest = builder.build(tmp_path / "a")
    _, second = builder.build(tmp_path / "b")
    assert manifest == second
    assert manifest["test_evaluation"] is False
    code = "".join(first["cells"][0]["source"])
    compile(code, "<notebook>", "exec")
    with zipfile.ZipFile(tmp_path / "a/source_snapshot.zip") as archive:
        assert sorted(archive.namelist()) == sorted(manifest["files"])
        assert "src/room_for_doubt/training/temporal_snapshots.py" in archive.namelist()
        assert (
            archive.read("configs/temporal_base.yaml")
            == (ROOT / "configs/temporal_base.yaml").read_bytes()
        )
        assert all(Path(n).suffix in {".py", ".yaml"} for n in archive.namelist())
        assert all(
            not Path(n).is_absolute() and ".." not in Path(n).parts
            for n in archive.namelist()
        )
        runner = archive.read("scripts/temporal_residual_pilot.py").decode()
        assert '"stage=evaluate"' not in runner
    assert "CUDA_WITNESS_PASS" in code
    assert "git push" not in code


def test_pipeline_exports_validation_summary_without_evaluate(tmp_path, monkeypatch):
    runner = module("temporal_residual_pilot")
    out = tmp_path / "run"
    stages = []

    def run(command, **kwargs):
        if "train-kaggle-base" in command:
            stages.append("base")
            path = out / "bases" / runner.BASE_ID
            path.mkdir(parents=True)
            (path / "metrics.json").write_text(
                json.dumps(dict(status="complete", test_evaluated=False))
            )
        elif "stage=prepare" in command:
            stages.append("prepare")
        elif "stage=fit" in command:
            stages.append("fit")
            path = out / "heads" / runner.HEAD_ID
            path.mkdir(parents=True)
            (path / "metrics.json").write_text(
                json.dumps(
                    dict(
                        status="policy_frozen",
                        pilot_diagnosis="synthetic",
                        teacher_val_gain_pp=1.0,
                        student_val_gain_pp=0.5,
                        selected_alpha=0.5,
                    )
                )
            )
        else:
            raise AssertionError("Unexpected stage or clean evaluation")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    metrics = runner.execute(out, tmp_path / "data")
    assert stages == ["base", "prepare", "fit"]
    assert metrics["status"] == "validation_complete_test_deferred"
    assert metrics["test_evaluated"] is False
    with zipfile.ZipFile(out / "temporal_summary.zip") as archive:
        assert "config.yaml" in archive.namelist()
        progress = json.loads(archive.read("progress.json"))
        assert progress["status"] == metrics["status"]
        assert progress["stage"] == "finished"
        assert not any(
            name.endswith((".pt", ".npy", ".npz")) for name in archive.namelist()
        )


def test_paused_base_never_starts_downstream(tmp_path, monkeypatch):
    runner = module("temporal_residual_pilot")
    out = tmp_path / "run"

    def run(command, **kwargs):
        assert "train-kaggle-base" in command
        path = out / "bases" / runner.BASE_ID
        path.mkdir(parents=True)
        (path / "metrics.json").write_text('{"status":"paused_time_budget"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.execute(out, tmp_path / "data")["status"] == "paused_base_time_budget"
    with zipfile.ZipFile(out / "temporal_summary.zip") as archive:
        assert (
            json.loads(archive.read("progress.json"))["status"]
            == "paused_base_time_budget"
        )


def test_replication_build_has_distinct_seeds_and_preserves_default(tmp_path):
    builder = module("build_temporal_residual_notebook")
    for seed in (43, 44):
        notebook, manifest = builder.build(tmp_path / str(seed), replication_seed=seed)
        code = "".join(notebook["cells"][0]["source"])
        compile(code, "replication", "exec")
        assert manifest["seed"] == seed and manifest["test_evaluation"] is True
        assert f'"--seed", "{seed}"' in code
        assert "scripts/temporal_replication.py" in code
        assert f"rfd_temporal_replication_seed{seed}" in code


def test_single_epoch_cache_preserves_inputs_and_source_targets(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "synthetic_fixture", ROOT / "tests/test_temporal_residual.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    synthetic_cache = fixture.synthetic_cache
    from room_for_doubt.data.temporal_evidence import load_evidence
    import numpy as np

    runner = module("temporal_replication")
    source = tmp_path / "source"
    original = synthetic_cache(source)
    for split, key in (("train", "teacher_probs"), ("val", "val_teacher_probs")):
        q = np.roll(original[key], 1, axis=1).copy()
        np.save(source / f"{split}_epoch001_probs.npy", q)
    runner.single_snapshot_cache(source, tmp_path / "control")
    manifest, control = load_evidence(tmp_path / "control")
    _, unchanged = load_evidence(source)
    assert [s["epoch"] for s in manifest["snapshots"]] == [1]
    assert not np.array_equal(control["teacher_probs"], original["teacher_probs"])
    for key in original:
        if key not in ("teacher_probs", "val_teacher_probs"):
            assert np.array_equal(control[key], original[key])
        assert np.array_equal(unchanged[key], original[key])


def test_selected_teacher_uses_noisy_validation_and_deterministic_ties(tmp_path):
    import numpy as np

    runner = module("temporal_selected_control")
    base = np.array([[0.6, 0.4], [0.6, 0.4]], dtype=np.float32)
    labels = np.array([0, 1])
    good = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    chosen, rows = runner.choose_epoch(base, {1: base, 2: good, 3: good}, labels)
    assert chosen["epoch"] == 2 and chosen["acc"] == 1
    tied, _ = runner.choose_epoch(base, {1: base, 2: base}, labels)
    assert tied["epoch"] == 1 and tied["alpha"] == 0
    builder = module("build_temporal_residual_notebook")
    notebook, m = builder.build(tmp_path / "launch", selected_control=True)
    code = "".join(notebook["cells"][0]["source"])
    compile(code, "selected_control", "exec")
    assert "scripts/temporal_selected_control.py" in code
    assert m["seeds"] == [43, 44] and m["test_evaluation"] is True
    assert (
        "train-kaggle-base"
        not in Path(ROOT / "scripts/temporal_selected_control.py").read_text()
    )
