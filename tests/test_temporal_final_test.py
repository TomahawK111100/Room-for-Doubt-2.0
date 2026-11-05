"""Evaluation launcher contracts; synthetic artifacts only, no real test reading."""

import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_final_notebook_is_evaluation_only_and_hash_bound(tmp_path):
    builder = load_module("build_temporal_test_notebook")
    nb = builder.build(tmp_path)
    code = "".join(nb["cells"][0]["source"])
    compile(code, "<final-notebook>", "exec")
    manifest = json.loads((tmp_path / "launch_manifest.json").read_text())
    import hashlib

    raw = (ROOT / "scripts/temporal_residual_final_test.py").read_bytes()
    assert manifest["script_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "stage=evaluate" in raw.decode()
    assert "stage=fit" not in raw.decode() and "train-kaggle-base" not in raw.decode()
    assert "EVALUATION_GPU_WITNESS_PASS" in code


def test_compact_summary_is_insufficient_for_final_evaluation(tmp_path):
    module = load_module("temporal_residual_final_test")
    source = ROOT / "outputs/temporal_pilot_import_20261027/source_summary"
    if not source.exists():
        pytest.skip("Local real compact import is absent")
    with pytest.raises(ValueError, match="Missing full Output artifact"):
        module.locate_pilot(source)


def test_exports_pairwise_comparisons_and_complete_predictions(tmp_path):
    module = load_module("temporal_residual_final_test")
    out = tmp_path / "output"
    run = out / "heads" / module.HEAD / "final_evaluation"
    (run / "tables").mkdir(parents=True)
    pilot = tmp_path / "input"
    (pilot / "heads" / module.HEAD / "artifacts").mkdir(parents=True)
    (pilot / "heads" / module.HEAD / "artifacts/policy_before_test.json").write_text(
        "{}"
    )
    labels = np.zeros(10000, dtype=np.int64)
    base = np.zeros((10000, 100), dtype=np.float32)
    base[:, 0] = 1
    student = base.copy()
    student[:100] = 0
    student[:100, 1] = 1
    np.savez_compressed(
        run / "predictions.npz",
        labels=labels,
        ids=np.arange(10000),
        base=base,
        corrected=student,
        snapshot_mean=base,
        teacher_interpolation=base,
    )
    (run / "metrics.json").write_text(
        json.dumps(
            dict(
                status="complete",
                methods=dict(
                    identity=dict(acc=1.0, nll=0.0),
                    temporal_residual=dict(acc=0.99, nll=0.1),
                ),
            )
        )
    )
    (run / "report.md").write_text("# Synthetic evaluation report\n")
    result = module.finish_exports(out, pilot)
    assert len(result["paired_comparisons"]) == 3
    assert result["paired_comparisons"][0]["gain_pp"] == -1
    assert result["test_samples"] == 10000
    import zipfile

    with zipfile.ZipFile(out / "temporal_test_results.zip") as archive:
        assert any(n.endswith("predictions.npz") for n in archive.namelist())
    with zipfile.ZipFile(out / "temporal_test_summary.zip") as archive:
        assert not any(n.endswith("predictions.npz") for n in archive.namelist())
        assert "config.yaml" in archive.namelist()


def test_raw_import_replays_saved_metrics_and_rejects_changed_predictions(tmp_path):
    export = load_module("temporal_residual_final_test")
    analysis = load_module("analyze_temporal_test")
    from room_for_doubt.evaluation.temporal_residual import evaluate_probabilities

    out = tmp_path / "output"
    run = out / "heads" / export.HEAD / "final_evaluation"
    (run / "tables").mkdir(parents=True)
    pilot = tmp_path / "input"
    artifact = pilot / "heads" / export.HEAD / "artifacts"
    artifact.mkdir(parents=True)
    (artifact / "policy_before_test.json").write_text("{}")
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 100, 10000)
    base = rng.dirichlet(np.ones(100), size=10000).astype(np.float32)
    student = base.copy()
    student[:100] = np.roll(base[:100], 1, axis=1)
    np.savez_compressed(
        run / "predictions.npz",
        ids=np.arange(10000),
        labels=labels,
        base=base,
        corrected=student,
        snapshot_mean=base,
        teacher_interpolation=base,
    )
    rows = evaluate_probabilities(labels, base, student)
    rows["snapshot_mean_5passes"] = rows["identity"]
    rows["teacher_interpolation_extra_passes"] = rows["identity"]
    (run / "metrics.json").write_text(json.dumps(dict(status="complete", methods=rows)))
    (run / "report.md").write_text("# Synthetic fixture\n")
    export.finish_exports(out, pilot)
    analysis.analyze(out / "temporal_test_results.zip", tmp_path / "import")
    checks = json.loads((tmp_path / "import/checks.json").read_text())
    assert len(checks) > 35 and all(r["passed"] for r in checks)
    with (run / "predictions.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="prediction SHA256"):
        analysis.analyze(out, tmp_path / "tampered_import")
