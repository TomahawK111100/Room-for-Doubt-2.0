"""S2 scientific boundaries, target alignment and restart invariants."""

from pathlib import Path
import numpy as np
import pytest
import torch
from room_for_doubt.data.posthoc_controls import (
    aligned_rows,
    mean_probabilities,
    inspect_inputs,
)
from room_for_doubt.analysis.posthoc_controls import (
    branches,
    targets,
    freeze_group,
    verify_group,
    choose_epoch,
)
from room_for_doubt.training.posthoc_controls import fit_head, input_base
from room_for_doubt.models.temporal_residual import (
    distillation_kl,
    TemporalResidualHead,
)
from room_for_doubt.evaluation.temporal_residual import corrected_probabilities


def arrays():
    rng = np.random.default_rng(7)

    def probs(n):
        p = rng.random((n, 100)).astype(np.float32)
        p /= p.sum(1, keepdims=True)
        return p

    return (
        dict(
            train_features=rng.normal(size=(40, 8)).astype(np.float32),
            train_probs=probs(40),
            train_ids=np.arange(40),
            val_ids=np.arange(40, 46),
            val_features=rng.normal(size=(6, 8)).astype(np.float32),
            val_probs=probs(6),
            val_noisy_labels=np.arange(6),
            meta_train_rows=np.arange(24),
            meta_val_rows=np.arange(24, 30),
        ),
        probs,
    )


def test_alignment_requires_exact_unique_membership():
    assert aligned_rows(np.array([3, 1, 2]), np.array([1, 2, 3])).tolist() == [1, 2, 0]
    for source, requested in [([1, 1], [1, 2]), ([1, 2], [1, 3]), ([1.0, 2.0], [1, 2])]:
        with pytest.raises(ValueError):
            aligned_rows(np.array(source), np.array(requested))


def test_probability_mean_uses_aligned_donors_and_excludes_recipient():
    a, probs = arrays()
    all_a = {42: a}
    for seed in (43, 44):
        all_a[seed] = {k: np.copy(v) for k, v in a.items()}
        for split in ("train", "val"):
            q = np.zeros_like(a[f"{split}_probs"])
            q[:, seed - 40] = 1
            all_a[seed][f"{split}_probs"] = q[::-1]
            all_a[seed][f"{split}_ids"] = a[f"{split}_ids"][::-1]
    out, _, _ = targets(42, "independent2", {42: {}}, all_a, None)
    assert np.all(out[0][:, 3:5] == 0.5)
    assert np.all(out[0].sum(1) == 1)
    assert not np.array_equal(out[0], a["train_probs"])
    with pytest.raises(ValueError):
        mean_probabilities([])


def test_missing_preflight_never_opens_test_or_trains(tmp_path, monkeypatch):
    monkeypatch.setattr(
        np,
        "load",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Array access")),
    )
    report, sources = inspect_inputs(tmp_path)
    assert report["status"] == "missing_input" and sources == {}
    assert [r["seed"] for r in report["studies"]] == [42, 43, 44]
    assert report["test_predictions_opened"] is False


def test_package_freeze_blocks_partial_and_modified_policy(tmp_path):
    jobs = {"base42_x": tmp_path / "head"}
    with pytest.raises(ValueError):
        freeze_group(tmp_path, jobs)
    p = jobs["base42_x"] / "artifacts/policy_before_test.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"alpha":0}')
    freeze_group(tmp_path, jobs)
    verify_group(tmp_path, jobs)
    p.write_text('{"alpha":1}')
    with pytest.raises(ValueError):
        verify_group(tmp_path, jobs)


def test_direct_head_and_onehot_ce_are_defined_correctly():
    p = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    h = torch.tensor([[1.0, -1.0], [-0.3, 0.3]])
    labels = torch.tensor([1, 0])
    q = torch.nn.functional.one_hot(labels, 2).float()
    assert torch.allclose(
        distillation_kl(p, h, q), torch.nn.functional.cross_entropy(p.log() + h, labels)
    )
    assert torch.allclose(input_base(p, "direct"), torch.full_like(p, 0.5))
    base = p.numpy()
    direct_res = h.numpy() - np.log(base)
    actual = corrected_probabilities(base, direct_res, 1)
    assert np.allclose(actual, h.softmax(1).numpy())
    assert np.array_equal(corrected_probabilities(base, direct_res, 0), base)


def test_selected_teacher_tie_is_earlier_epoch():
    p = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    best, rows = choose_epoch(p, {5: p, 1: p}, np.array([0, 1]))
    assert best["epoch"] == 1 and best["alpha"] == 0 and len(rows) == 8


def test_declared_package_has_thirteen_new_heads_without_refitting_old_controls():
    a, b = branches("A"), branches("B")
    assert len(a) == 10 and len(b) == 3
    assert (42, "selected1") in a and (43, "selected1") not in a
    assert set(a).isdisjoint(set(b))


def test_interrupted_head_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import room_for_doubt.training.posthoc_controls as m

    torch.set_num_threads(1)
    a, probs = arrays()
    qt, qv = probs(40), probs(6)
    contract = dict(
        classes=100,
        hidden_dim=4,
        parametrization="residual",
        branch="early2",
        test_only=True,
    )
    original = m.atomic_save

    def interrupt_after_last(saved, path):
        original(saved, path)
        if Path(path).name == "last.pt" and saved["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(m, "atomic_save", interrupt_after_last)
    with pytest.raises(RuntimeError, match="interruption"):
        fit_head(
            tmp_path / "resume", a, qt, qv, contract, "cpu", epochs=2, batch_size=8
        )
    assert not (tmp_path / "resume/artifacts/policy_before_test.json").exists()
    monkeypatch.setattr(m, "atomic_save", original)
    p1 = fit_head(
        tmp_path / "resume", a, qt, qv, contract, "cpu", epochs=2, batch_size=8
    )
    p2 = fit_head(
        tmp_path / "fresh", a, qt, qv, contract, "cpu", epochs=2, batch_size=8
    )
    state1 = torch.load(tmp_path / "resume/checkpoints/best.pt", weights_only=False)[
        "state_dict"
    ]
    state2 = torch.load(tmp_path / "fresh/checkpoints/best.pt", weights_only=False)[
        "state_dict"
    ]
    assert all(torch.equal(state1[k], state2[k]) for k in state1)
    assert p1["policy"] == p2["policy"]
    with pytest.raises(ValueError, match="another target"):
        fit_head(
            tmp_path / "resume",
            a,
            qt,
            qv,
            {**contract, "branch": "wrong"},
            "cpu",
            epochs=2,
            batch_size=8,
        )


def test_runtime_plan_reuses_base_features_and_identical_teacher():
    from room_for_doubt.evaluation.posthoc_runtime import ForwardPlan

    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 2)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.fc(x)

    model = Small()
    head = TemporalResidualHead(torch.zeros(3), torch.ones(3), 2, 4)
    x = torch.ones(2, 3)
    student = ForwardPlan({"base": model}, "base", "student", alpha=1.0, head=head)
    p = student(x)
    assert model.calls == 1 and p.shape == (2, 2)
    mixed = ForwardPlan({"base": model}, "base", "mixture", ["base"], 0.5)
    q = mixed(x)
    assert model.calls == 2 and torch.allclose(p, q)


def test_build_s2_notebooks_are_deterministic_and_preflight_precedes_cuda(tmp_path):
    import importlib.util, zipfile

    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "s2build", root / "scripts/build_posthoc_controls.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a, ma = m.build(tmp_path / "a", "A")
    b, mb = m.build(tmp_path / "b", "B")
    assert ma["source_archive_sha256"] == mb["source_archive_sha256"]
    assert ma["new_head_fits"] == 10 and mb["new_head_fits"] == 3
    code = "".join(a["cells"][0]["source"])
    compile(code, "s2.ipynb", "exec")
    assert code.index('"stage=preflight"') < code.index(
        "assert torch.cuda.is_available()"
    )
    assert "git push" not in code and "kaggle kernels output" not in code
    with zipfile.ZipFile(tmp_path / "a/source_snapshot.zip") as z:
        assert set(z.namelist()) == set(ma["files"])
        assert all(Path(n).suffix in (".py", ".yaml", ".md") for n in z.namelist())
        assert "src/room_for_doubt/training/posthoc_controls.py" in z.namelist()


def test_canonical_witness_rejects_drift_before_fit(tmp_path, monkeypatch):
    import room_for_doubt.analysis.posthoc_controls as m
    from types import SimpleNamespace

    a, _ = arrays()
    source = {42: {"base": tmp_path / "base"}}
    monkeypatch.setattr(
        m, "load_completed_run", lambda *args: {"checkpoint_digest": "pinned"}
    )
    monkeypatch.setattr(m, "restore_base", lambda *args: object())
    monkeypatch.setattr(m, "extract", lambda *args: (a["val_features"], a["val_probs"]))
    cfg = SimpleNamespace(output_dir=str(tmp_path), device="cpu", image_batch_size=64)
    (tmp_path / "artifacts").mkdir()
    assert m.canonical_witness(cfg, source, None, None, {42: a})[0]["seed"] == 42
    drift = np.array(a["val_features"])
    drift[0, 0] += 1
    monkeypatch.setattr(m, "extract", lambda *args: (drift, a["val_probs"]))
    with pytest.raises(ValueError, match="before fit"):
        m.canonical_witness(cfg, source, None, None, {42: a})


def test_export_keeps_selection_evidence_without_large_targets(tmp_path):
    import zipfile
    from types import SimpleNamespace
    from room_for_doubt.analysis.posthoc_controls import exports

    target = tmp_path / "targets/base42_selected1"
    target.mkdir(parents=True)
    (target / "selection_grid.csv").write_text("epoch,alpha\n3,0.5\n")
    (target / "selection.json").write_text('{"epoch":3}')
    np.save(target / "train_targets.npy", np.ones((2, 3)))
    cache = tmp_path / "evaluation_cache"
    cache.mkdir()
    (cache / "internal.json").write_text("{}")
    (tmp_path / "weights.pt").write_bytes(b"weights")
    exports(
        tmp_path,
        {"status": "complete"},
        SimpleNamespace(package="A", device="cpu", run_id="test"),
    )
    for name in ("posthoc_summary.zip", "posthoc_results.zip"):
        with zipfile.ZipFile(tmp_path / name) as archive:
            names = archive.namelist()
            assert "targets/base42_selected1/selection_grid.csv" in names
            assert "targets/base42_selected1/selection.json" in names
            assert not any(n.endswith((".npy", ".pt", ".zip")) for n in names)
            assert not any(n.startswith("evaluation_cache/") for n in names)


def test_benchmark_notebook_cannot_default_to_fit(tmp_path):
    import importlib.util

    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "runtimebuild", root / "scripts/build_posthoc_controls.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook, manifest = module.build(tmp_path, "A", stage="benchmark")
    code = "".join(notebook["cells"][0]["source"])
    assert 'MODE = "benchmark"' in code
    assert "rfd_posthoc_runtime_v1" in code
    assert manifest["new_head_fits"] == 0
    assert manifest["default_stage"] == "benchmark"
    diagnostic = (tmp_path / "posthoc_controls_A_preflight.ipynb").read_text()
    import json

    assert 'MODE = "preflight"' in "".join(json.loads(diagnostic)["cells"][0]["source"])
    with pytest.raises(ValueError, match="single package A"):
        module.build(tmp_path / "bad", "B", stage="benchmark")


def test_benchmark_stage_never_calls_fit_or_test_evaluation(tmp_path, monkeypatch):
    import room_for_doubt.analysis.posthoc_controls as module
    import room_for_doubt.evaluation.posthoc_runtime as timing
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/posthoc_controls.yaml")
    cfg.output_dir = str(tmp_path / "output")
    cfg.input_root = str(tmp_path / "inputs")
    cfg.data_dir = str(tmp_path / "data")
    cfg.stage, cfg.require_cuda, cfg.device = "benchmark", False, "cpu"
    sources = {s: {"cache": tmp_path / str(s)} for s in (42, 43, 44)}
    monkeypatch.setattr(
        module, "inspect_inputs", lambda _: ({"status": "verified_files"}, sources)
    )
    monkeypatch.setattr(module, "training_data", lambda _: ("images", "noisy"))
    monkeypatch.setattr(module, "verify_data", lambda *a: None)
    monkeypatch.setattr(module, "load_evidence", lambda _: (None, {}))

    def forbidden(*args, **kwargs):
        pytest.fail("Runtime stage must never fit or evaluate clean test")

    for name in (
        "canonical_witness",
        "resource_smoke",
        "target_contract",
        "fit_head",
        "freeze_group",
        "evaluate_group",
    ):
        monkeypatch.setattr(module, name, forbidden)
    calls = []

    def fake_benchmark(*args):
        calls.append(args)
        return {"status": "complete"}

    monkeypatch.setattr(timing, "benchmark", fake_benchmark)
    result = module.execute(cfg)
    assert result["status"] == "runtime_complete"
    assert len(calls) == 1
    assert calls[0][2:5] == ("images", "noisy", "cpu")
