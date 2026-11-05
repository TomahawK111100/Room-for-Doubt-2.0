"""Synthetic correctness checks, never evidence of benchmark improvement."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from room_for_doubt.models.temporal_residual import (
    TemporalResidualHead,
    residual_posterior,
    distillation_kl,
)
from room_for_doubt.data.temporal_evidence import seal_evidence, load_evidence
from room_for_doubt.data.meta_splits import reserve_meta_splits
from room_for_doubt.training.temporal_residual import fit, load_frozen, feature_stats
from room_for_doubt.training.temporal_snapshots import (
    snapshot_epochs,
    save_snapshot,
    verify_snapshots,
)
from room_for_doubt.training.kaggle_base import atomic_save, resume_contract
from room_for_doubt.evaluation.temporal_residual import (
    select_alpha,
    evaluate_probabilities,
)


def synthetic_cache(root, n=100):
    root.mkdir(parents=True)
    rng = np.random.default_rng(813)
    train_ids, val_ids = np.arange(n, dtype=np.int64), np.arange(
        n, n + 20, dtype=np.int64
    )
    f = rng.normal(size=(n, 4)).astype(np.float32)
    valf = rng.normal(size=(20, 4)).astype(np.float32)

    def q(x):
        z = np.stack((1.8 * x[:, 0], -1.8 * x[:, 0], 0.5 * x[:, 1]), 1)
        return torch.from_numpy(z).softmax(1).numpy()

    arrays = dict(
        train_ids=train_ids,
        val_ids=val_ids,
        train_features=f,
        val_features=valf,
        train_probs=np.full((n, 3), 1 / 3, dtype=np.float32),
        val_probs=np.full((20, 3), 1 / 3, dtype=np.float32),
        teacher_probs=q(f),
        val_teacher_probs=q(valf),
        val_noisy_labels=q(valf).argmax(1),
    )
    partitions = reserve_meta_splits(train_ids, 20261015)
    for name in ("train", "val"):
        arrays[f"meta_{name}_rows"] = partitions[f"meta_{name}_ids"]
    for name, value in arrays.items():
        np.save(root / f"{name}.npy", value)
    seal_evidence(
        root,
        dict(
            classes=3,
            feature_width=4,
            class_axis=[0, 1, 2],
            split_seed=20261015,
            base_digest="synthetic-not-a-real-base",
            snapshots=[dict(epoch=e, sha256=f"synthetic-{e}") for e in range(1, 6)],
            dataset=dict(name="synthetic", noisy_split="synthetic", val_ratio=0.1),
        ),
    )
    return arrays


def fit_config(root):
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/temporal_residual.yaml")
    cfg.cache_dir = str(root)
    cfg.dataset = dict(name="synthetic", noisy_split="synthetic", val_ratio=0.1)
    cfg.epochs = 15
    cfg.batch_size = 32
    cfg.hidden_dim = 16
    cfg.learning_rate = 0.03
    return cfg


def test_identity_and_class_correction_exceed_view_reweighting_ceiling():
    p = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
    head = TemporalResidualHead(np.zeros(4), np.ones(4), 2, 8)
    assert torch.equal(head(torch.ones(2, 4)), torch.zeros(2, 2))
    assert torch.allclose(residual_posterior(p, head(torch.ones(2, 4)), 1), p)
    h = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
    assert torch.equal(residual_posterior(p, h, 0), p)
    assert (residual_posterior(p, h, 1).argmax(1) == 1).all()
    # Every nonnegative convex view mixture is class0; residual can change it.
    for w in np.linspace(0, 1, 101):
        assert (w * p[0] + (1 - w) * p[1]).argmax() == 0
    loss = distillation_kl(
        p, h.requires_grad_(), torch.tensor([[0.1, 0.9], [0.1, 0.9]])
    )
    loss.backward()
    assert torch.isfinite(h.grad).all()


def test_fit_normalization_does_not_see_holdout():
    features = np.array([[1.0, 2.0], [3.0, 4.0], [1e8, -1e8]], dtype=np.float32)
    mean, scale = feature_stats(features, np.array([0, 1]))
    np.testing.assert_array_equal(mean, [2.0, 3.0])
    np.testing.assert_array_equal(scale, [1.0, 1.0])


def test_evidence_rejects_corruption_overlap_and_partition_drift(tmp_path):
    root = tmp_path / "cache"
    arrays = synthetic_cache(root)
    load_evidence(root)
    changed = arrays["train_probs"].copy()
    changed[0] = [0.8, 0.1, 0.1]
    np.save(root / "train_probs.npy", changed)
    with pytest.raises(ValueError, match="SHA256"):
        load_evidence(root)
    np.save(root / "train_probs.npy", arrays["train_probs"])
    metadata = json.loads((root / "manifest.json").read_text())
    metadata = {
        k: v for k, v in metadata.items() if k not in ("schema", "scope", "arrays")
    }
    np.save(root / "val_ids.npy", np.arange(20))
    with pytest.raises(ValueError, match="overlap"):
        seal_evidence(root, metadata)
    np.save(root / "val_ids.npy", arrays["val_ids"])
    np.save(root / "meta_val_rows.npy", arrays["meta_train_rows"][:10])
    with pytest.raises(ValueError, match="Meta partition"):
        seal_evidence(root, metadata)


def test_noisy_validation_policy_and_undefined_auroc():
    p = np.array([[0.6, 0.4], [0.6, 0.4]], dtype=np.float32)
    h = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    assert select_alpha(p, h, np.array([0, 0]), [0.0, 1.0])[0]["alpha"] == 0
    assert select_alpha(p, h, np.array([1, 1]), [0.0, 1.0])[0]["alpha"] == 1
    assert (
        evaluate_probabilities(np.array([0, 0]), p, p)["identity"]["own_error_auroc"]
        is None
    )
    with pytest.raises(ValueError, match="identity"):
        select_alpha(p, h, [1, 1], [0.5, 1.0])


def test_fit_saves_min_kl_freezes_policy_and_binds_inputs(tmp_path):
    root = tmp_path / "cache"
    synthetic_cache(root)
    cfg = fit_config(root)
    out = tmp_path / "run"
    metrics = fit(cfg, out, torch.device("cpu"))
    assert metrics["best_meta_val_kl"] < metrics["initial_meta_val_kl"] * 0.4
    assert not metrics["test_evaluated"]
    head, policy = load_frozen(out, root, torch.device("cpu"))
    checkpoint = torch.load(out / "checkpoints/best.pt", weights_only=False)
    history = np.genfromtxt(out / "tables/training.csv", delimiter=",", names=True)
    assert checkpoint["val_loss"] == min(
        metrics["initial_meta_val_kl"], history["meta_val_kl"].min()
    )
    assert checkpoint["epoch"] == metrics["best_epoch"]
    assert policy["test_accessed"] is False
    assert policy["policy"]["alpha"] in cfg.alphas
    for name in ("config.yaml", "metrics.json", "run_metadata.json", "report.md"):
        assert (out / name).exists()
    with pytest.raises(ValueError, match="already exists"):
        fit(cfg, out, torch.device("cpu"))
    path = out / "artifacts/policy_before_test.json"
    value = json.loads(path.read_text())
    value["policy"]["alpha"] = 0.123
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_frozen(out, root, torch.device("cpu"))


def test_snapshots_are_opt_in_real_hashed_weights_and_resume_safe(tmp_path):
    config = dict(seed=42, epochs=2, max_hours=1, temporal_snapshot_epochs=[1, 2])
    assert resume_contract(dict(seed=42, epochs=2)) == resume_contract(
        dict(seed=42, epochs=2, temporal_snapshot_epochs=[])
    )
    model = torch.nn.Linear(3, 2)
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    save_snapshot(model, config, 0, ckpts, atomic_save)
    config["max_hours"] = 2  # Allowed execution-only resume change.
    save_snapshot(model, config, 0, ckpts, atomic_save)
    save_snapshot(model, config, 1, ckpts, atomic_save)
    rows = verify_snapshots(tmp_path, config, [1, 2])
    assert [r["epoch"] for r in rows] == [1, 2]
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(ValueError, match="differs"):
        save_snapshot(model, config, 1, ckpts, atomic_save)
    path = Path(rows[0]["path"])
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        verify_snapshots(tmp_path, config, [1, 2])
    with pytest.raises(ValueError, match="sorted unique"):
        snapshot_epochs(dict(epochs=2, temporal_snapshot_epochs=[2, 1]))


def test_canonical_pipeline_freezes_before_test_and_recovers_interruption(
    tmp_path, monkeypatch
):
    """Real tiny weights/images through extraction; dataset/base loader mocked only."""
    import room_for_doubt.analysis.temporal_distillation as pipeline
    from room_for_doubt.training.temporal_snapshots import file_digest

    events = []
    rng = np.random.default_rng(17)
    images = rng.integers(0, 255, (50, 8, 8, 3), dtype=np.uint8)
    noisy = np.arange(50, dtype=np.int64) % 3

    class Raw:
        data = images

        @property
        def targets(self):
            raise AssertionError("Clean training labels must never be used")

    class TestRaw:
        def __init__(self, root, train, download):
            assert train is False
            assert (tmp_path / "run/artifacts/policy_before_test.json").exists()
            events.append("test_access")
            self.data, self.targets = images[:12], noisy[:12]

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x.mean((2, 3)))

    source = tmp_path / "source"
    ckpts = source / "checkpoints"
    ckpts.mkdir(parents=True)
    config = dict(
        seed=42,
        epochs=2,
        image_size=8,
        split_seed=20261015,
        temporal_snapshot_epochs=[1, 2],
        defer_test_evaluation=True,
        dataset=dict(
            name="synthetic", num_classes=3, noisy_split="synthetic", val_ratio=0.2
        ),
    )
    torch.manual_seed(9)
    model = Tiny()
    save_snapshot(model, config, 0, ckpts, atomic_save)
    with torch.no_grad():
        model.fc.weight.add_(torch.eye(3) * 0.2)
    save_snapshot(model, config, 1, ckpts, atomic_save)
    selected = {k: v.clone() for k, v in model.state_dict().items()}
    atomic_save(dict(state_dict=selected), ckpts / "best.pt")
    train_ids, val_ids = np.arange(40), np.arange(40, 50)
    bundle = dict(
        source=source,
        config=config,
        checkpoint=dict(state_dict=selected),
        checkpoint_digest=file_digest(ckpts / "best.pt"),
        data=dict(
            base_train_ids=train_ids,
            base_val_ids=val_ids,
            **reserve_meta_splits(train_ids, config["split_seed"]),
        ),
    )

    def restore(bundle, device):
        value = Tiny()
        value.load_state_dict(bundle["checkpoint"]["state_dict"])
        return value.to(device).eval()

    monkeypatch.setattr(
        pipeline, "source_data", lambda cfg: (bundle, Raw(), noisy, TestRaw)
    )
    monkeypatch.setattr(pipeline, "restore_base", restore)
    cfg = fit_config(tmp_path / "cache")
    cfg.teacher_epochs = [1, 2]
    cfg.epochs = 2
    cfg.image_batch_size = 7
    cfg.data_dir = str(tmp_path / "images")
    out, device = tmp_path / "run", torch.device("cpu")
    assert pipeline.prepare(cfg, out, device)["test_accessed"] is False
    assert not events
    assert pipeline.prepare(cfg, out, device)["status"] == "cache_already_verified"
    fit(cfg, out, device)
    assert not events

    # An updated source manifest must not silently replace the sealed teacher.
    original_verify = pipeline.verify_snapshots

    def drifted(*args):
        rows = original_verify(*args)
        rows[0]["sha256"] = "different-teacher"
        return rows

    monkeypatch.setattr(pipeline, "verify_snapshots", drifted)
    with pytest.raises(ValueError, match="sealed training evidence"):
        pipeline.final_evaluate(cfg, out, device)
    assert not events
    monkeypatch.setattr(pipeline, "verify_snapshots", original_verify)

    original_extract = pipeline.extract

    def interrupted(*args):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(pipeline, "extract", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.final_evaluate(cfg, out, device)
    assert not (out / "final_evaluation/metrics.json").exists()
    monkeypatch.setattr(pipeline, "extract", original_extract)
    result = pipeline.final_evaluate(cfg, out, device)
    assert result["policy_selected_before_test"] and result["status"] == "complete"
    assert "snapshot_mean_2passes" in result["methods"]
    for name in (
        "report.md",
        "config.yaml",
        "metrics.json",
        "run_metadata.json",
        "predictions.npz",
        "tables/methods.csv",
    ):
        assert (out / "final_evaluation" / name).exists()
    with pytest.raises(ValueError, match="already exists"):
        pipeline.final_evaluate(cfg, out, device)


def test_source_data_verifies_actual_runtime_data_root(tmp_path, monkeypatch):
    import room_for_doubt.analysis.temporal_distillation as pipeline

    root = tmp_path / "source"
    root.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            dict(
                data_dir="/old/unavailable/path",
                dataset=dict(name="animal10n", noisy_split="noisy"),
            )
        ),
        root / "config.yaml",
    )
    checked = []
    monkeypatch.setattr(pipeline, "Animal10N", lambda *args, **kw: object())
    monkeypatch.setattr(
        pipeline, "verify_identity", lambda raw, cfg: checked.append(cfg["data_dir"])
    )
    monkeypatch.setattr(pipeline, "load_observed_labels", lambda *args: np.arange(10))
    monkeypatch.setattr(pipeline, "load_completed_run", lambda *args: {})
    cfg = OmegaConf.create(
        dict(source_run=str(root), data_dir=str(tmp_path / "actual"))
    )
    pipeline.source_data(cfg)
    assert checked == [str(tmp_path / "actual")]
