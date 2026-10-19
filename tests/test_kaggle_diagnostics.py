import json
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tests.test_post_kaggle import source, TinyBase  # noqa: F401
from room_for_doubt.evaluation.diagnostic_metrics import (
    class_average,
    regression_metrics,
    probability_metrics,
    static_badness,
)
from room_for_doubt.inference.diagnostic_cache import align_saved_probabilities
from room_for_doubt.training.diagnostic_controls import control_target

ROOT = Path(__file__).parents[1]


def test_hydra_strict_config_declares_derived_source_fields():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="diagnose_kaggle")
    assert OmegaConf.is_struct(cfg)
    cfg.dataset = dict(name="cifar100n", num_classes=100)
    cfg.base_digest = "test-source"
    with pytest.raises(Exception, match="not in struct"):
        cfg.undeclared_field = 1


def test_metrics_controls_and_probability_alignment(tmp_path):
    labels = np.array([0, 0, 1, 1])
    values = np.array([[1.0], [3.0], [100.0], [200.0]])
    means = class_average(values, labels, np.array([0, 2]), 3)
    assert np.array_equal(means[:, 0], [1, 100, 50.5])
    assert regression_metrics([1, 2, 3], [1, 2, 3])["r2"] == 1
    assert regression_metrics([1, 1], [2, 2])["correlation"] is None
    p = np.array([[0.8, 0.2], [0.3, 0.7]])
    m = probability_metrics([0, 1], p)
    assert m["acc"] == 1 and m["brier"] == pytest.approx(0.13)
    assert np.allclose(static_badness(p, "gap"), [0.4, 0.6])
    with pytest.raises(ValueError, match="distributions"):
        probability_metrics([0, 1], p * 2)
    path = tmp_path / "p.npz"
    np.savez(path, ids=[8, 3], probabilities=p)
    assert np.array_equal(align_saved_probabilities(path, np.array([3, 8])), p[::-1])
    with pytest.raises(ValueError, match="IDs differ"):
        align_saved_probabilities(path, np.array([3, 9]))
    data = {"targets": np.arange(150).reshape(3, 50)}
    assert np.all(control_target(data, [3, 4, 5], "best_margin", 5)[:, 0] == [3, 4, 5])
    assert np.all(control_target(data, None, "early_mean", 5)[:, 0] == [2, 52, 102])


def test_unattended_diagnostics_freezes_before_clean_labels_and_resumes(
    source,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    import room_for_doubt.analysis.kaggle_diagnostics as runner
    import room_for_doubt.data.kaggle_artifacts as source_tools
    import room_for_doubt.training.diagnostic_controls as controls
    import room_for_doubt.evaluation.diagnostic_study as study
    from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor

    base, noisy = source
    post, out = tmp_path / "post", tmp_path / "diagnostic"
    for name in ("artifacts", "features", "cache"):
        (post / name).mkdir(parents=True)
    post_cfg = OmegaConf.load(ROOT / "configs/post_kaggle.yaml")
    post_cfg.source_run, post_cfg.device = str(base), "cpu"
    post_cfg.seeds, post_cfg.epochs, post_cfg.batch_size = [42], 2, 16
    post_cfg.views, post_cfg.temperatures = 2, [0.1, 1.0]
    OmegaConf.save(post_cfg, post / "config.yaml")
    bundle = source_tools.load_completed_run(base, noisy)
    digest = bundle["checkpoint_digest"]
    (post / "artifacts/source.json").write_text(json.dumps(dict(base_digest=digest)))
    (post / "artifacts/all_policies_before_test.json").write_text(
        json.dumps(dict(base_digest=digest))
    )
    (post / "metrics.json").write_text(json.dumps(dict(status="complete")))
    data = bundle["data"]
    np.savez(
        post / "features/features.npz",
        sample_ids=data["sample_ids"],
        dataset_name=np.array("cifar100n"),
        preprocessing=np.array("imagenet_pil_bilinear_224"),
        features=np.random.default_rng(2)
        .normal(size=(len(data["sample_ids"]), 512))
        .astype("float32"),
    )
    torch.save({}, post / "features/backbone.pt")
    for split, ids in (("val", data["base_val_ids"]), ("test", np.arange(8))):
        protocol = dict(
            base_digest=digest, views=2, seed=20261015, image_size=8, ids=ids.tolist()
        )
        (post / "cache" / f"{split}_complete.json").write_text(json.dumps(protocol))
        p = np.full((len(ids), 2, 100), 0.01, dtype="float32")
        p[:, :, 0] += 0.5
        p /= p.sum(2, keepdims=True)
        np.save(post / "cache" / f"{split}_probs.npy", p)
        np.save(
            post / "cache" / f"{split}_features.npy",
            np.zeros((len(ids), 2, 512), dtype="float32"),
        )
        np.save(post / "cache" / f"{split}_ids.npy", ids)
        np.save(
            post / "cache" / f"{split}_labels.npy",
            noisy[ids] if split == "val" else ids % 2,
        )
        for mode in ("mean", "full"):
            np.save(
                post / "artifacts" / f"{split}_badness_{mode}_42.npy",
                np.full((len(ids), 2), 0.5),
            )

    class Images:
        def __init__(self, root, train, download):
            self.train = train
            if not train:
                assert (out / "artifacts/diagnostic_policies_before_test.json").exists()
            self.data = np.random.default_rng(5).integers(
                0, 255, (40 if train else 8, 32, 32, 3), dtype="uint8"
            )

        @property
        def targets(self):
            assert (out / "artifacts/diagnostic_policies_before_test.json").exists(), (
                "Clean labels accessed before policies"
            )
            return np.arange(40 if self.train else 8) % 2

    class TinyHead(ConditionedPredictor):
        def __init__(self, classes, pretrained=False, outputs=1):
            torch.nn.Module.__init__(self)
            self.features = torch.nn.Identity()
            self.class_emb = torch.nn.Embedding(classes, 32)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(544, 16), torch.nn.ReLU(), torch.nn.Linear(16, outputs)
            )

        def predict_from_features(self, features, classes):
            return super().predict_from_features(features, classes).expand(-1, 50)

    def existing_head(*args):
        torch.manual_seed(10)
        return TinyHead(100).eval()

    monkeypatch.setattr(runner, "CIFAR100", Images)
    monkeypatch.setattr(runner, "load_observed_labels", lambda *args: noisy)
    monkeypatch.setattr(
        source_tools,
        "make_pretrained_classifier",
        lambda classes, weights: TinyBase(classes),
    )
    monkeypatch.setattr(controls, "ScalarMarginPredictor", TinyHead)
    monkeypatch.setattr(study, "load_head", existing_head)
    cfg = OmegaConf.load(ROOT / "configs/diagnose_kaggle.yaml")
    OmegaConf.set_struct(cfg, True)
    cfg.source_run, cfg.post_run, cfg.device = str(base), str(post), "cpu"
    cfg.epochs, cfg.batch_size, cfg.seeds = 2, 16, [42]
    cfg.view_counts, cfg.temperatures = [1, 2], [0.1, 1.0]
    cfg.min_free_gb, cfg.image_batch_size = 0, 8
    cfg.max_hours = 1e-12
    paused = runner.run_diagnostics(cfg, out)
    assert paused["status"] == "paused_time_budget" and not paused["test_evaluated"]
    assert (out / "summary_bundle.zip").exists()
    cfg.max_hours = 8
    result = runner.run_diagnostics(cfg, out)
    assert result["status"] == "complete" and result["controls_trained_or_reused"] == 3
    assert len(result["test_rows"]) == 22
    assert all(
        k in result["test_rows"][0]
        for k in ("nll", "brier", "ci_low_pp_vs_base", "ci_high_pp_vs_msp")
    )
    assert (out / "tables/reconstruction.csv").exists()
    assert (out / "tables/train_noise_diagnostic.csv").exists()
    assert (out / "tables/numerical_audit.csv").exists()
    repeated = runner.run_diagnostics(cfg, out)
    assert repeated["test_rows"] == result["test_rows"]
    cfg.early_epochs = 6
    with pytest.raises(ValueError, match="scientific configuration differs"):
        runner.run_diagnostics(cfg, out)


def test_input_discovery_requires_completed_post_and_base(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "bootstrap_diag", ROOT / "scripts/kaggle_diagnose.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="post-kaggle"):
        module.find_inputs([tmp_path])
    post, base = tmp_path / "post", tmp_path / "base"
    for folder in (
        post / "artifacts",
        post / "cache",
        post / "features",
        base / "checkpoints",
    ):
        folder.mkdir(parents=True)
    (post / "artifacts/all_policies_before_test.json").write_text("{}")
    (post / "cache/test_complete.json").write_text("{}")
    (post / "features/backbone.pt").write_bytes(b"")
    (post / "metrics.json").write_text('{"status":"complete"}')
    (base / "checkpoints/best.pt").write_bytes(b"")
    OmegaConf.save(OmegaConf.create(dict(source_run=str(base))), post / "config.yaml")
    assert module.find_inputs([tmp_path]) == (base, post)
