import json
from pathlib import Path
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tests.test_post_kaggle import source, TinyBase  # noqa: F401
from room_for_doubt.data.kaggle_artifacts import load_completed_run
from room_for_doubt.inference.classifier_features import (
    fit_projection,
    project_features,
)
from room_for_doubt.models.feature_margin import FeatureMarginHead

ROOT = Path(__file__).parents[1]


def test_pca_fits_only_reserved_training_rows_and_heads_match(tmp_path):
    f = np.random.default_rng(7).normal(size=(30, 8)).astype("float32")
    fit = np.arange(20)
    a = fit_projection(f, fit, tmp_path / "a.npz", 4, 42)
    changed = f.copy()
    changed[20:] += 10000
    b = fit_projection(changed, fit, tmp_path / "b.npz", 4, 42)
    assert np.array_equal(a["mean"], b["mean"])
    assert np.array_equal(a["components"], b["components"])
    projected = project_features(f, a, tmp_path / "projected.npy")
    assert projected.shape == (30, 4)
    assert np.allclose(projected, (f - a["mean"]) @ a["components"].T)
    torch.manual_seed(42)
    left = FeatureMarginHead(10, 4)
    torch.manual_seed(42)
    right = FeatureMarginHead(10, 4)
    assert all(
        torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items()
    )
    assert not any(name.startswith("features.") for name in left.state_dict())
    assert left.predict_from_features(
        torch.zeros(3, 4), torch.zeros(3, dtype=torch.long)
    ).shape == (3, 50)


def test_joint_validation_policy_ties_prefer_fewer_views(tmp_path, monkeypatch):
    import room_for_doubt.evaluation.feature_study as evaluation

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "tables").mkdir()
    cfg = OmegaConf.create(
        dict(view_counts=[1, 2], temperatures=[0.1, 1.0], base_digest="source")
    )
    p = np.array([[[0.8, 0.2], [0.7, 0.3]], [[0.2, 0.8], [0.3, 0.7]]])
    monkeypatch.setattr(evaluation, "score_heads", lambda *args: {"msp": 1 - p.max(-1)})
    frozen = evaluation.freeze_policies(cfg, tmp_path, p, np.array([0, 1]), {}, "cpu")
    assert frozen["selected"]["msp"]["views"] == 1
    assert frozen["selected"]["msp"]["temperature"] == "uniform"
    assert frozen["selected"]["tta"]["views"] == 1


@pytest.mark.parametrize("supplied_post", [True, False])
def test_feature_study_strict_hydra_end_to_end_no_clean_labels_before_freeze(
    source,  # noqa: F811
    tmp_path,
    monkeypatch,
    supplied_post,
):
    import room_for_doubt.analysis.feature_study as runner
    import room_for_doubt.data.kaggle_artifacts as source_tools
    import room_for_doubt.inference.post_kaggle_cache as shared_cache

    base, noisy = source
    post, out = tmp_path / "post", tmp_path / "features_study"
    for name in ("artifacts", "features", "cache"):
        (post / name).mkdir(parents=True)
    bundle = load_completed_run(base, noisy)
    data, digest = bundle["data"], bundle["checkpoint_digest"]
    post_cfg = OmegaConf.load(ROOT / "configs/post_kaggle.yaml")
    post_cfg.seeds, post_cfg.epochs, post_cfg.batch_size = [42], 2, 16
    post_cfg.views, post_cfg.temperatures = 2, [0.1, 1.0]
    OmegaConf.save(post_cfg, post / "config.yaml")
    (post / "artifacts/source.json").write_text(json.dumps(dict(base_digest=digest)))
    (post / "artifacts/all_policies_before_test.json").write_text(
        json.dumps(dict(base_digest=digest))
    )
    (post / "metrics.json").write_text('{"status":"complete"}')
    rng = np.random.default_rng(5)
    np.savez(
        post / "features/features.npz",
        sample_ids=data["sample_ids"],
        dataset_name=np.array("cifar100n"),
        preprocessing=np.array("imagenet_pil_bilinear_224"),
        features=rng.normal(size=(len(data["sample_ids"]), 512)).astype("float32"),
    )
    for split, ids in (("val", data["base_val_ids"]), ("test", np.arange(8))):
        protocol = dict(
            base_digest=digest, views=2, seed=20261015, image_size=8, ids=ids.tolist()
        )
        (post / "cache" / f"{split}_complete.json").write_text(json.dumps(protocol))
        p = torch.softmax(TinyBase().fc.bias.detach(), 0).numpy()
        np.save(
            post / "cache" / f"{split}_probs.npy",
            np.broadcast_to(p, (len(ids), 2, 100)).copy(),
        )
        np.save(
            post / "cache" / f"{split}_features.npy",
            rng.normal(size=(len(ids), 2, 512)).astype("float32"),
        )
        np.save(post / "cache" / f"{split}_ids.npy", ids)
        np.save(
            post / "cache" / f"{split}_labels.npy",
            noisy[ids] if split == "val" else ids % 2,
        )

    class Images:
        def __init__(self, root, train, download):
            self.train = train
            if not train:
                assert (out / "artifacts/policies_before_test.json").exists()
            self.data = np.random.default_rng(8).integers(
                0, 255, (40 if train else 8, 32, 32, 3), dtype="uint8"
            )

        @property
        def targets(self):
            assert (out / "artifacts/policies_before_test.json").exists()
            return np.arange(40 if self.train else 8) % 2

    monkeypatch.setattr(runner, "CIFAR100", Images)
    monkeypatch.setattr(runner, "load_observed_labels", lambda *args: noisy)
    monkeypatch.setattr(
        source_tools,
        "make_pretrained_classifier",
        lambda classes, weights: TinyBase(classes),
    )

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Identity()

        def extract_features(self, x):
            return x.mean((1, 2, 3))[:, None].expand(-1, 512)

    monkeypatch.setattr(shared_cache, "frozen_encoder", lambda *args: TinyEncoder())
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="feature_study")
    assert OmegaConf.is_struct(cfg)
    cfg.source_run, cfg.post_run, cfg.device = (
        str(base),
        str(post) if supplied_post else None,
        "cpu",
    )
    cfg.epochs, cfg.batch_size, cfg.seeds = 2, 16, [42]
    cfg.view_counts, cfg.temperatures, cfg.feature_dim = [1, 2], [0.1, 1.0], 2
    cfg.min_free_gb, cfg.max_hours = 0, 1e-12
    first = runner.run_study(cfg, out)
    assert first["status"] == "paused_time_budget" and not first["test_evaluated"]
    cfg.max_hours = 8
    result = runner.run_study(cfg, out)
    assert result["status"] == "complete" and result["heads_trained_or_reused"] == 6
    assert len(result["selected_results"]) == 11
    assert (out / "tables/real_history_epoch_noise.csv").exists()
    assert (out / "summary_bundle.zip").exists()
    assert all("ci_low_pp_vs_base" in row for row in result["selected_results"])
    repeated = runner.run_study(cfg, out)
    assert repeated["selected_results"] == result["selected_results"]
    cfg.pca_seed += 1
    with pytest.raises(ValueError, match="scientific configuration differs"):
        runner.run_study(cfg, out)
