import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import roc_auc_score

from tests.test_post_kaggle import source, TinyBase  # noqa: F401
from room_for_doubt.evaluation.paired_targets import paired_auroc_interval

ROOT = Path(__file__).parents[1]


def test_paired_auc_matches_explicit_resampling_with_ties():
    y = np.array([0, 0, 0, 1, 1, 1])
    a = np.array([0.2, 0.2, 0.8, 0.5, 0.5, 0.8])
    b = np.array([0.1, 0.3, 0.3, 0.2, 0.6, 0.9])
    rng, draws = np.random.default_rng(9), []
    for _ in range(2):
        weights = np.zeros((16, 6), dtype=int)
        weights[:, 3:] = rng.multinomial(3, [1 / 3] * 3, 16)
        weights[:, :3] = rng.multinomial(3, [1 / 3] * 3, 16)
        for w in weights:
            indices = np.repeat(np.arange(6), w)
            draws.append(
                roc_auc_score(y[indices], b[indices])
                - roc_auc_score(y[indices], a[indices])
            )
    result = paired_auroc_interval(y, a, b, draws=32, seed=9)
    assert result["auroc_difference"] == pytest.approx(
        roc_auc_score(y, b) - roc_auc_score(y, a)
    )
    assert [result["auroc_ci_low"], result["auroc_ci_high"]] == pytest.approx(
        np.quantile(draws, [0.025, 0.975])
    )
    same = paired_auroc_interval(y, a, a, draws=32)
    assert set(same.values()) == {0.0}
    with pytest.raises(ValueError, match="both error"):
        paired_auroc_interval(np.zeros(6), a, b)
    with pytest.raises(ValueError, match="aligned"):
        paired_auroc_interval(y, a[:-1], b)


def test_augmented_target_bounds_resume_rng_and_observed_labels(
    source, tmp_path, monkeypatch  # noqa: F811
):  # noqa: F811
    import room_for_doubt.inference.augmented_margin as module
    from room_for_doubt.data.kaggle_artifacts import load_completed_run

    root, noisy = source
    bundle = load_completed_run(root, noisy)
    model = TinyBase()
    batches = []
    model.register_forward_pre_hook(lambda module, args: batches.append(len(args[0])))
    monkeypatch.setattr(module, "restore_base", lambda *args: model)
    images = np.random.default_rng(1).integers(0, 255, (40, 32, 32, 3), dtype="uint8")
    state = torch.random.get_rng_state()
    with pytest.raises(TimeoutError):
        module.augmented_margin(
            bundle, images, tmp_path / "partial", 5, 19, 64, "cpu", deadline=-1
        )
    assert torch.equal(state, torch.random.get_rng_state())
    actual = module.augmented_margin(
        bundle, images, tmp_path / "partial", 5, 19, 64, "cpu"
    )
    expected = module.augmented_margin(
        bundle, images, tmp_path / "fresh", 5, 19, 64, "cpu"
    )
    assert np.array_equal(actual, expected) and max(batches) <= 64
    p = model.fc.bias.detach().softmax(0).numpy()
    manual = np.array(
        [p[y] - np.delete(p, y).max() for y in bundle["data"]["noisy_labels"]]
    )
    assert np.allclose(actual, manual)
    with pytest.raises(ValueError, match="protocol differs"):
        module.augmented_margin(bundle, images, tmp_path / "partial", 5, 20, 64, "cpu")


@pytest.fixture
def complete_features(source, tmp_path, monkeypatch):  # noqa: F811
    import room_for_doubt.analysis.feature_study as runner
    import room_for_doubt.data.kaggle_artifacts as source_tools
    import room_for_doubt.inference.post_kaggle_cache as shared_cache

    base, noisy = source
    study = tmp_path / "original_study"

    class Images:
        def __init__(self, root, train, download):
            self.train = train
            self.data = np.random.default_rng(8).integers(
                0, 255, (40 if train else 8, 32, 32, 3), dtype="uint8"
            )

        @property
        def targets(self):
            assert (study / "artifacts/policies_before_test.json").exists()
            return np.arange(40 if self.train else 8) % 2

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Identity()

        def extract_features(self, x):
            return x.mean((1, 2, 3))[:, None].expand(-1, 512)

    monkeypatch.setattr(runner, "CIFAR100", Images)
    monkeypatch.setattr(runner, "load_observed_labels", lambda *args: noisy)
    monkeypatch.setattr(
        source_tools,
        "make_pretrained_classifier",
        lambda classes, weights: TinyBase(classes),
    )
    monkeypatch.setattr(shared_cache, "frozen_encoder", lambda *args: Encoder())
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="feature_study")
    cfg.source_run, cfg.device = str(base), "cpu"
    cfg.epochs, cfg.seeds, cfg.batch_size = 2, [42], 16
    cfg.view_counts, cfg.temperatures, cfg.feature_dim = [1, 2], [0.1, 1.0], 2
    cfg.min_free_gb = 0
    assert runner.run_study(cfg, study)["status"] == "complete"
    return base, noisy, study


def test_direct_contrasts_and_augmented_control_end_to_end(
    complete_features, tmp_path, monkeypatch
):
    import room_for_doubt.analysis.feature_contrasts as contrasts
    import room_for_doubt.analysis.augmentation_control as control

    base, noisy, study = complete_features
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="feature_contrasts")
        launch = compose(config_name="augmentation_control")
    cfg.study_run, cfg.auroc_bootstrap_draws = str(study), 32
    metrics = contrasts.run_contrasts(cfg, tmp_path / "contrasts")
    assert metrics["comparisons"] == 4
    assert (tmp_path / "contrasts/artifacts/paired_selected_predictions.npz").exists()
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in study.rglob("*")
        if p.is_file()
    }
    out = tmp_path / "control"

    class ObservedOnly:
        def __init__(self, root, train, download):
            assert train
            self.data = np.random.default_rng(8).integers(
                0, 255, (40, 32, 32, 3), dtype="uint8"
            )

        @property
        def targets(self):
            raise AssertionError("No clean training-label access allowed")

    original_split = control.shared_split

    def guarded_split(study, post, cfg, split):
        if split == "test":
            assert (out / "artifacts/policies_before_test.json").exists()
        return original_split(study, post, cfg, split)

    monkeypatch.setattr(control, "CIFAR100", ObservedOnly)
    monkeypatch.setattr(control, "load_observed_labels", lambda *args: noisy)
    monkeypatch.setattr(control, "shared_split", guarded_split)
    launch.study_run, launch.source_run, launch.device = str(study), str(base), "cpu"
    launch.min_free_gb, launch.auroc_bootstrap_draws = 0, 32
    result = control.run_control(launch, out)
    assert result["status"] == "complete" and result["original_heads_reused"] == 3
    assert result["new_head_count"] == 1 and result["paired_comparisons"] == 3
    assert len(list((out / "heads").iterdir())) == 4
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items()
    )
    repeated = control.run_control(launch, out)
    assert repeated["selected_results"] == result["selected_results"]
    launch.augmentation_seed += 1
    with pytest.raises(ValueError, match="scientific configuration differs"):
        control.run_control(launch, out)


def test_cache_id_permutation_rejected(complete_features):
    from room_for_doubt.data.feature_artifacts import completed_study, shared_split

    _, _, study = complete_features
    cfg, _ = completed_study(study)
    path = study / "shared/cache/test_ids.npy"
    np.save(path, np.load(path)[::-1])
    with pytest.raises(ValueError, match="IDs or provenance differ"):
        shared_split(study, study / "shared", cfg, "test")
