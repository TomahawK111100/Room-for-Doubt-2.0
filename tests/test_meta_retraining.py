import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from room_for_doubt.data.meta_splits import reserve_meta_splits, validate_meta_splits
from room_for_doubt.models import conditioned_trajectory, flow_matching
from room_for_doubt.training.meta_retraining import fit_head
from tests.test_frozen_backbone_and_ablation import TinyBackbone


def test_reserved_splits_are_reproducible_and_disjoint():
    ids = np.arange(100)
    splits = reserve_meta_splits(ids, 42)
    validate_meta_splits(splits, ids)
    assert [len(v) for v in splits.values()] == [60, 10, 15, 15]
    for k, v in splits.items():
        np.testing.assert_array_equal(v, reserve_meta_splits(ids, 42)[k])
    splits["detector_fit_ids"][0] = splits["meta_train_ids"][0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_meta_splits(splits, ids)


@pytest.mark.parametrize(
    "kind,module,cls",
    [
        ("conditioned", conditioned_trajectory, "ConditionedPredictor"),
        ("cfm", flow_matching, "CFMTrajectoryModel"),
    ],
)
@pytest.mark.parametrize("validation_metric", ["native", "trajectory_mse"])
def test_fit_selects_best_and_can_resume(
    tmp_path, monkeypatch, kind, module, cls, validation_metric
):
    monkeypatch.setattr(module, "resnet18", lambda **kwargs: TinyBackbone())
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = getattr(module, cls)(num_classes=10, pretrained=False)
    ids = np.arange(40)
    data = dict(
        sample_ids=ids,
        noisy_labels=np.arange(40) % 10,
        targets=np.zeros((40, 50), dtype=np.float32),
        **reserve_meta_splits(ids, 42),
    )
    features = np.ones((40, 512), dtype=np.float32)
    cfg = OmegaConf.create(
        dict(
            epochs=2,
            batch_size=8,
            learning_rate=0.001,
            meta_model=kind,
            split_seed=42,
            seed=1,
            resume_path=None,
            target_path="synthetic",
            cache_path="synthetic",
            validation_metric=validation_metric,
            validation_flow_samples=2,
            validation_flow_steps=2,
        )
    )
    before = {k: v.clone() for k, v in model.features.state_dict().items()}
    metrics = fit_head(model, data, features, cfg, tmp_path, torch.device("cpu"))
    best = torch.load(tmp_path / "checkpoints/best.pt", weights_only=False)
    assert best["val_loss"] == metrics["best_val_loss"]
    assert best["epoch"] == metrics["best_epoch"]
    for k, v in model.features.state_dict().items():
        assert torch.equal(v, before[k])
    cfg.resume_path = str(tmp_path / "checkpoints/last.pt")
    resumed = tmp_path / "resumed"
    result = fit_head(model, data, features, cfg, resumed, torch.device("cpu"))
    assert result == metrics
    assert (resumed / "checkpoints/best.pt").exists()
