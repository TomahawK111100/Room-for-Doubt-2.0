"""Synthetic smoke fixtures, never publication measurements."""

import numpy as np
import pytest
import torch
from torch import nn
from room_for_doubt.models import conditioned_trajectory, flow_matching
from room_for_doubt.analysis.matched_lda import (
    paired_representations,
    evaluate_matched_lda,
)


class TinyBackbone(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 512, 1),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(512, 1000),
        )


@pytest.mark.parametrize(
    "module,model_name",
    [
        (conditioned_trajectory, "ConditionedPredictor"),
        (flow_matching, "CFMTrajectoryModel"),
    ],
)
def test_training_freezes_backbone_buffers_but_updates_head(
    monkeypatch, module, model_name
):
    monkeypatch.setattr(module, "resnet18", lambda **kwargs: TinyBackbone())
    model = getattr(module, model_name)(num_classes=10)
    before = {k: v.clone() for k, v in model.features.state_dict().items()}
    model.train()
    assert not model.features.training
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    images = torch.randn(2, 3, 8, 8)
    classes = torch.tensor([0, 1])
    if model_name == "ConditionedPredictor":
        output = model(images, classes)
    else:
        output = model.vector_field(
            torch.randn(2, 50), torch.rand(2), model.extract_features(images), classes
        )
    output.square().mean().backward()
    optimizer.step()
    assert all(p.grad is None for p in model.features.parameters())
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    for k, v in model.features.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_paired_features_use_one_backbone_forward(monkeypatch):
    monkeypatch.setattr(
        conditioned_trajectory, "resnet18", lambda **kwargs: TinyBackbone()
    )
    model = conditioned_trajectory.ConditionedPredictor().train()
    calls = []
    hook = model.features.register_forward_hook(lambda *args: calls.append(1))
    labels = torch.tensor([0, 1])
    static, dynamic = paired_representations(model, torch.randn(2, 3, 8, 8), labels)
    hook.remove()
    assert len(calls) == 1
    with torch.no_grad():
        expected = model.predict_from_features(
            torch.from_numpy(static[:, :512]), labels
        ).numpy()
    np.testing.assert_allclose(dynamic, expected)
    assert static.shape == (2, 544) and dynamic.shape == (2, 50)


def arrays():
    rng = np.random.default_rng(42)
    labels = np.tile([0, 1], 12)
    static = rng.normal(size=(24, 4))
    dynamic = rng.normal(size=(24, 3))
    return (
        static,
        dynamic,
        labels,
        np.arange(24),
        np.arange(12),
        np.arange(12, 24),
        np.arange(100, 110),
    )


def test_matched_lda_uses_same_heldout_counts():
    result = evaluate_matched_lda(*arrays())
    assert set(result) == {"static_shared", "predicted_traj"}
    assert all(r["fit_n"] == 12 and r["eval_n"] == 12 for r in result.values())


def test_rejects_meta_training_overlap():
    a = list(arrays())
    a[-1] = np.array([0])
    with pytest.raises(ValueError, match="overlap meta-model"):
        evaluate_matched_lda(*a)


def test_rejects_detector_train_evaluation_overlap():
    a = list(arrays())
    a[-2] = np.arange(11, 24)
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_matched_lda(*a)
