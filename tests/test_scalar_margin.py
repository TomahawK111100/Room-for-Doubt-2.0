import numpy as np
import pytest
import torch
from room_for_doubt.analysis.target_ablation import regression_targets
from room_for_doubt.models.scalar_margin import ScalarMarginPredictor


def test_regression_targets_preserve_samples_and_epoch_mean():
    x = np.arange(100, dtype=np.float32).reshape(2, 50)
    assert np.array_equal(regression_targets(x, "last"), np.repeat(x[:, -1:], 50, 1))
    assert np.array_equal(
        regression_targets(x, "mean"), np.repeat(x.mean(1, keepdims=True), 50, 1)
    )
    with pytest.raises(ValueError):
        regression_targets(x, "unknown")


def test_scalar_expansion_has_exact_scalar_mse_and_gradient():
    torch.manual_seed(42)
    model = ScalarMarginPredictor(100)
    features = torch.randn(3, 512)
    labels = torch.tensor([0, 50, 99])
    output = model.predict_from_features(features, labels)
    assert output.shape == (3, 50) and torch.equal(
        output, output[:, :1].expand_as(output)
    )
    target = torch.tensor([0.1, 0.2, 0.3])
    expanded = ((output - target[:, None]) ** 2).mean()
    direct = ((output[:, 0] - target) ** 2).mean()
    assert torch.allclose(expanded, direct)
    expanded.backward()
    assert model.mlp[-1].weight.grad.shape == (1, 256)
    assert all(p.grad is None for p in model.features.parameters())
