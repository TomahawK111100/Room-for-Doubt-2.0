import numpy as np
import pytest
import torch

from room_for_doubt.evaluation.metrics import calculate_aurc
from room_for_doubt.inference.flow_sampling import generate_trajectories


def test_aurc_uses_descending_confidence():
    # Sorted errors: 0, 1, 1. Risks: 0, 1/2, 2/3.
    assert calculate_aurc([0.3, 0.9, 0.6], [1, 0, 1]) == pytest.approx(7 / 18)
    assert calculate_aurc([0.2, 0.8], [0, 0]) == 0


@pytest.mark.parametrize("num_samples", [5, 20])
def test_flow_integration_preserves_class_conditioning(num_samples):
    class ConstantField:
        def vector_field(self, state, times, features, labels):
            return labels[:, None].expand_as(state).float()

    features = torch.zeros(2, 512)
    labels = torch.tensor([1, 3])
    torch.manual_seed(42)
    initial = torch.randn(2 * num_samples, 50).reshape(2, num_samples, 50)
    torch.manual_seed(42)
    result = generate_trajectories(ConstantField(), features, labels, num_samples)
    np.testing.assert_allclose(
        result.numpy(), (initial + labels[:, None, None]).numpy(), atol=2e-6
    )
