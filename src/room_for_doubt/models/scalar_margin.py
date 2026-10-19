"""Matched hidden layers for scalar margin controls."""

import torch
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor


class ScalarMarginPredictor(ConditionedPredictor):
    def __init__(self, num_classes=100, pretrained=False):
        super().__init__(num_classes, pretrained)
        self.mlp[-1] = torch.nn.Linear(256, 1)

    def predict_from_features(self, features, classes):
        scalar = super().predict_from_features(features, classes)
        # Repetition preserves compatibility with the common trajectory MSE
        # and scoring path; it does not create additional learnable outputs.
        return scalar.expand(-1, 50)
