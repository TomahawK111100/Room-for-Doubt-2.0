"""Feature-only scalar head; no second image encoder is stored in checkpoints."""

import torch
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor


class FeatureMarginHead(ConditionedPredictor):
    def __init__(self, classes, feature_dim=512):
        torch.nn.Module.__init__(self)
        self.features = torch.nn.Identity()
        self.class_emb = torch.nn.Embedding(classes, 32)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + 32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def predict_from_features(self, features, classes):
        return super().predict_from_features(features, classes).expand(-1, 50)
