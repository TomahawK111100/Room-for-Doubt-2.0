import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

TARGET_TRAJ_LEN = 50


class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, TARGET_TRAJ_LEN),
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)
