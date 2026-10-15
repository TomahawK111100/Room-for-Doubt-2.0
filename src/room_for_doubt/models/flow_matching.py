import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CFMVectorField(nn.Module):
    def __init__(self, num_classes=100, feature_dim=512, traj_dim=50):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 32)
        self.mlp = nn.Sequential(
            nn.Linear(traj_dim + 1 + feature_dim + 32, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, traj_dim),
        )

    def forward(self, x_t, t, img_features, class_labels):
        c_emb = self.class_emb(class_labels)
        return self.mlp(torch.cat([x_t, t.unsqueeze(1), img_features, c_emb], dim=1))


class CFMTrajectoryModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        self.features = nn.Sequential(
            *list(
                resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                ).children()
            )[:-1]
        )
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()
        self.vector_field = CFMVectorField(num_classes=num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def extract_features(self, img):
        with torch.no_grad():
            return torch.flatten(self.features(img), 1)
