import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ConditionedPredictor(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained=True):
        super().__init__()
        # Frozen resnet18 without the final fc layer
        resnet = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze resnet parameters
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        # Embedding for the class label
        self.class_emb = nn.Embedding(num_classes, 32)

        # MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(512 + 32, 256), nn.ReLU(), nn.Linear(256, 50)
        )

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def extract_features(self, img):
        with torch.no_grad():
            return torch.flatten(self.features(img), 1)

    def predict_from_features(self, f, cls):

        # Extract class embeddings
        c = self.class_emb(cls)  # [B, 32]

        # Concatenate and pass through MLP
        x = torch.cat([f, c], dim=1)  # [B, 544]
        out = self.mlp(x)  # [B, 50]
        return out

    def forward(self, img, cls):
        return self.predict_from_features(self.extract_features(img), cls)
