import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet18_Weights


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, pretrained=False, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, split: str):
        imgs, noisy_labels, clean_labels, sample_ids = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, noisy_labels)
        probs = F.softmax(logits, dim=1)
        acc = (logits.argmax(1) == noisy_labels).float().mean()
        trajectory = {
            "per_sample_loss": F.cross_entropy(logits, noisy_labels, reduction="none").detach(),
            "per_sample_probs": probs.detach(),
            "noisy_labels": noisy_labels.detach(),
            "clean_labels": clean_labels.detach(),
            "sample_ids": sample_ids.detach(),
        }
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, prog_bar=(split != "train"))
        if split != "train":
            self.log(f"{split}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "trajectory": trajectory}

    def training_step(self, batch, batch_idx):
        out = self._step(batch, "train")
        self.log("train_loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True)
        return out

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        imgs, labels, _, _ = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
