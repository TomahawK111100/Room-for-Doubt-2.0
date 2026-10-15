import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import pytorch_lightning as pl


class DirectCertaintyEstimator(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Frozen resnet18 without the final fc layer
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze resnet parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Embedding for the class label
        self.class_emb = nn.Embedding(10, 32)

        # MLP for predicting the scalar 'Badness' score directly
        self.mlp = nn.Sequential(nn.Linear(512 + 32, 256), nn.ReLU(), nn.Linear(256, 1))

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, img, cls):
        # Extract image features
        f = self.features(img)  # [B, 512, 1, 1]
        f = torch.flatten(f, 1)  # [B, 512]

        # Extract class embeddings
        c = self.class_emb(cls)  # [B, 32]

        # Concatenate and pass through MLP
        x = torch.cat([f, c], dim=1)  # [B, 544]
        out = self.mlp(x)  # [B, 1]

        return out.squeeze(-1)  # [B]

    def training_step(self, batch, batch_idx):
        images, labels, badness = batch
        logits = self(images, labels)

        loss = self.criterion(logits, badness)

        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == badness).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, badness = batch
        logits = self(images, labels)

        loss = self.criterion(logits, badness)

        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == badness).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
