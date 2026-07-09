import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet34, ResNet34_Weights


class ResNetFeatureExtractor(nn.Module):
    """
    Backbone module that extracts both logits and pre-classification descriptors (features).
    """
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet34(weights=weights)
        # Adjust for CIFAR (32x32) image sizes
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
        # Extract the original fc layer's input features and replace it with Identity
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        # Custom head for classification
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.model(x)
        logits = self.fc(features)
        return logits, features


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, pretrained=False, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = ResNetFeatureExtractor(num_classes=num_classes, pretrained=pretrained)
        self.learning_rate = learning_rate

    def forward(self, x):
        logits, _ = self.backbone(x)
        return logits

    def _step(self, batch, split: str):
        imgs, noisy_labels, clean_labels, sample_ids = batch
        logits, features = self.backbone(imgs)
        loss = F.cross_entropy(logits, noisy_labels)
        probs = F.softmax(logits, dim=1)
        acc = (logits.argmax(1) == noisy_labels).float().mean()
        
        # Uncertainty metric: predictive entropy
        predictive_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        trajectory = {
            "per_sample_loss": F.cross_entropy(logits, noisy_labels, reduction="none").detach(),
            "per_sample_probs": probs.detach(),
            "predictive_entropy": predictive_entropy.detach(),
            "noisy_labels": noisy_labels.detach(),
            "clean_labels": clean_labels.detach(),
            "sample_ids": sample_ids.detach(),
        }
        
        # Логируем только для валидации/теста здесь, чтобы избежать дублирования во время train
        if split != "train":
            self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{split}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            
        return {"loss": loss, "trajectory": trajectory}

    def training_step(self, batch, batch_idx):
        out = self._step(batch, "train")
        # Единственная точка логирования train_loss
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Add Cosine Annealing scheduler over max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class DescriptorDistillationModule(pl.LightningModule):
    """
    Teacher-Student Knowledge Distillation tracking per-sample uncertainty and descriptor dynamics.
    """
    def __init__(
        self,
        num_classes: int = 10,
        pretrained_teacher: bool = True,
        learning_rate: float = 1e-3,
        distill_alpha: float = 0.5,
        temperature: float = 2.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.distill_alpha = distill_alpha
        self.temperature = temperature
        
        # Student model (learning from scratch or fine-tuning)
        self.student = ResNetFeatureExtractor(num_classes=num_classes, pretrained=False)
        
        # Teacher model (usually pre-trained, providing robust descriptors)
        self.teacher = ResNetFeatureExtractor(num_classes=num_classes, pretrained=pretrained_teacher)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, x):
        logits, _ = self.student(x)
        return logits

    def _distillation_loss(self, student_logits, student_features, teacher_logits, teacher_features, labels):
        # Standard cross-entropy on hard labels
        loss_cls = F.cross_entropy(student_logits, labels)
        
        # Descriptor distance (MSE on high-dimensional embeddings)
        loss_desc = F.mse_loss(student_features, teacher_features)
        
        # Combine losses
        total_loss = (1.0 - self.distill_alpha) * loss_cls + self.distill_alpha * loss_desc
        return total_loss, loss_cls, loss_desc

    def _step(self, batch, split: str):
        imgs, noisy_labels, clean_labels, sample_ids = batch
        
        # Get student and teacher representations
        student_logits, student_features = self.student(imgs)
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher(imgs)
            
        loss, loss_cls, loss_desc = self._distillation_loss(
            student_logits, student_features, teacher_logits, teacher_features, noisy_labels
        )
        
        probs = F.softmax(student_logits, dim=1)
        acc = (student_logits.argmax(1) == noisy_labels).float().mean()
        
        # --- Uncertainty & Dynamics Tracking ---
        # 1. Predictive Entropy: H(Y|X) = -sum(P(Y|X) * log(P(Y|X)))
        predictive_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # 2. Prediction Margin: P(y_top1) - P(y_top2)
        top2_probs, _ = torch.topk(probs, 2, dim=1)
        prediction_margin = top2_probs[:, 0] - top2_probs[:, 1]
        
        # 3. Descriptor Divergence: L2 distance per sample between student and teacher
        descriptor_distance = torch.norm(student_features - teacher_features, p=2, dim=1)
        
        trajectory = {
            "per_sample_loss": F.cross_entropy(student_logits, noisy_labels, reduction="none").detach(),
            "per_sample_probs": probs.detach(),
            "predictive_entropy": predictive_entropy.detach(),
            "prediction_margin": prediction_margin.detach(),
            "descriptor_distance": descriptor_distance.detach(),
            "noisy_labels": noisy_labels.detach(),
            "clean_labels": clean_labels.detach(),
            "sample_ids": sample_ids.detach(),
        }
        
        # Logging
        if split != "train":
            self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{split}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            
        self.log(f"{split}_loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log(f"{split}_loss_desc", loss_desc, on_step=False, on_epoch=True)
            
        return {"loss": loss, "trajectory": trajectory}

    def training_step(self, batch, batch_idx):
        out = self._step(batch, "train")
        # Единственная точка логирования train_loss для дистилляции
        self.log("train_loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True)
        return out

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        imgs, labels, _, _ = batch
        logits, _ = self.student(imgs)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}