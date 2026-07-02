
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet18_Weights

# Simple ResNet block (from PyTorch example, for custom ResNet building if needed)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input if stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, pretrained=True, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # Use official weights
        else:
            self.model = resnet18() # Randomly initialized

        # Adapt the first convolution layer for CIFAR if needed (32x32 input)
        # Default ResNet uses kernel_size=7, stride=2, padding=3 which reduces 32x32 too much
        # For CIFAR, common practice is kernel_size=3, stride=1, padding=1
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity() # Remove maxpool for CIFAR

        # Adjust the final fully connected layer for num_classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, noisy_labels, clean_labels, sample_ids = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, noisy_labels) # Use noisy labels for training

        probs = F.softmax(logits, dim=1)
        
        # Log per-sample trajectories
        # Detach from graph and move to CPU for logging
        per_sample_loss = F.cross_entropy(logits, noisy_labels, reduction=\'none\').detach().cpu()
        per_sample_probs = probs.detach().cpu()
        noisy_labels_cpu = noisy_labels.detach().cpu()
        clean_labels_cpu = clean_labels.detach().cpu()
        sample_ids_cpu = sample_ids.detach().cpu()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("hp_metric", loss) # Required for some loggers like TensorBoard to track best models

        return {
            "loss": loss,
            "per_sample_loss": per_sample_loss,
            "per_sample_probs": per_sample_probs,
            "noisy_labels": noisy_labels_cpu,
            "clean_labels": clean_labels_cpu,
            "sample_ids": sample_ids_cpu
        }

    def validation_step(self, batch, batch_idx):
        imgs, noisy_labels, clean_labels, sample_ids = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, noisy_labels) # Use noisy labels for validation
        acc = (logits.argmax(dim=1) == noisy_labels).float().mean()

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        imgs, labels, clean_labels, sample_ids = batch # For test, labels are clean CIFAR-10 labels
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    # Basic test of the model
    model = ResNetClassifier()
    print(model)

    # Test a dummy forward pass
    dummy_input = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (1, 10) for CIFAR-10

    # Verify model adaptations for CIFAR
    assert model.model.conv1.kernel_size == (3, 3)
    assert model.model.conv1.stride == (1, 1)
    assert isinstance(model.model.maxpool, nn.Identity)
    print("Model adaptations for CIFAR verified.")
