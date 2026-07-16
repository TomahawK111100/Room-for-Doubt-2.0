import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.dataset import CIFAR_MEAN, CIFAR_STD
from src.model import ResNetClassifier


class RawCIFAR10Test(Dataset):
    def __init__(self, data_dir: str):
        self.base = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        return img, int(label), int(idx)


class MetaRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = torch.nn.Identity()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


def _load_student(checkpoint_path: str, num_classes: int = 10) -> ResNetClassifier:
    return ResNetClassifier.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
        pretrained=False,
        learning_rate=1e-3,
    )


def _load_meta_model(ckpt_path: str) -> MetaRegressor:
    from train_meta import MetaRegressor as TrainMetaRegressor

    model = TrainMetaRegressor()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace("_orig_mod.", "")] = value

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _student_predict(student: ResNetClassifier, imgs: torch.Tensor) -> torch.Tensor:
    logits = student(imgs)
    return F.softmax(logits, dim=1)


@torch.no_grad()
def _meta_predict(meta_model: MetaRegressor, imgs: torch.Tensor) -> torch.Tensor:
    return meta_model(imgs)


def _build_tta_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def _build_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


@torch.no_grad()
def _evaluate_tta(
    student: ResNetClassifier,
    meta_model: MetaRegressor,
    dataloader: DataLoader,
    device: torch.device,
    num_augs: int = 15,
    temperature: float = 10.0,
) -> Tuple[float, float, int]:
    student.eval()
    meta_model.eval()

    baseline_correct = 0
    tta_correct = 0
    total = 0

    tta_transform = _build_tta_transform()
    eval_transform = _build_eval_transform()

    for images, labels, sample_ids in dataloader:
        labels = labels.to(device)
        total += labels.size(0)

        batch_baseline_imgs = []
        batch_tta_probs: List[torch.Tensor] = []

        for image in images:
            baseline_img = eval_transform(image)
            aug_views = []
            for _ in range(num_augs):
                aug_img = tta_transform(image)
                aug_views.append(aug_img)
            aug_batch = torch.stack(aug_views, dim=0).to(device)
            batch_baseline_imgs.append(baseline_img)

            probs = _student_predict(student, aug_batch)
            uncertainty = _meta_predict(meta_model, aug_batch)
            weights = torch.exp(-uncertainty / temperature)
            weights = weights / weights.sum().clamp_min(1e-8)
            final_probs = torch.sum(weights.unsqueeze(1) * probs, dim=0)
            batch_tta_probs.append(final_probs)

        baseline_batch = torch.stack(batch_baseline_imgs, dim=0).to(device)
        baseline_probs = _student_predict(student, baseline_batch)

        baseline_preds = baseline_probs.argmax(dim=1)
        tta_probs = torch.stack(batch_tta_probs, dim=0)
        tta_preds = tta_probs.argmax(dim=1)

        baseline_correct += (baseline_preds == labels).sum().item()
        tta_correct += (tta_preds == labels).sum().item()

    baseline_acc = baseline_correct / total
    tta_acc = tta_correct / total
    return baseline_acc, tta_acc, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate uncertainty-weighted TTA on CIFAR-10 test data.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to CIFAR-10 data")
    parser.add_argument("--student_ckpt", type=str, default="checkpoints/best.ckpt", help="Path to student checkpoint")
    parser.add_argument("--meta_ckpt", type=str, default="checkpoints/meta_model.pth", help="Path to meta-regressor weights")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for test evaluation")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4, help="Number of dataloader workers")
    parser.add_argument("--num_augs", type=int, default=15, help="Number of TTA augmentations per test image")
    parser.add_argument("--temperature", type=float, default=10.0, help="Temperature for uncertainty weighting")
    args = parser.parse_args()

    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = _load_student(args.student_ckpt).to(device)
    meta_model = _load_meta_model(args.meta_ckpt).to(device)

    try:
        student = torch.compile(student)
    except Exception:
        pass

    try:
        meta_model = torch.compile(meta_model)
    except Exception:
        pass

    test_set = RawCIFAR10Test(args.data_dir)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    baseline_acc, tta_acc, total = _evaluate_tta(
        student=student,
        meta_model=meta_model,
        dataloader=test_loader,
        device=device,
        num_augs=args.num_augs,
        temperature=args.temperature,
    )

    improvement = tta_acc - baseline_acc
    print("Uncertainty-Weighted TTA Evaluation")
    print(f"Test samples: {total}")
    print(f"Baseline accuracy (no TTA): {baseline_acc:.4f}")
    print(f"Uncertainty-Weighted TTA accuracy: {tta_acc:.4f}")
    print(f"Accuracy gain: {improvement:+.4f}")


if __name__ == "__main__":
    main()
