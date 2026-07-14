import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from src.dataset import CIFAR_MEAN, CIFAR_STD, CleanCIFARTest
from src.model import ResNetClassifier


class MetaRegressor(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


def _load_student(checkpoint_path: str, num_classes: int = 10) -> ResNetClassifier:
    return ResNetClassifier.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
        pretrained=False,
        learning_rate=1e-3,
    )


def _load_meta_model(weights_path: str) -> MetaRegressor:
    model = MetaRegressor(lr=1e-4)
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _collect_test_predictions(
    student: ResNetClassifier,
    meta_model: MetaRegressor,
    dataloader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    student.eval()
    meta_model.eval()

    rows: List[Dict[str, float]] = []

    for imgs, labels, _, sample_ids in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        sample_ids = sample_ids.to(device)

        logits = student(imgs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        predicted_flips = meta_model(imgs).to(device)

        top2_probs, _ = torch.topk(probs, k=2, dim=1)
        predictive_margin = top2_probs[:, 0] - top2_probs[:, 1]

        correct = (preds == labels).float()

        for i in range(imgs.size(0)):
            rows.append(
                {
                    "sample_id": int(sample_ids[i].item()),
                    "label": int(labels[i].item()),
                    "pred": int(preds[i].item()),
                    "correct": float(correct[i].item()),
                    "student_confidence": float(probs[i, preds[i]].item()),
                    "predictive_margin": float(predictive_margin[i].item()),
                    "predicted_flips": float(predicted_flips[i].item()),
                }
            )

    return pd.DataFrame(rows)


def _compute_selective_curve(df: pd.DataFrame, rejection_rates: List[int]) -> pd.DataFrame:
    base_accuracy = float(df["correct"].mean())
    results: List[Dict[str, float]] = []

    sorted_df = df.sort_values("predicted_flips", ascending=False).reset_index(drop=True)
    total = len(sorted_df)

    for rejection_rate in rejection_rates:
        reject_count = int(np.floor(total * (rejection_rate / 100.0)))
        kept_df = sorted_df.iloc[reject_count:]
        accuracy = float(kept_df["correct"].mean()) if len(kept_df) > 0 else np.nan
        improvement = accuracy - base_accuracy if not np.isnan(accuracy) else np.nan

        results.append(
            {
                "rejection_rate": rejection_rate,
                "kept_samples": len(kept_df),
                "accuracy": accuracy,
                "accuracy_gain_vs_baseline": improvement,
            }
        )

    return pd.DataFrame(results)


def _plot_curve(results_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(8.5, 5.5))
    sns.lineplot(
        data=results_df,
        x="rejection_rate",
        y="accuracy",
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
    )
    plt.title("Selective Inference: Accuracy vs. Rejection Rate")
    plt.xlabel("Rejection Rate (%)")
    plt.ylabel("Student Accuracy on Remaining Samples")
    plt.ylim(0.0, 1.0)
    plt.xlim(results_df["rejection_rate"].min(), results_df["rejection_rate"].max())
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _print_results_table(results_df: pd.DataFrame) -> None:
    display_df = results_df.copy()
    display_df["accuracy"] = display_df["accuracy"].map(lambda x: f"{x:.4f}")
    display_df["accuracy_gain_vs_baseline"] = display_df["accuracy_gain_vs_baseline"].map(lambda x: f"{x:+.4f}")
    print("\nSelective Inference Results")
    print(display_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective inference for student + meta regressor.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to CIFAR-10 data directory")
    parser.add_argument("--student_ckpt", type=str, default="checkpoints/best.ckpt", help="Path to student checkpoint")
    parser.add_argument("--meta_ckpt", type=str, default="checkpoints/meta_model.pth", help="Path to meta-regressor weights")
    parser.add_argument("--batch_size", type=int, default=256, help="Test batch size")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4, help="Dataloader workers")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory for plots")
    args = parser.parse_args()

    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = _load_student(args.student_ckpt)
    student = student.to(device)
    try:
        student = torch.compile(student)
    except Exception:
        pass

    meta_model = _load_meta_model(args.meta_ckpt).to(device)
    try:
        meta_model = torch.compile(meta_model)
    except Exception:
        pass

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_set = CleanCIFARTest(args.data_dir, dataset_name="cifar10n", transform=transform_eval)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    df = _collect_test_predictions(student, meta_model, test_loader, device)
    rejection_rates = list(range(0, 51, 5))
    results_df = _compute_selective_curve(df, rejection_rates)

    baseline_accuracy = float(results_df.loc[results_df["rejection_rate"] == 0, "accuracy"].iloc[0])
    results_df["accuracy_gain_vs_baseline"] = results_df["accuracy"] - baseline_accuracy

    output_path = Path(args.output_dir) / "selective_inference_curve.png"
    _plot_curve(results_df, output_path)
    _print_results_table(results_df)

    print(f"\nBaseline accuracy: {baseline_accuracy:.4f}")
    print(f"Saved curve to: {output_path}")


if __name__ == "__main__":
    main()
