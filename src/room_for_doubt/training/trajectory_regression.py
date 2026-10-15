from room_for_doubt.models.trajectory_regression import TrajectoryPredictor
from room_for_doubt.data.trajectory_regression import TrajectoryDataset
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
TARGET_TRAJ_LEN = 50
TRAIN_END = 45000
VAL_END = 50000


def _load_indices() -> Tuple[List[int], List[int]]:
    train_indices = list(range(0, TRAIN_END))
    val_indices = list(range(TRAIN_END, VAL_END))
    return train_indices, val_indices


def _build_loaders(
    data_dir: str, trajectories_path: str, batch_size: int, num_workers: int
):
    train_indices, val_indices = _load_indices()

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_set = TrajectoryDataset(
        data_dir, trajectories_path, train_indices, transform_train
    )
    val_set = TrajectoryDataset(
        data_dir, trajectories_path, val_indices, transform_eval
    )

    pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_batches = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a trajectory prediction head on stage1 margins."
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--trajectories_path",
        type=str,
        default="artifacts/trajectories/stage1_trajectories.json",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--save_path", type=str, default="checkpoints/trajectory_head.pth"
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("medium")

    device = _get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = _build_loaders(
        data_dir=args.data_dir,
        trajectories_path=args.trajectories_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = TrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        val_loss = _evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={avg_train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved trajectory head state_dict to {save_path}")


if __name__ == "__main__":
    main()
