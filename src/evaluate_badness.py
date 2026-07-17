import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.dataset import CIFAR_MEAN, CIFAR_STD
from src.train_traj_head import TrajectoryPredictor


TARGET_TRAJ_LEN = 50
TRAIN_END = 45000
VAL_END = 50000


class Stage1TrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, trajectories_path: str, indices: List[int], transform=None):
        self.cifar = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        self.transform = transform
        self.indices = list(indices)

        with open(trajectories_path, "r") as handle:
            self.trajectories: Dict[str, dict] = json.load(handle)

        self.features: List[np.ndarray] = []
        self.labels: List[int] = []

        for idx in self.indices:
            sample_id = int(idx)
            info = self.trajectories.get(str(sample_id), {})
            margins = info.get("margins", [])
            noisy_label = int(info.get("noisy_label", 0))
            clean_label = int(info.get("clean_label", 0))
            feature = self._pad_or_truncate(margins, TARGET_TRAJ_LEN)
            self.features.append(feature)
            self.labels.append(1 if noisy_label != clean_label else 0)

    @staticmethod
    def _pad_or_truncate(values: List[float], target_len: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return np.zeros(target_len, dtype=np.float32)
        if arr.size >= target_len:
            return arr[:target_len]
        pad_value = arr[-1]
        pad = np.full(target_len - arr.size, pad_value, dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_id = self.indices[idx]
        img, _ = self.cifar[sample_id]
        if self.transform is not None:
            img = self.transform(img)
        feature = np.asarray(self.features[idx], dtype=np.float32)
        label = int(self.labels[idx])
        return img, feature, label


class RawCIFAR10Test(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.base = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, _ = self.base[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(idx)


def _load_train_indices() -> Tuple[List[int], List[int]]:
    train_indices = list(range(0, TRAIN_END))
    val_indices = list(range(TRAIN_END, VAL_END))
    return train_indices, val_indices


def _build_stage1_arrays(data_dir: str, trajectories_path: str) -> Tuple[np.ndarray, np.ndarray]:
    train_indices, _ = _load_train_indices()
    dataset = Stage1TrajectoryDataset(data_dir, trajectories_path, train_indices, transform=None)
    x = np.stack(dataset.features, axis=0).astype(np.float32)
    y = np.asarray(dataset.labels, dtype=np.int64)
    return x, y


def _build_test_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_set = RawCIFAR10Test(data_dir, transform=transform)
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _predict_trajectories(model: TrajectoryPredictor, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        traj = model(imgs)
        outputs.append(traj.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _load_trajectory_predictor(weights_path: str, device: torch.device) -> TrajectoryPredictor:
    model = TrajectoryPredictor().to(device)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate badness scores with LDA and a trajectory predictor.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--trajectories_path", type=str, default="results/stage1_trajectories.json")
    parser.add_argument("--trajectory_head_path", type=str, default="checkpoints/trajectory_head.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output_path", type=str, default="results/test_badness_scores.npy")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("medium")

    x_train, y_train = _build_stage1_arrays(args.data_dir, args.trajectories_path)

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    train_acc = float(lda.score(x_train, y_train))
    print(f"LDA train accuracy: {train_acc:.4f}")

    device = _get_device()
    print(f"Using device: {device}")

    model = _load_trajectory_predictor(args.trajectory_head_path, device)
    test_loader = _build_test_loader(args.data_dir, args.batch_size, args.num_workers)

    predicted_trajectories = _predict_trajectories(model, test_loader, device)
    badness_scores = lda.predict_proba(predicted_trajectories)[:, 1].astype(np.float32)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, badness_scores)
    print(f"Saved badness scores to {output_path}")
    print(f"Badness score stats: min={badness_scores.min():.4f}, mean={badness_scores.mean():.4f}, max={badness_scores.max():.4f}")


if __name__ == "__main__":
    main()
