import json
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

TARGET_TRAJ_LEN = 50


class TrajectoryDataset(Dataset):
    def __init__(
        self, data_dir: str, trajectories_path: str, indices: List[int], transform=None
    ):
        self.cifar = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )
        self.transform = transform
        self.indices = list(indices)

        with open(trajectories_path, "r") as handle:
            self.trajectories: Dict[str, dict] = json.load(handle)

        self.targets: List[np.ndarray] = []
        self.sample_ids: List[int] = []

        for idx in self.indices:
            sample_id = int(idx)
            info = self.trajectories.get(str(sample_id), {})
            margins = info.get("margins", [])
            traj = self._pad_or_truncate(margins, TARGET_TRAJ_LEN)
            self.targets.append(traj)
            self.sample_ids.append(sample_id)

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
        sample_id = self.sample_ids[idx]
        img, _ = self.cifar[sample_id]
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, target
