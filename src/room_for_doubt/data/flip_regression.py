import json
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class MetaRegressionDataset(Dataset):
    def __init__(self, data_dir: str, trajectories_path: str, transform=None):
        self.cifar = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )
        self.transform = transform

        with open(trajectories_path, "r") as f:
            self.trajectories = json.load(f)

        self.valid_indices = []
        self.targets = []

        for idx in range(len(self.cifar)):
            str_idx = str(idx)
            if str_idx in self.trajectories:
                self.valid_indices.append(idx)
                flips = self.trajectories[str_idx].get("flips", 0)
                self.targets.append(float(flips))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        orig_idx = self.valid_indices[i]
        img, _ = self.cifar[orig_idx]

        if self.transform:
            img = self.transform(img)

        target = torch.tensor(self.targets[i], dtype=torch.float32)
        return img, target


class ValSubsetWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        orig_idx = self.subset.indices[idx]
        dataset = self.subset.dataset
        real_cifar_idx = dataset.valid_indices[orig_idx]

        img, _ = dataset.cifar[real_cifar_idx]
        if self.transform:
            img = self.transform(img)

        target = torch.tensor(dataset.targets[orig_idx], dtype=torch.float32)
        return img, target
