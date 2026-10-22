import os
import urllib.request
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR10N_URL = (
    "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"
)
CIFAR100N_URL = (
    "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt"
)

SPLIT_KEY_MAP = {
    "aggregate": "aggre_label",
    "worse": "worse_label",
    "random1": "random_label1",
    "random2": "random_label2",
    "random3": "random_label3",
    "noisy100": "noisy_label",
    "noisy100fine": "noisy_label",
    "noisy_coarse": "noisy_coarse_label",
}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def get_train_val_indices(
    num_samples: int, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if num_samples < 2 or not 0 < val_ratio < 1:
        raise ValueError("Split requires at least two samples and 0 < val_ratio < 1")
    n_val = int(num_samples * val_ratio)
    if not 0 < n_val < num_samples:
        raise ValueError("Split must contain both training and validation samples")
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return indices[n_val:], indices[:n_val]


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise RuntimeError(
                f"Could not download CIFAR-N annotations from {url}. "
                f"Place the official annotation file at {path}."
            ) from e


def _load_annotation_fields(data_dir, dataset_name, noisy_split, include_clean):
    if dataset_name not in {"cifar10n", "cifar100n"}:
        raise ValueError(f"Unsupported noisy dataset: {dataset_name}")
    if noisy_split not in SPLIT_KEY_MAP:
        raise ValueError(f"Unknown noisy split: {noisy_split}")
    key = SPLIT_KEY_MAP[noisy_split]

    if dataset_name == "cifar10n":
        path = os.path.join(data_dir, "CIFAR-10_human.pt")
        _download(CIFAR10N_URL, path)
    else:
        path = os.path.join(data_dir, "CIFAR-100_human.pt")
        _download(CIFAR100N_URL, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Official CIFAR-N annotations are missing: {path}")

    raw = torch.load(path, map_location="cpu", weights_only=False)
    noisy = np.array(raw[key], dtype=np.int64)
    clean = np.array(raw["clean_label"], dtype=np.int64) if include_clean else None
    num_classes = 10 if dataset_name == "cifar10n" else 100
    fields = (
        (("noisy", noisy), ("clean", clean)) if include_clean else (("noisy", noisy),)
    )
    for name, values in fields:
        if values.shape != (50000,):
            raise ValueError(
                f"Expected 50000 {name} labels in {path}; got {values.shape}"
            )
        if np.any(values < 0) or np.any(values >= num_classes):
            raise ValueError(f"Out-of-range {name} labels in {path}")
    return noisy, clean


def load_noisy_labels(
    data_dir: str, dataset_name: str, noisy_split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return observed and clean labels for explicitly labeled evaluation/oracle use."""
    return _load_annotation_fields(data_dir, dataset_name, noisy_split, True)


def load_observed_labels(
    data_dir: str, dataset_name: str, noisy_split: str
) -> np.ndarray:
    """Load observed annotations without requiring a clean_label field."""
    if dataset_name == "animal10n":
        from room_for_doubt.data.animal10n import Animal10N

        if noisy_split != "human":
            raise ValueError("Animal-10N exposes only the original human labels")
        return Animal10N(data_dir, train=True).targets
    return _load_annotation_fields(data_dir, dataset_name, noisy_split, False)[0]


class NoisyCIFARDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        noisy_split: str,
        indices: np.ndarray,
        transform=None,
        num_classes: int = 10,
    ):
        train = dataset_name == "cifar10n"
        cifar_cls = datasets.CIFAR10 if train else datasets.CIFAR100
        # download=True, чтобы не лезть в упавший интернет
        base = cifar_cls(root=data_dir, train=True, download=True, transform=None)
        noisy_all, clean_all = load_noisy_labels(data_dir, dataset_name, noisy_split)

        self.data = base.data[indices]
        self.noisy_targets = noisy_all[indices]
        self.clean_targets = clean_all[indices]
        self.sample_ids = indices.copy()
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return (
            img,
            int(self.noisy_targets[idx]),
            int(self.clean_targets[idx]),
            int(self.sample_ids[idx]),
        )


class CleanCIFARTest(Dataset):
    def __init__(self, data_dir: str, dataset_name: str, transform=None):
        train = dataset_name == "cifar10n"
        cifar_cls = datasets.CIFAR10 if train else datasets.CIFAR100
        self.base = cifar_cls(
            root=data_dir, train=False, download=True, transform=transform
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        return img, label, label, idx


class CIFARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        dataset_name: str = "cifar10n",
        noisy_split: str = "worse",
        batch_size: int = 256,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        seed: int = 42,
        num_classes: int = 10,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.noisy_split = noisy_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.num_classes = num_classes
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        self.transform_eval = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    def prepare_data(self) -> None:
        train = self.dataset_name == "cifar10n"
        cifar_cls = datasets.CIFAR10 if train else datasets.CIFAR100
        # Отключаем принудительное скачивание (download=True)
        cifar_cls(self.data_dir, train=True, download=True)
        cifar_cls(self.data_dir, train=False, download=True)
        load_noisy_labels(self.data_dir, self.dataset_name, self.noisy_split)

    def setup(self, stage: Optional[str] = None) -> None:
        n_train = 50000 if self.dataset_name == "cifar10n" else 50000
        self.train_indices, self.val_indices = get_train_val_indices(
            n_train, self.val_ratio, self.seed
        )

        if stage in ("fit", None):
            self.train_set = NoisyCIFARDataset(
                self.data_dir,
                self.dataset_name,
                self.noisy_split,
                self.train_indices,
                self.transform_train,
                self.num_classes,
            )
            self.val_set = NoisyCIFARDataset(
                self.data_dir,
                self.dataset_name,
                self.noisy_split,
                self.val_indices,
                self.transform_eval,
                self.num_classes,
            )
        if stage in ("test", None):
            self.test_set = CleanCIFARTest(
                self.data_dir, self.dataset_name, self.transform_eval
            )

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        # Универсальная проверка для включения pin_memory как на CUDA, так и на MPS
        use_pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
        persistent_workers = self.num_workers > 0
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            # Every training sample must contribute a margin at every epoch.
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False)
