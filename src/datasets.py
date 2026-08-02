from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_dataloaders(
    dataset_name: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset_key = dataset_name.lower()
    if dataset_key == "cifar10":
        dataset_cls = datasets.CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        num_classes = 10
    elif dataset_key == "cifar100":
        dataset_cls = datasets.CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}. Use 'cifar10' or 'cifar100'.")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = dataset_cls(root="./data", train=True, download=True, transform=train_transform)
    test_set = dataset_cls(root="./data", train=False, download=True, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, test_loader, num_classes
