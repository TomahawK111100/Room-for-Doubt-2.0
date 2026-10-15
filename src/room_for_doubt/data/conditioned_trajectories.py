import torch
from torch.utils.data import Dataset
import torchvision
from room_for_doubt.data.trajectory_records import load_verified_trajectories


class CondTrajDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "cifar10",
        root_dir: str = "./data",
        traj_file: str = "artifacts/trajectories/stage1_trajectories.json",
        transform=None,
        seed=42,
        val_ratio=0.1,
        start_epoch=0,
        missing_epoch_policy="drop",
    ):
        """
        Dataset that loads CIFAR images alongside their trajectory margins and noisy labels.
        """
        if dataset_name == "cifar10":
            dataset_cls = torchvision.datasets.CIFAR10
        elif dataset_name == "cifar100":
            dataset_cls = torchvision.datasets.CIFAR100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.cifar = dataset_cls(
            root=root_dir, train=True, download=False, transform=transform
        )
        self.verified = load_verified_trajectories(
            traj_file, dataset_name, self.cifar.targets, seed, val_ratio
        )
        self.trajectories = self.verified.records
        self.sample_ids, self.targets, self.excluded_ids = (
            self.verified.aligned_targets(
                start_epoch=start_epoch, missing_policy=missing_epoch_policy
            )
        )
        self.trajectory_summary = {
            "base_train_samples": len(self.verified.train_ids),
            "usable_train_samples": len(self.sample_ids),
            "excluded_train_samples": len(self.excluded_ids),
            "validation_samples": len(self.verified.val_ids),
            "epoch_start": start_epoch,
            "epoch_end": start_epoch + 49,
            "missing_epoch_policy": missing_epoch_policy,
            "margin_definition": "probability_margin",
        }

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = int(self.sample_ids[idx])
        image, _ = self.cifar[sample_id]
        noisy_label = self.trajectories[str(sample_id)]["noisy_label"]
        margins = self.targets[idx]

        return (
            image,
            torch.tensor(noisy_label, dtype=torch.long),
            torch.tensor(margins, dtype=torch.float32),
        )
