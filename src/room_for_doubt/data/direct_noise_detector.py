import json
import torch
from torch.utils.data import Dataset
import torchvision


class DirectCertaintyDataset(Dataset):
    def __init__(
        self,
        root_dir="./data",
        traj_file="artifacts/trajectories/stage1_trajectories.json",
        transform=None,
    ):
        """
        Dataset that loads CIFAR-10 images alongside their noisy labels and computes 'Badness'.
        Badness is defined as 1 if the noisy label is incorrect, 0 otherwise.
        """
        self.cifar = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=transform
        )
        with open(traj_file, "r") as f:
            self.trajectories = json.load(f)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, _ = self.cifar[idx]

        str_idx = str(idx)
        if str_idx in self.trajectories:
            data = self.trajectories[str_idx]
            noisy_label = data.get("noisy_label", 0)
            clean_label = data.get("true_label", data.get("clean_label", 0))
            # Badness: 1.0 if the noisy label doesn't match the clean label (i.e. it's a bad prediction)
            badness = 1.0 if noisy_label != clean_label else 0.0
        else:
            noisy_label = 0
            badness = 0.0

        return (
            image,
            torch.tensor(noisy_label, dtype=torch.long),
            torch.tensor(badness, dtype=torch.float32),
        )
