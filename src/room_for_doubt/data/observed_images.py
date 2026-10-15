"""Indexed images with observed labels only; no clean training targets exposed."""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ObservedImages(Dataset):
    def __init__(self, images, labels, ids, transform):
        self.images, self.labels = images, np.asarray(labels)
        self.ids, self.transform = np.asarray(ids), transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = int(self.ids[index])
        return (
            self.transform(Image.fromarray(self.images[sid])),
            int(self.labels[sid]),
            sid,
        )


def imagenet_transform(size, train=False):
    ops = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if train
        else []
    )
    return transforms.Compose(
        ops
        + [
            transforms.Resize((size, size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
