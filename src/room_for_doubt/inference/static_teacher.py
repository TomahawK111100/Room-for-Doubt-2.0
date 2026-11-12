"""Static five-view control for early-snapshot distillation on native CIFAR images."""

from functools import partial
from tempfile import TemporaryDirectory

import numpy as np

from room_for_doubt.inference.temporal_evidence import extract


STATIC_VIEWS = ("identity", "horizontal_flip", "right", "left", "down")
CROP_OFFSETS = {"right": (2, 0), "left": (2, 4), "down": (0, 2)}


def static_view(image, view):
    """Apply the fixed transform before the standard ImageNet preprocessing."""
    if image.shape != (32, 32, 3) or image.dtype != np.uint8:
        raise ValueError("Static teacher requires native 32x32 RGB uint8 images")
    if view == "identity":
        return image
    if view == "horizontal_flip":
        return image[:, ::-1].copy()
    top, left = CROP_OFFSETS[view]
    padded = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    return padded[top : top + 32, left : left + 32]


def static_teacher_probabilities(model, images, ids, image_size, batch_size, device, scratch):
    """Average probabilities in ID order using one frozen checkpoint."""
    total = np.zeros((len(ids), model.fc.out_features), dtype=np.float64)
    with TemporaryDirectory(dir=scratch) as directory:
        for view in STATIC_VIEWS:
            _, probabilities = extract(
                model, images, ids, image_size, batch_size, device, directory,
                view, False, image_view=partial(static_view, view=view),
            )
            total += probabilities
            del probabilities
    result = (total / len(STATIC_VIEWS)).astype(np.float32)
    result /= result.sum(axis=1, keepdims=True)
    return result
