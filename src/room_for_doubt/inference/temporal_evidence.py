"""Bounded-memory canonical FP32 extraction for real retained teacher weights."""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from room_for_doubt.data.observed_images import imagenet_transform


@torch.no_grad()
def extract(
    model, images, ids, image_size, batch_size, device, directory, prefix, with_features,
    *, image_view=None,
):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if batch_size < 1 or len(ids) == 0:
        raise ValueError("Positive extraction batch size and nonempty IDs required")
    model = model.to(device).eval()
    classes, width = model.fc.out_features, model.fc.in_features
    probs = np.lib.format.open_memmap(
        root / f"{prefix}_probs.npy",
        mode="w+",
        dtype="float32",
        shape=(len(ids), classes),
    )
    features = None
    captured = []
    hook = None
    if with_features:
        features = np.lib.format.open_memmap(
            root / f"{prefix}_features.npy",
            mode="w+",
            dtype="float32",
            shape=(len(ids), width),
        )
        hook = model.fc.register_forward_pre_hook(
            lambda _, inputs: captured.append(inputs[0].detach())
        )
    transform = imagenet_transform(image_size)

    def input_tensor(image):
        if image_view is not None:
            image = image_view(image)
        return transform(Image.fromarray(image))

    try:
        for start in range(0, len(ids), batch_size):
            part = ids[start : start + batch_size]
            x = torch.stack(
                [input_tensor(images[int(i)]) for i in part]
            ).to(device)
            logits = model(x).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite canonical logits")
            probs[start : start + len(part)] = logits.softmax(1).cpu().numpy()
            if with_features:
                if len(captured) != 1:
                    raise ValueError("Expected exactly one penultimate feature tensor")
                features[start : start + len(part)] = (
                    captured.pop().float().cpu().numpy()
                )
        probs.flush()
        if features is not None:
            features.flush()
    finally:
        if hook is not None:
            hook.remove()
    return features, probs
