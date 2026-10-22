"""Fixed-checkpoint margin under the actual training augmentation distribution."""

import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from room_for_doubt.data.kaggle_artifacts import restore_base
from room_for_doubt.data.observed_images import imagenet_transform


@torch.no_grad()
def augmented_margin(
    bundle, images, out, views, seed, batch_size, device, deadline=None
):
    if not 1 <= views <= 64 or batch_size < views:
        raise ValueError(
            "Require 1..64 augmentations and a batch budget at least as large"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids, labels = bundle["data"]["sample_ids"], bundle["data"]["noisy_labels"]
    native = bundle["config"].get("dataset", {}).get("native_size", 32)
    spec = dict(
        base_digest=bundle["checkpoint_digest"],
        views=int(views),
        seed=int(seed),
        ids=ids.tolist(),
        observed_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        image_size=bundle["config"]["image_size"],
        precision="fp32",
        transform=f"RandomCrop{native}_padding{native // 8}_constant_then_RandomHorizontalFlip_then_imagenet_resize_v1",
    )
    protocol, complete, progress = [
        out / (n + ".json") for n in ("protocol", "complete", "progress")
    ]
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError("Augmented margin protocol differs")
    path = out / "mean_margin.npy"
    if complete.exists():
        value = np.load(path, mmap_mode="r")
        if (
            json.loads(complete.read_text()) != spec
            or value.shape != (len(ids),)
            or not np.isfinite(value).all()
            or np.any(np.abs(value) > 1.00001)
        ):
            raise ValueError("Invalid completed augmented margins")
        return value
    if path.exists() and not protocol.exists():
        raise ValueError("Partial margins require their protocol")
    protocol.write_text(json.dumps(spec))
    value = (
        np.load(path, mmap_mode="r+")
        if path.exists()
        else np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=(len(ids),)
        )
    )
    if value.shape != (len(ids),) or value.dtype != np.float32:
        raise ValueError("Partial margin dimensions differ")
    start = json.loads(progress.read_text())["rows"] if progress.exists() else 0
    if not isinstance(start, int) or not 0 <= start <= len(ids):
        raise ValueError("Invalid augmented margin resume position")
    with torch.random.fork_rng(devices=[]):
        model = restore_base(bundle, device)
    transform = imagenet_transform(
        bundle["config"]["image_size"], train=True, native_size=native
    )
    # batch_size counts expanded views, not source images.
    per_batch = min(batch_size, 64) // views
    for row in range(start, len(ids), per_batch):
        stop = min(row + per_batch, len(ids))
        tensors = []
        for index in range(row, stop):
            for draw in range(views):
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(
                        (int(seed) + int(ids[index]) * 1000003 + draw) % (2**63 - 1)
                    )
                    tensors.append(transform(Image.fromarray(images[int(ids[index])])))
        p = model(torch.stack(tensors).to(device)).float().softmax(1)
        y = torch.as_tensor(np.repeat(labels[row:stop], views), device=device)
        observed = p[torch.arange(len(y), device=device), y].clone()
        p[torch.arange(len(y), device=device), y] = -1
        value[row:stop] = (
            (observed - p.max(1).values).reshape(-1, views).mean(1).cpu().numpy()
        )
        value.flush()
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(rows=stop)))
        temporary.replace(progress)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Augmented margin budget exhausted; cache can resume")
    if not np.isfinite(value).all() or np.any(np.abs(value) > 1.00001):
        raise ValueError("Invalid augmented margin values")
    complete.write_text(json.dumps(spec))
    return value
