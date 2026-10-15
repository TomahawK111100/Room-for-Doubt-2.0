"""Reuse exactly the same base predictions and augmented views across methods."""

from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
from hydra.utils import to_absolute_path
from room_for_doubt.models.classifier import ResNetClassifier
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor
from room_for_doubt.data.preprocessing import meta_transform
from room_for_doubt.data.cifar_noisy import CIFAR_MEAN, CIFAR_STD


def make_views(image, count, seed):
    rng = np.random.default_rng(seed)
    result = [image]
    for _ in range(count - 1):
        view = image[:, ::-1] if rng.random() < 0.5 else image
        view = np.pad(view, ((4, 4), (4, 4), (0, 0)), mode="reflect")
        y, x = rng.integers(0, 9, size=2)
        result.append(view[y : y + 32, x : x + 32].copy())
    return result


def completed_rows(probs, features):
    valid = np.all(np.isclose(probs.sum(axis=-1), 1.0, atol=1e-5), axis=1)
    valid &= np.isfinite(features).all(axis=(1, 2)) & (features != 0).any(axis=(1, 2))
    missing = np.flatnonzero(~valid)
    end = int(missing[0]) if len(missing) else len(valid)
    if valid[end:].any():
        raise ValueError("Partial cache must be a contiguous completed prefix")
    return end


@torch.no_grad()
def build_cache(cfg, data, device):
    if (
        cfg.base_normalization != "historical_cifar10"
        or cfg.meta_preprocessing != "imagenet_pil_bilinear_224"
    ):
        raise ValueError("Fixed base/meta preprocessing mismatch")
    cache = Path(to_absolute_path(cfg.cache_dir))
    cache.mkdir(parents=True, exist_ok=True)
    count = 1 if cfg.dataset.name == "cifar10n" else cfg.views
    spec = dict(
        dataset=cfg.dataset.name,
        base_checkpoint=to_absolute_path(cfg.base_checkpoint),
        views=count,
        inference_seed=cfg.inference_seed,
        base_normalization=cfg.base_normalization,
        meta_preprocessing=cfg.meta_preprocessing,
    )
    if (cache / "complete.json").exists():
        if json.loads((cache / "complete.json").read_text()) != spec:
            raise ValueError("Existing inference cache uses a different protocol")
        return cache
    partial = cache / "cache_protocol.json"
    if not partial.exists() and (cache / "val_probs.npy").exists():
        raise ValueError(
            "Existing partial arrays require their original cache protocol"
        )
    if partial.exists() and json.loads(partial.read_text()) != spec:
        raise ValueError("Partial inference cache uses a different protocol")
    partial.write_text(json.dumps(spec, indent=2))
    ckpt = torch.load(spec["base_checkpoint"], map_location="cpu", weights_only=False)
    if ckpt["hyper_parameters"]["num_classes"] != cfg.dataset.num_classes:
        raise ValueError("Wrong base checkpoint class count")
    base_model = ResNetClassifier(num_classes=cfg.dataset.num_classes, pretrained=False)
    base_model.load_state_dict(
        {k.removeprefix("_orig_mod."): v for k, v in ckpt["state_dict"].items()}
    )
    base_model.to(device).eval()
    meta_model = ConditionedPredictor(cfg.dataset.num_classes, pretrained=False)
    meta_model.features.load_state_dict(
        torch.load(
            Path(to_absolute_path(cfg.feature_path)) / "backbone.pt", weights_only=True
        )
    )
    meta_model.to(device).eval()
    if cfg.memory_format != "channels_last":
        raise ValueError("Fixed campaign uses channels_last inference memory layout")
    base_model.to(memory_format=torch.channels_last)
    meta_model.to(memory_format=torch.channels_last)
    transform = meta_transform()
    mean = torch.tensor(CIFAR_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(CIFAR_STD, device=device)[None, :, None, None]
    cls = {"cifar10n": CIFAR10, "cifar100n": CIFAR100}[cfg.dataset.name]
    for split, train in [("val", True), ("test", False)]:
        dataset = cls(
            to_absolute_path(cfg.dataset.data_dir), train=train, download=False
        )
        ids = data["base_val_ids"] if train else np.arange(len(dataset))
        reuse = (cache / f"{split}_probs.npy").exists() and (
            cache / f"{split}_features.npy"
        ).exists()
        probs = np.lib.format.open_memmap(
            cache / f"{split}_probs.npy",
            mode="r+" if reuse else "w+",
            dtype="float32",
            shape=(len(ids), count, cfg.dataset.num_classes),
        )
        feats = np.lib.format.open_memmap(
            cache / f"{split}_features.npy",
            mode="r+" if reuse else "w+",
            dtype="float32",
            shape=(len(ids), count, 512),
        )
        # Batch size is the number of views to cap 224x224 activation memory.
        image_batch = max(1, cfg.batch_size // count)
        if probs.shape != (len(ids), count, cfg.dataset.num_classes) or feats.shape != (
            len(ids),
            count,
            512,
        ):
            raise ValueError("Partial cache array shape mismatch")
        completed = completed_rows(probs, feats) if reuse else 0
        print(
            f"{split}: reusing {completed}/{len(ids)} complete cached images",
            flush=True,
        )
        for start in range(completed, len(ids), image_batch):
            batch_ids = ids[start : start + image_batch]
            views = [
                v
                for sid in batch_ids
                for v in make_views(
                    dataset.data[int(sid)],
                    count,
                    cfg.inference_seed + int(sid) + (100000 if not train else 0),
                )
            ]
            base_images = (
                torch.from_numpy(np.stack(views)).permute(0, 3, 1, 2).to(device).float()
                / 255
            )
            base_images = base_images.contiguous(memory_format=torch.channels_last)
            probabilities = (
                base_model((base_images - mean) / std).softmax(1).cpu().numpy()
            )
            meta_images = torch.stack(
                [transform(Image.fromarray(v)) for v in views]
            ).to(device)
            meta_images = meta_images.contiguous(memory_format=torch.channels_last)
            features = meta_model.extract_features(meta_images).cpu().numpy()
            probs[start : start + len(batch_ids)] = probabilities.reshape(
                len(batch_ids), count, -1
            )
            feats[start : start + len(batch_ids)] = features.reshape(
                len(batch_ids), count, -1
            )
            if start // image_batch % 100 == 0:
                print(
                    f"{split} shared inference {start}/{len(ids)} images x {count} views",
                    flush=True,
                )
        probs.flush()
        feats.flush()
        np.save(cache / f"{split}_ids.npy", ids)
        # Targets are stored for evaluation, never used by inference/selection here.
        np.save(cache / f"{split}_clean_labels.npy", np.asarray(dataset.targets)[ids])
    (cache / "complete.json").write_text(json.dumps(spec, indent=2))
    return cache
