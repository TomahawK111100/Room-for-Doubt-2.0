"""Shared deterministic views for the pretrained 224px base and meta encoder."""

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from room_for_doubt.data.observed_images import ObservedImages, imagenet_transform
from room_for_doubt.data.preprocessing import meta_transform
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor
from room_for_doubt.evaluation.campaign_cache import make_views
from room_for_doubt.data.kaggle_artifacts import restore_base, checkpoint_digest


def frozen_encoder(classes, backbone=None):
    model = ConditionedPredictor(classes, pretrained=backbone is None)
    if backbone is not None:
        model.features.load_state_dict(
            torch.load(backbone, map_location="cpu", weights_only=True)
        )
    return model.eval()


@torch.no_grad()
def prepare_features(bundle, images, root, cfg, device):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = dict(
        base_digest=bundle["checkpoint_digest"],
        preprocessing="imagenet_pil_bilinear_224",
        sample_ids=bundle["data"]["sample_ids"].tolist(),
    )
    marker = root / "complete.json"
    if marker.exists():
        if json.loads(marker.read_text()) != spec:
            raise ValueError("Train-feature cache belongs to a different base/split")
        return
    model = frozen_encoder(bundle["config"]["dataset"]["num_classes"]).to(device)
    ids = bundle["data"]["sample_ids"]
    labels = np.zeros(len(images), dtype=np.int64)
    labels[ids] = bundle["data"]["noisy_labels"]
    ds = ObservedImages(images, labels, ids, meta_transform())
    features = []
    for batch, (x, _, _) in enumerate(
        DataLoader(
            ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
        )
    ):
        features.append(model.extract_features(x.to(device)).cpu().numpy())
        if batch % 25 == 0:
            print(
                f"Frozen features: {min((batch + 1) * cfg.batch_size, len(ds))}/{len(ds)}",
                flush=True,
            )
    np.savez_compressed(
        root / "features.npz",
        features=np.concatenate(features),
        sample_ids=ids,
        dataset_name=bundle["data"]["dataset_name"],
        preprocessing=np.array("imagenet_pil_bilinear_224"),
    )
    torch.save(model.features.cpu().state_dict(), root / "backbone.pt")
    marker.write_text(json.dumps(spec))


@torch.no_grad()
def split_cache(bundle, images, ids, labels, split, root, feature_root, cfg, device):
    if split not in {"val", "test"}:
        raise ValueError("Unknown inference split")
    if cfg.views < 1 or cfg.image_batch_size < 1:
        raise ValueError("Positive views/batch required")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    backbone = Path(feature_root) / "backbone.pt"
    spec = dict(
        base_digest=bundle["checkpoint_digest"],
        encoder_digest=checkpoint_digest(backbone),
        views=int(cfg.views),
        seed=int(cfg.inference_seed),
        image_size=bundle["config"]["image_size"],
        ids=np.asarray(ids).tolist(),
    )
    protocol = root / f"{split}_protocol.json"
    complete = root / f"{split}_complete.json"
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError("Inference cache protocol mismatch")
    protocol.write_text(json.dumps(spec))
    if complete.exists():
        if json.loads(complete.read_text()) != spec:
            raise ValueError("Completed cache mismatch")
        if not np.array_equal(
            np.load(root / f"{split}_labels.npy"), np.asarray(labels)[ids]
        ):
            raise ValueError("Cached labels mismatch")
        return
    classes = bundle["config"]["dataset"]["num_classes"]
    n = len(ids)

    def array(name, shape):
        path = root / f"{split}_{name}.npy"
        if path.exists():
            value = np.load(path, mmap_mode="r+")
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError("Partial cache dimensions mismatch")
            return value
        return np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)

    probs = array("probs", (n, cfg.views, classes))
    features = array("features", (n, cfg.views, 512))
    progress = root / f"{split}_progress.json"
    start = int(json.loads(progress.read_text())["rows"]) if progress.exists() else 0
    if not 0 <= start <= n:
        raise ValueError("Invalid cache progress")
    base = restore_base(bundle, device)
    encoder = frozen_encoder(classes, backbone).to(device)
    base_tf = imagenet_transform(bundle["config"]["image_size"])
    meta_tf = meta_transform()
    # image_batch_size counts source images here; each expands into cfg.views
    # full-resolution tensors. Bound the expanded batch, including on resume.
    images_per_batch = min(cfg.image_batch_size, max(1, 64 // cfg.views))
    print(f"{split} cache: at most {images_per_batch * cfg.views} views per forward", flush=True)
    for row in range(start, n, images_per_batch):
        batch_ids = ids[row : row + images_per_batch]
        views = [
            Image.fromarray(v)
            for sid in batch_ids
            for v in make_views(
                images[int(sid)],
                cfg.views,
                cfg.inference_seed + int(sid) + (100000 if split == "test" else 0),
            )
        ]
        x = torch.stack([base_tf(v) for v in views]).to(device)
        probs[row : row + len(batch_ids)] = (
            base(x)
            .float()
            .softmax(1)
            .cpu()
            .numpy()
            .reshape(len(batch_ids), cfg.views, classes)
        )
        x = torch.stack([meta_tf(v) for v in views]).to(device)
        features[row : row + len(batch_ids)] = (
            encoder.extract_features(x)
            .cpu()
            .numpy()
            .reshape(len(batch_ids), cfg.views, 512)
        )
        probs.flush()
        features.flush()
        tmp = progress.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(rows=row + len(batch_ids))))
        tmp.replace(progress)
        if row % max(images_per_batch * 20, 1) == 0:
            print(f"{split} shared views: {row}/{n}", flush=True)
    if (
        not np.isfinite(probs).all()
        or not np.allclose(probs.sum(-1), 1, atol=1e-5)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid cache values")
    np.save(root / f"{split}_ids.npy", ids)
    np.save(root / f"{split}_labels.npy", np.asarray(labels)[ids])
    complete.write_text(json.dumps(spec))
