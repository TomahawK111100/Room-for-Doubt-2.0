"""Frozen classifier penultimate features and train-only PCA projection."""

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from room_for_doubt.data.observed_images import imagenet_transform
from room_for_doubt.data.kaggle_artifacts import restore_base
from room_for_doubt.evaluation.campaign_cache import make_views


@torch.no_grad()
def cache_classifier_features(
    bundle, images, ids, split, out, cfg, device, reference=None
):
    root = Path(out) / "native_features"
    root.mkdir(exist_ok=True)
    count = 1 if split == "train" else max(cfg.view_counts)
    spec = dict(
        base_digest=cfg.base_digest,
        split=split,
        ids=np.asarray(ids).tolist(),
        views=int(count),
        inference_seed=cfg.inference_seed,
        image_size=bundle["config"]["image_size"],
        precision="fp32",
    )
    marker = root / f"{split}_complete.json"
    progress = root / f"{split}_progress.json"
    protocol = root / f"{split}_protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError("Native-feature cache belongs to another run")
    protocol.write_text(json.dumps(spec))
    if marker.exists():
        if json.loads(marker.read_text()) != spec:
            raise ValueError("Completed native-feature cache differs")
        return np.load(root / f"{split}_features.npy", mmap_mode="r"), np.load(
            root / f"{split}_probs.npy", mmap_mode="r"
        )
    model = restore_base(bundle, device)
    captured = []
    hook = model.fc.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach())
    )
    transform = imagenet_transform(bundle["config"]["image_size"])
    classes, width = model.fc.out_features, model.fc.in_features

    def array(name, shape):
        p = root / f"{split}_{name}.npy"
        if p.exists():
            value = np.load(p, mmap_mode="r+")
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError("Native feature cache dimensions differ")
            return value
        return np.lib.format.open_memmap(p, mode="w+", dtype="float32", shape=shape)

    features = array("features", (len(ids), count, width))
    probabilities = array("probs", (len(ids), count, classes))
    start = int(json.loads(progress.read_text())["rows"]) if progress.exists() else 0
    if not 0 <= start <= len(ids):
        raise ValueError("Invalid native-feature resume position")
    images_per_batch = max(1, cfg.image_batch_size // count)
    try:
        for row in range(start, len(ids), images_per_batch):
            batch_ids = ids[row : row + images_per_batch]
            views = [
                Image.fromarray(v)
                for sid in batch_ids
                for v in make_views(
                    images[int(sid)],
                    count,
                    cfg.inference_seed + int(sid) + (100000 if split == "test" else 0),
                )
            ]
            logits = model(torch.stack([transform(v) for v in views]).to(device))
            if len(captured) != 1:
                raise ValueError("Expected one classifier-head input")
            f = captured.pop().float().cpu().numpy()
            p = logits.float().softmax(1).cpu().numpy()
            features[row : row + len(batch_ids)] = f.reshape(
                len(batch_ids), count, width
            )
            probabilities[row : row + len(batch_ids)] = p.reshape(
                len(batch_ids), count, classes
            )
            features.flush()
            probabilities.flush()
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(rows=row + len(batch_ids))))
            temporary.replace(progress)
            if row % max(images_per_batch * 50, 1) == 0:
                print(f"Classifier features {split}: {row}/{len(ids)}", flush=True)
    finally:
        hook.remove()
    if not np.isfinite(features).all() or not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite classifier features/probabilities")
    if reference is not None:
        delta = float(np.max(np.abs(probabilities - reference[:, :count])))
        (root / f"{split}_alignment.json").write_text(
            json.dumps(
                dict(
                    max_probability_difference=delta,
                    prediction_disagreements=int(
                        (
                            probabilities.argmax(-1) != reference[:, :count].argmax(-1)
                        ).sum()
                    ),
                )
            )
        )
        if delta > 1e-3:
            raise ValueError(
                "Classifier feature extraction does not reproduce source probabilities"
            )
    marker.write_text(json.dumps(spec))
    return features, probabilities


def fit_projection(features, fit_rows, path, dimensions, seed):
    """PCA sees meta-training images only; no labels, validation or test inputs."""
    path = Path(path)
    spec = dict(
        dimensions=int(dimensions),
        seed=int(seed),
        fit_rows=np.asarray(fit_rows).tolist(),
        input_features=int(features.shape[-1]),
    )
    marker = path.with_suffix(".json")
    if path.exists():
        if json.loads(marker.read_text()) != spec:
            raise ValueError("PCA fit partition or dimensions differ")
        with np.load(path) as f:
            return dict(
                mean=f["mean"],
                components=f["components"],
                explained_variance_ratio=f["explained_variance_ratio"],
            )
    if dimensions > min(len(fit_rows), features.shape[-1]):
        raise ValueError("PCA output dimension exceeds available meta-training rank")
    pca = PCA(
        n_components=dimensions,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=3,
    )
    pca.fit(np.asarray(features[fit_rows], dtype=np.float32))
    result = dict(
        mean=pca.mean_,
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    np.savez_compressed(path, **result)
    marker.write_text(json.dumps(spec))
    return result


def project_features(features, projection, path):
    path = Path(path)
    shape = (*features.shape[:-1], len(projection["components"]))
    marker = path.with_suffix(".complete.json")
    if marker.exists():
        value = np.load(path, mmap_mode="r")
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid projected features")
        return value
    value = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)
    source = features.reshape(-1, features.shape[-1])
    target = value.reshape(-1, shape[-1])
    for start in range(0, len(source), 2048):
        target[start : start + 2048] = (
            source[start : start + 2048] - projection["mean"]
        ) @ projection["components"].T
    value.flush()
    marker.write_text(json.dumps(dict(shape=shape)))
    return value
