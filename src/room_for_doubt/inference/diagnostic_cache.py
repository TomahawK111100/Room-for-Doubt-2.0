"""Bounded forward passes for static supervision and numerical reconciliation."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from room_for_doubt.data.observed_images import ObservedImages, imagenet_transform
from room_for_doubt.data.kaggle_artifacts import restore_base
from room_for_doubt.reporting.artifacts import write_csv


@torch.no_grad()
def image_predictions(bundle, images, ids, batch_size, device, amp=False):
    ids = np.asarray(ids)
    dataset = ObservedImages(
        images,
        np.zeros(len(images), dtype=np.int64),
        ids,
        imagenet_transform(bundle["config"]["image_size"]),
    )
    model = restore_base(bundle, device)
    result = []
    for batch, (x, _, _) in enumerate(
        DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ):
        with torch.autocast(
            device_type=device.type, enabled=amp and device.type == "cuda"
        ):
            logits = model(x.to(device))
        result.append(logits.float().softmax(1).cpu().numpy())
        if batch % 50 == 0:
            print(
                f"Base {'AMP' if amp else 'FP32'} forward: {min((batch + 1) * batch_size, len(ids))}/{len(ids)}",
                flush=True,
            )
    del model
    return np.concatenate(result)


def best_train_predictions(bundle, raw, out, cfg, device):
    path = Path(out) / "artifacts/best_train_predictions.npz"
    ids = bundle["data"]["sample_ids"]
    if path.exists():
        with np.load(path) as f:
            if (
                not np.array_equal(f["ids"], ids)
                or f["base_digest"].item() != bundle["checkpoint_digest"]
            ):
                raise ValueError("Static-target cache differs from base/IDs")
            p = f["probabilities"]
    else:
        p = image_predictions(bundle, raw.data, ids, cfg.image_batch_size, device)
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            temporary,
            ids=ids,
            probabilities=p,
            base_digest=np.array(bundle["checkpoint_digest"]),
        )
        temporary.replace(path)
    if (
        p.shape != (len(ids), cfg.dataset.num_classes)
        or not np.isfinite(p).all()
        or not np.allclose(p.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid static-target probabilities")
    labels = bundle["data"]["noisy_labels"]
    observed = p[np.arange(len(p)), labels]
    other = p.copy()
    other[np.arange(len(p)), labels] = -1
    return p, (observed - other.max(1)).astype("float32")


def align_saved_probabilities(path, ids):
    with np.load(path) as f:
        old_ids, p = f["ids"], f["probabilities"]
        if len(np.unique(old_ids)) != len(old_ids) or set(old_ids.tolist()) != set(
            ids.tolist()
        ):
            raise ValueError("Original prediction IDs differ from new cache")
        order = {int(s): i for i, s in enumerate(old_ids)}
        return p[[order[int(s)] for s in ids]]


def compare_probabilities(name, reference, candidate, labels):
    if reference.shape != candidate.shape:
        raise ValueError("Prediction shapes differ")
    delta = np.abs(reference - candidate)
    disagreements = reference.argmax(1) != candidate.argmax(1)
    return dict(
        comparison=name,
        samples=len(labels),
        prediction_disagreements=int(disagreements.sum()),
        reference_acc=float((reference.argmax(1) == labels).mean()),
        candidate_acc=float((candidate.argmax(1) == labels).mean()),
        max_probability_difference=float(delta.max()),
        mean_probability_difference=float(delta.mean()),
    )


def numerical_audit(bundle, post, out, test, cfg, device):
    cache = Path(post) / "cache"
    ids = np.load(cache / "test_ids.npy")
    labels = np.load(cache / "test_labels.npy")
    cached = np.load(cache / "test_probs.npy", mmap_mode="r")[:, 0]
    rows = []
    reruns = {}
    for amp in (False, True):
        name = "amp" if amp else "fp32"
        path = Path(out) / "artifacts" / f"audit_{name}.npz"
        if path.exists():
            with np.load(path) as f:
                if not np.array_equal(f["ids"], ids):
                    raise ValueError("Audit ID mismatch")
                reruns[name] = f["probabilities"]
        else:
            reruns[name] = image_predictions(
                bundle, test.data, ids, cfg.image_batch_size, device, amp
            )
            np.savez_compressed(path, ids=ids, probabilities=reruns[name])
        rows.append(
            compare_probabilities(
                "cached_fp32_vs_rerun_" + name, cached, reruns[name], labels
            )
        )
    original = bundle["source"] / "artifacts/test_predictions.npz"
    if original.exists():
        source = align_saved_probabilities(original, ids)
        rows.append(
            compare_probabilities("original_vs_cached_fp32", source, cached, labels)
        )
        for name, p in reruns.items():
            rows.append(
                compare_probabilities("original_vs_rerun_" + name, source, p, labels)
            )
    write_csv(Path(out) / "tables/numerical_audit.csv", rows)
    return rows
