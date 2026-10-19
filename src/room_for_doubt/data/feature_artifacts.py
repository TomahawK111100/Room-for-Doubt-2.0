"""Read-only identity checks for a completed feature study and shared caches."""

import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf


def completed_study(study):
    study = Path(study)
    cfg = OmegaConf.load(study / "config.yaml")
    metrics = json.loads((study / "metrics.json").read_text())
    frozen = json.loads((study / "artifacts/policies_before_test.json").read_text())
    if metrics.get("status") != "complete" or frozen["base_digest"] != cfg.base_digest:
        raise ValueError("Require a complete, matching feature study")
    return cfg, frozen


def shared_split(study, cache_run, cfg, split):
    study, cache = Path(study), Path(cache_run) / "cache"
    protocol = json.loads((cache / f"{split}_complete.json").read_text())
    feature_source = (
        Path(cfg.study_run) if cfg.get("scope") == "augmentation_control" else study
    )
    native = json.loads(
        (feature_source / "native_features" / f"{split}_complete.json").read_text()
    )
    ids = np.load(cache / f"{split}_ids.npy")
    if (
        protocol["base_digest"] != cfg.base_digest
        or native["base_digest"] != cfg.base_digest
        or native["precision"] != "fp32"
        or protocol["seed"] != cfg.inference_seed
        or native["inference_seed"] != cfg.inference_seed
        or native["views"] != max(cfg.view_counts)
        or protocol["views"] < max(cfg.view_counts)
        or protocol["image_size"] != native["image_size"]
        or not np.array_equal(ids, protocol["ids"])
        or not np.array_equal(ids, native["ids"])
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Shared cache / feature IDs or provenance differ")
    p = np.load(cache / f"{split}_probs.npy", mmap_mode="r")
    labels = np.load(cache / f"{split}_labels.npy")
    if (
        p.shape != (len(ids), protocol["views"], cfg.dataset.num_classes)
        or labels.shape != ids.shape
    ):
        raise ValueError("Shared cache dimensions differ")
    return p[:, : max(cfg.view_counts)], labels, ids
