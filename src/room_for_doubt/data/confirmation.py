"""Immutable inputs and validation-only handoff for S4."""

import json
from pathlib import Path

from omegaconf import OmegaConf
from room_for_doubt.training.temporal_snapshots import file_digest

SEEDS = (42, 43, 44)
METHODS = ("base", "early5", "direct_soft5", "early5_teacher")
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
PROTOCOL = "confirmation_v1"
FROZEN_FILES = (
    "config.yaml",
    "base/config.yaml",
    "base/metrics.json",
    "base/checkpoints/best.pt",
    "base/checkpoints/temporal/manifest.json",
    *(f"base/checkpoints/temporal/epoch_{n:03}.pt" for n in range(1, 6)),
    "base/artifacts/meta_targets.npz",
    "base/artifacts/split_ids.npz",
    "base/history.csv",
    "evidence/manifest.json",
    *(
        f"{method}/{file}"
        for method in METHODS[1:3]
        for file in (
            "artifacts/policy_before_test.json",
            "checkpoints/best.pt",
            "metrics.json",
        )
    ),
    "runtime/config.yaml",
    "runtime/samples.json",
    "runtime/tables/runtime.csv",
    "runtime/metrics.json",
    "runtime/run_metadata.json",
)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def check_base_config(config, seed):
    expected = {
        "seed": seed,
        "base_split_seed": 42,
        "split_seed": 20261015,
        "epochs": 50,
        "image_size": 224,
        "batch_size": 128,
        "temporal_snapshot_epochs": [1, 2, 3, 4, 5],
        "defer_test_evaluation": True,
        "model": {"name": "resnet50", "weights": "IMAGENET1K_V2"},
        "dataset": {
            "name": "cifar10n",
            "noisy_split": "worse",
            "num_classes": 10,
            "val_ratio": 0.1,
        },
        "loss": {"q": 0.7, "ce_warmup_epochs": 5},
        "optimizer": {
            "name": "SGD",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
        },
    }
    if seed not in SEEDS or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Base config differs from the frozen S4 recipe")


def verify_frozen(root, seed, protocol_hash):
    root = Path(root)
    manifest = json.loads((root / "frozen.json").read_text())
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("seed") != seed
        or manifest.get("protocol_sha256") != protocol_hash
        or manifest.get("test_accessed") is not False
        or set(manifest.get("files", ())) != set(FROZEN_FILES)
    ):
        raise ValueError(f"Invalid S4 manifest for seed {seed}")
    for name, digest in manifest["files"].items():
        if file_digest(root / name) != digest:
            raise ValueError(f"Frozen file changed for seed {seed}: {name}")
    config = OmegaConf.to_container(
        OmegaConf.load(root / "base/config.yaml"), resolve=True
    )
    check_base_config(config, seed)
    base_metrics = json.loads((root / "base/metrics.json").read_text())
    if (
        base_metrics.get("status") != "complete"
        or base_metrics.get("test_evaluated") is not False
    ):
        raise ValueError("S4 requires a completed base with deferred evaluation")
    return manifest


def seal_frozen(root, seed, protocol_hash):
    root = Path(root)
    path = root / "frozen.json"
    if not path.exists():
        write_json(
            path,
            {
                "protocol": PROTOCOL,
                "seed": seed,
                "protocol_sha256": protocol_hash,
                "test_accessed": False,
                "files": {name: file_digest(root / name) for name in FROZEN_FILES},
            },
        )
    return verify_frozen(root, seed, protocol_hash)
