"""Verify annotations, reserve split IDs, and prepare compact training inputs."""

from pathlib import Path
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10, CIFAR100
from room_for_doubt.data.cifar_noisy import load_noisy_labels
from room_for_doubt.data.trajectory_records import (
    historical_trajectory_path,
    load_verified_trajectories,
)
from room_for_doubt.data.meta_splits import reserve_meta_splits, validate_meta_splits
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory


@hydra.main(
    config_path=str(config_directory()),
    config_name="meta_retraining",
    version_base=None,
)
def main(cfg):
    name = cfg.dataset.name.removesuffix("n")
    cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[name]
    root = to_absolute_path(cfg.dataset.data_dir)
    base = cls(root, train=True, download=False)
    noisy, clean = load_noisy_labels(root, cfg.dataset.name, cfg.dataset.noisy_split)
    if not np.array_equal(clean, base.targets):
        raise ValueError(
            "Official annotation clean fields disagree with CIFAR identity"
        )
    source = to_absolute_path(str(historical_trajectory_path(name, cfg.base_seed)))
    verified = load_verified_trajectories(
        source, name, base.targets, cfg.base_seed, cfg.dataset.val_ratio
    )
    for sid, record in verified.records.items():
        if record["noisy_label"] != noisy[int(sid)]:
            raise ValueError(
                "Trajectory annotation does not match requested official noisy split"
            )
    ids, targets, excluded = verified.aligned_targets()
    splits = reserve_meta_splits(ids, cfg.split_seed)
    validate_meta_splits(splits, ids)
    out = Path(HydraConfig.get().runtime.output_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "artifacts/meta_targets.npz",
        sample_ids=ids,
        targets=targets,
        noisy_labels=noisy[ids],
        excluded_ids=excluded,
        base_train_ids=verified.train_ids,
        base_val_ids=verified.val_ids,
        dataset_name=np.array(cfg.dataset.name),
        noisy_split=np.array(cfg.dataset.noisy_split),
        split_seed=np.array(cfg.split_seed),
        **splits,
    )
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    metrics = {k.removesuffix("_ids") + "_n": len(v) for k, v in splits.items()}
    metrics.update(usable_n=len(ids), excluded_n=len(excluded), annotation_mismatches=0)
    meta = dict(
        seed=cfg.split_seed,
        trajectory_source=source,
        checkpoint_path=None,
        device="CPU",
        scope="historical_complete_case_salvage",
        base_checkpoint_trajectory_link="not independently established",
        clean_label_use="identity verification only; no selection or targets",
    )
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "Epochs 0–49 probability margins; complete cases only. Label-independent 60/10/15/15 meta-train/meta-val/detector-fit/detector-eval split. Official noisy annotations match every trajectory record. Base validation stays reserved. Historical checkpoint/trajectory lineage is unresolved: this is a salvage experiment, not a certified new base run.",
    )
    print(out / "artifacts/meta_targets.npz")


if __name__ == "__main__":
    main()
