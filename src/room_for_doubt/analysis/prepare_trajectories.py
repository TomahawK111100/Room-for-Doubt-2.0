"""Prepare epoch-aligned historical targets without training or inference."""

from pathlib import Path
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchvision.datasets import CIFAR10, CIFAR100

from room_for_doubt.data.trajectory_records import (
    historical_trajectory_path,
    load_verified_trajectories,
)
from room_for_doubt.reporting.artifacts import (
    generate_report,
    save_metrics,
    save_run_metadata,
)
from room_for_doubt.utils.paths import config_directory


@hydra.main(
    config_path=str(config_directory()),
    config_name="trajectory_preparation",
    version_base=None,
)
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    names = {"cifar10n": ("cifar10", CIFAR10), "cifar100n": ("cifar100", CIFAR100)}
    if cfg.dataset.name not in names:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
    if cfg.trajectory.margin_definition != "probability_margin":
        raise ValueError(
            "Historical targets contain probability margins; no logit conversion is available"
        )
    name, dataset_cls = names[cfg.dataset.name]
    cfg.source_path = to_absolute_path(
        cfg.source_path or str(historical_trajectory_path(name, cfg.seed))
    )
    out = Path(HydraConfig.get().runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    base = dataset_cls(
        root=to_absolute_path(cfg.dataset.data_dir), train=True, download=False
    )
    verified = load_verified_trajectories(
        cfg.source_path, name, base.targets, cfg.seed, cfg.dataset.val_ratio
    )
    ids, targets, excluded = verified.aligned_targets(
        cfg.trajectory.num_epochs,
        cfg.trajectory.start_epoch,
        cfg.trajectory.missing_epoch_policy,
    )
    artifact_dir = out / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        artifact_dir / "trajectory_targets.npz",
        sample_ids=ids,
        targets=targets,
        noisy_labels=np.asarray([verified.records[str(i)]["noisy_label"] for i in ids]),
        base_train_ids=verified.train_ids,
        val_ids=verified.val_ids,
        excluded_train_ids=excluded,
        epochs=np.arange(
            cfg.trajectory.start_epoch,
            cfg.trajectory.start_epoch + cfg.trajectory.num_epochs,
        ),
    )
    metrics = {
        "base_train_samples": len(verified.train_ids),
        "usable_train_samples": len(ids),
        "excluded_train_samples": len(excluded),
        "validation_samples": len(verified.val_ids),
        "retained_fraction": len(ids) / len(verified.train_ids),
        "target_epochs": cfg.trajectory.num_epochs,
        "train_val_overlap": len(set(ids) & set(verified.val_ids)),
    }
    config = OmegaConf.to_container(cfg, resolve=True)
    save_metrics(out, metrics)
    metadata = dict(
        seed=cfg.seed,
        checkpoint_path=None,
        device="CPU data preparation",
        trajectory_source=cfg.source_path,
        label_verification="clean field against local CIFAR identity; noisy annotation variant not verified",
        model_training=False,
        model_inference=False,
        missing_epoch_policy=cfg.trajectory.missing_epoch_policy,
    )
    save_run_metadata(out, cfg.run_id, config, **metadata)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        metadata,
        "Historical probability margins aligned by explicit epoch IDs. Complete-case selection uses only epoch availability; excluded training IDs are not moved to validation. This selection changes the meta/detector training population and may introduce selection bias. No values are interpolated or imputed. No model training, detector fitting or inference was performed; old meta checkpoints are not certified for these targets. Dataset noisy_split is a requested configuration field, not verified artifact provenance.",
    )
    print(
        f"Prepared {len(ids)} complete trajectories; excluded {len(excluded)}. Output: {out}"
    )


if __name__ == "__main__":
    main()
