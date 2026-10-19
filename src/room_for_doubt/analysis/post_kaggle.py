"""Prepare -> train -> validate -> evaluate the completed Kaggle base artifacts."""

import json
import shutil
from pathlib import Path
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10, CIFAR100
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.kaggle_artifacts import load_completed_run
from room_for_doubt.inference.post_kaggle_cache import prepare_features, split_cache
from room_for_doubt.training.post_kaggle_meta import train_heads
from room_for_doubt.evaluation.post_kaggle import freeze_policies, summarize_test
from room_for_doubt.training.meta_retraining import choose_device
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory


def run_pipeline(cfg, out):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if cfg.optimizer != "adam" or cfg.meta_model != "conditioned":
        raise ValueError("This comparison fixes Adam and conditioned MLP heads")
    if any(not np.isfinite(t) or t <= 0 for t in cfg.temperatures):
        raise ValueError("Temperatures must be finite and positive")
    if cfg.stage not in {"all", "prepare", "train", "evaluate"} or not cfg.source_run:
        raise ValueError("Set source_run and stage=all|prepare|train|evaluate")
    if (
        list(cfg.modes) != ["mean", "full"]
        or len(set(cfg.seeds)) != len(cfg.seeds)
        or not len(cfg.seeds)
    ):
        raise ValueError("Require mean/full modes and unique nonempty seeds")
    if cfg.epochs < 1 or cfg.views < 1 or cfg.batch_size < 1:
        raise ValueError("Positive epochs/views/batch required")
    source = Path(to_absolute_path(cfg.source_run))
    source_config = OmegaConf.load(source / "config.yaml")
    data_dir = to_absolute_path(cfg.data_dir)
    noisy = load_observed_labels(
        data_dir, source_config.dataset.name, source_config.dataset.noisy_split
    )
    bundle = load_completed_run(source, noisy)
    if cfg.split_seed != bundle["config"]["split_seed"]:
        raise ValueError("Use source meta split seed")
    device = choose_device(cfg.device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < cfg.min_free_gb * 1024**3:
        raise RuntimeError(
            "Insufficient output disk space: use Kaggle working storage or free space first"
        )
    (out / "artifacts").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    cfg_path = out / "config.yaml"
    if cfg_path.exists():
        previous = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        current = OmegaConf.to_container(cfg, resolve=True)
        for key in [
            "seeds",
            "modes",
            "epochs",
            "batch_size",
            "learning_rate",
            "split_seed",
            "views",
            "inference_seed",
            "temperatures",
        ]:
            if previous[key] != current[key]:
                raise ValueError("Existing pipeline scientific config differs: " + key)
    config = OmegaConf.to_container(cfg, resolve=True)
    config["dataset"] = bundle["config"]["dataset"]
    config["base_checkpoint"] = str(source / "checkpoints/best.pt")
    config["base_image_size"] = bundle["config"]["image_size"]
    manifest = out / "artifacts/source.json"
    identity = dict(
        base_digest=bundle["checkpoint_digest"],
        source=str(source.resolve()),
        dataset=bundle["config"]["dataset"]["name"],
    )
    if manifest.exists():
        previous_source = json.loads(manifest.read_text())
        if any(previous_source[k] != identity[k] for k in ["base_digest", "dataset"]):
            raise ValueError("Output directory belongs to another source")
    OmegaConf.save(OmegaConf.create(config), cfg_path, resolve=True)
    manifest.write_text(json.dumps(identity, indent=2))
    cls = {"cifar10n": CIFAR10, "cifar100n": CIFAR100}[identity["dataset"]]
    raw = cls(data_dir, train=True, download=True)
    meta = dict(
        seed=list(cfg.seeds),
        device=str(device),
        checkpoint_path=str(source / "checkpoints/best.pt"),
        best_epoch=bundle["checkpoint"]["epoch"],
        best_val_loss=bundle["checkpoint"]["val_loss"],
        base_digest=bundle["checkpoint_digest"],
        source_config=bundle["config"],
    )
    if cfg.stage in {"all", "prepare"}:
        np.savez_compressed(out / "artifacts/meta_targets.npz", **bundle["data"])
        prepare_features(bundle, raw.data, out / "features", cfg, device)
    if cfg.stage in {"all", "train"}:
        train_heads(cfg, bundle, out, device)
    metrics = dict(status=cfg.stage + "_complete", test_evaluated=False)
    if cfg.stage in {"all", "evaluate"}:
        split_cache(
            bundle,
            raw.data,
            bundle["data"]["base_val_ids"],
            noisy,
            "val",
            out / "cache",
            out / "features",
            cfg,
            device,
        )
        frozen = freeze_policies(cfg, bundle, out, device)
        # Test dataset is first instantiated after the full policy marker exists.
        test = cls(data_dir, train=False, download=True)
        split_cache(
            bundle,
            test.data,
            np.arange(len(test.data)),
            np.asarray(test.targets),
            "test",
            out / "cache",
            out / "features",
            cfg,
            device,
        )
        metrics = summarize_test(cfg, bundle, out, device, frozen)
        metrics["test_evaluated"] = True
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "New pretrained base only, complete 50-epoch source verification. Same view cache across base/TTA/MSP/mean/full. Every temperature frozen on noisy validation before test caching/correctness; no test-based method selection. Mean/full meta MSE uses separate reserved meta validation. One base seed, three meta initializations by default; conditional paired intervals do not establish base-seed robustness. Source Kaggle base was already test-evaluated, so this is not an untouched confirmatory holdout. AUROC/AUPRC always refer to common original-view base errors; accuracy/ECE refer to each final predictor. AURC omitted because of known tie-order ambiguity. Full and scalar heads differ in output parameters. Scalar mean still uses history supervision.",
    )
    if metrics.get("aggregate"):
        with (out / "report.md").open("a") as f:
            f.write(
                "\n## Shared-view comparison\n\n| Method | Accuracy, % | Error AUROC on base | Gain vs TTA, pp |\n|---|---:|---:|---:|\n"
            )
            for row in metrics["aggregate"]:
                f.write(
                    f"| {row['method']} | {100 * row['acc']:.2f} | {row['base_error_auroc']:.4f} | {row['gain_vs_tta_pp']:.2f} |\n"
                )
    return metrics


@hydra.main(
    config_path=str(config_directory()), config_name="post_kaggle", version_base=None
)
def main(cfg):
    run_pipeline(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
