"""Cheap validation-only alternatives to learned badness weighting."""

from pathlib import Path
import csv
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from room_for_doubt.data.cifar_noisy import load_noisy_labels
from room_for_doubt.evaluation.selection import weighted_consensus
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory


@hydra.main(
    config_path=str(config_directory()),
    config_name="confidence_diagnostic",
    version_base=None,
)
def main(cfg):
    if cfg.label_source != "noisy_validation_only":
        raise ValueError("Diagnostic selection uses noisy validation only")
    cache = Path(to_absolute_path(cfg.cache_dir))
    probs = np.load(cache / "val_probs.npy", mmap_mode="r")
    ids = np.load(cache / "val_ids.npy")
    noisy, _ = load_noisy_labels(
        to_absolute_path(cfg.data_dir), cfg.dataset.name, cfg.dataset.noisy_split
    )
    labels = noisy[ids]
    uniform = float(np.mean(probs.mean(1).argmax(1) == labels))
    rows = [
        dict(
            method="uniform", temperature="uniform", val_noisy_acc=uniform, delta_pp=0.0
        )
    ]
    scores = {
        "msp": 1 - probs.max(2),
        "entropy": -(probs * np.log(np.maximum(probs, 1e-12))).sum(2)
        / np.log(probs.shape[2]),
    }
    for method, badness in scores.items():
        for temperature in cfg.temperatures:
            pred = weighted_consensus(probs, badness, temperature).argmax(1)
            acc = float(np.mean(pred == labels))
            rows.append(
                dict(
                    method=method,
                    temperature=float(temperature),
                    val_noisy_acc=acc,
                    delta_pp=(acc - uniform) * 100,
                )
            )
    best = max(rows, key=lambda r: r["val_noisy_acc"])
    out = Path(HydraConfig.get().runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    metrics = dict(
        validation_samples=len(ids),
        uniform_val_noisy_acc=uniform,
        best_method=best["method"],
        best_temperature=best["temperature"],
        best_val_noisy_acc=best["val_noisy_acc"],
        best_delta_pp=best["delta_pp"],
        test_evaluated=False,
    )
    meta = dict(
        checkpoint_path=None,
        device="CPU cached validation predictions",
        supervision="noisy validation labels",
        initialization_seeds="not applicable; no new model fitted",
    )
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "Validation-only exploratory diagnostic after the fixed campaign. Uses identical 20 cached views and no test arrays. Entropy is divided by the known maximum log(number of classes) for weight scaling; raw accuracy is not normalized. The strongest of 15 candidates is a selection result, not an independent performance estimate or evidence of test accuracy improvement. No model training or new image inference.",
    )
    with (out / "tables/summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(metrics)


if __name__ == "__main__":
    main()
