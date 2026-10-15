"""Bounded cached-view study; all choices use observed validation labels."""

import csv
import json
from pathlib import Path
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.evaluation.corrected_campaign import predict_arrays
from room_for_doubt.evaluation.trajectory_scoring import mean_margin_badness
from room_for_doubt.evaluation.selection import (
    weighted_consensus,
    paired_accuracy_interval,
)
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory


def consensus(probs, badness, alpha, temperature, mixture):
    uniform = probs.mean(1)
    if mixture == 0:
        return uniform
    score = alpha * badness + (1 - alpha) * (1 - probs.max(2))
    return (1 - mixture) * uniform + mixture * weighted_consensus(
        probs, score, temperature
    )


@hydra.main(
    config_path=str(config_directory()), config_name="hybrid_tta", version_base=None
)
def main(cfg):
    torch.set_num_threads(4)
    out = Path(HydraConfig.get().runtime.output_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    cache = Path(to_absolute_path(cfg.cache_dir))
    study = Path(to_absolute_path(cfg.study_dir))
    probs = np.load(cache / "val_probs.npy", mmap_mode="r")
    features = np.load(cache / "val_features.npy", mmap_mode="r")
    ids = np.load(cache / "val_ids.npy")
    noisy = load_observed_labels(
        to_absolute_path(cfg.data_dir), cfg.dataset.name, cfg.dataset.noisy_split
    )[ids]
    selected, sweeps, sources = [], [], []
    for seed in cfg.seeds:
        run = Path(to_absolute_path(cfg.head_template.format(seed=seed)))
        saved = torch.load(
            run / "checkpoints/best.pt", map_location="cpu", weights_only=False
        )
        assert saved["config"]["seed"] == seed
        assert saved["config"]["dataset"]["name"] == cfg.dataset.name
        model = ConditionedPredictor(probs.shape[2], pretrained=False)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        paths = predict_arrays(
            model,
            features.reshape(-1, 512),
            probs.argmax(2).reshape(-1),
            cfg,
            torch.device("cpu"),
        )
        badness = mean_margin_badness(paths).reshape(probs.shape[:2])
        prior = json.loads(
            (
                study
                / f"artifacts/selection_cifar100n_conditioned_mean_margin_{seed}.json"
            ).read_text()
        )["tta_temperature"]
        # Prior policy first: exact validation ties retain it.
        candidates = [
            dict(alpha=1.0, temperature=float(prior), mixture=1.0),
            dict(alpha=1.0, temperature=1.0, mixture=0.0),
        ]
        candidates += [
            dict(alpha=float(a), temperature=float(t), mixture=float(m))
            for a in cfg.alphas
            for t in cfg.temperatures
            for m in cfg.mixtures
        ]
        rows = [
            dict(
                seed=seed,
                **c,
                val_noisy_acc=float(
                    np.mean(consensus(probs, badness, **c).argmax(1) == noisy)
                ),
            )
            for c in candidates
        ]
        best = max(rows, key=lambda r: r["val_noisy_acc"])
        selected.append(best)
        sweeps.extend(rows)
        sources.append(str(run / "checkpoints/best.pt"))
        np.savez_compressed(
            out / f"artifacts/validation_{seed}.npz",
            ids=ids,
            noisy_labels=noisy,
            badness=badness,
        )
        print("validation", best, flush=True)
    # Persist all selections before reading test labels or correctness.
    (out / "artifacts/selections.json").write_text(json.dumps(selected, indent=2))
    test_probs = np.load(cache / "test_probs.npy", mmap_mode="r")
    labels = np.load(cache / "test_clean_labels.npy")
    rows = []
    for selection in selected:
        seed = selection["seed"]
        with np.load(
            study / f"artifacts/test_cifar100n_conditioned_mean_margin_{seed}.npz"
        ) as previous:
            assert np.array_equal(labels, previous["labels"])
            badness = previous["tta_badness"].copy()
            prior_pred = previous["weighted_probabilities"].argmax(1)
        params = {k: selection[k] for k in ["alpha", "temperature", "mixture"]}
        pred = consensus(test_probs, badness, **params).argmax(1)
        interval = paired_accuracy_interval(labels, prior_pred, pred)
        row = dict(
            seed=seed,
            accuracy=float(np.mean(pred == labels)),
            prior_accuracy=float(np.mean(prior_pred == labels)),
            **interval,
        )
        rows.append(row)
        np.savez_compressed(
            out / f"artifacts/test_{seed}.npz",
            labels=labels,
            predictions=pred,
            prior_predictions=prior_pred,
        )
    base_top2 = np.argsort(-test_probs[:, 0], axis=1, kind="stable")[:, :2]
    # Oracle diagnostic only: no deployable prediction or selection uses true labels.
    oracle_top2 = float(np.mean((base_top2 == labels[:, None]).any(1)))
    metrics = dict(
        results=rows,
        mean_accuracy=float(np.mean([r["accuracy"] for r in rows])),
        mean_gain_over_prior_pp=float(np.mean([r["gain_pp"] for r in rows])),
        oracle_base_top2_ceiling=oracle_top2,
        target_accuracy=0.70,
        target_reached=bool(np.mean([r["accuracy"] for r in rows]) >= 0.70),
        selections=selected,
    )
    config = OmegaConf.to_container(cfg, resolve=True)
    meta = dict(
        checkpoint_path=sources,
        device="CPU; cached views",
        seed=list(cfg.seeds),
        base_checkpoint_assumption="owner confirmed",
    )
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "Exploratory bounded follow-up on previously inspected tests. Twenty candidates per meta seed including duplicate controls; all selections frozen on noisy validation before test evaluation. Three meta seeds share one base model. Oracle base Top-2 ceiling uses clean test labels only for an explicitly oracle diagnostic; it bounds original-view Top-2 correction only, not arbitrary classifiers or TTA. Validation arrays and full sweeps saved; cached val/test probabilities remain at cache_dir. No training or new encoder inference.",
    )
    for name, data in [("summary", rows), ("validation_sweep", sweeps)]:
        with (out / f"tables/{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
