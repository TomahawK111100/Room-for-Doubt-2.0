"""Fixed held-out ablation, noisy-val selection and paired test evaluation."""

from pathlib import Path
import csv
import json
import time
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor
from room_for_doubt.models.flow_matching import CFMTrajectoryModel
from room_for_doubt.inference.flow_sampling import sample_flow_trajectories
from room_for_doubt.data.cifar_noisy import load_noisy_labels
from room_for_doubt.data.meta_splits import validate_meta_splits
from room_for_doubt.analysis.matched_lda import evaluate_matched_lda
from room_for_doubt.training.meta_retraining import choose_device
from room_for_doubt.evaluation.campaign_cache import build_cache
from room_for_doubt.evaluation.selection import (
    top2_correct,
    select_threshold,
    select_temperature,
    weighted_consensus,
    error_metrics,
)
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory


@torch.no_grad()
def predicted_trajectories(model, features, candidates, cfg, generator):
    if isinstance(model, ConditionedPredictor):
        return model.predict_from_features(features, candidates)[:, None, :]
    return sample_flow_trajectories(
        model, features, candidates, cfg.flow_samples, cfg.flow_steps, generator
    )


@torch.no_grad()
def predict_arrays(model, features, candidates, cfg, device, detector=None):
    rng = torch.Generator().manual_seed(cfg.inference_seed)
    result = []
    for start in range(0, len(features), 1024):
        f = torch.as_tensor(np.array(features[start : start + 1024]), device=device)
        y = torch.as_tensor(
            candidates[start : start + 1024], device=device, dtype=torch.long
        )
        paths = predicted_trajectories(model, f, y, cfg, rng).cpu().numpy()
        if detector is None:
            result.append(paths.mean(axis=1))
        else:
            result.append(
                detector.predict_proba(paths.reshape(-1, 50))[:, 1]
                .reshape(len(f), -1)
                .mean(axis=1)
            )
    return np.concatenate(result)


def fit_oracle_detector(cfg, data, clean, noisy):
    rows = {int(s): i for i, s in enumerate(data["sample_ids"])}
    fit = np.array([rows[int(s)] for s in data["detector_fit_ids"]])
    labels = (noisy[data["sample_ids"]] != clean[data["sample_ids"]]).astype(int)
    detector = (
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        if cfg.dataset.name == "cifar10n"
        else RandomForestClassifier(
            n_estimators=cfg.rf_trees,
            min_samples_leaf=cfg.rf_min_leaf,
            random_state=cfg.base_seed,
            n_jobs=4,
        )
    )
    detector.fit(data["targets"][fit], labels[fit])
    return detector, labels


@hydra.main(
    config_path=str(config_directory()),
    config_name="final_evaluation",
    version_base=None,
)
def main(cfg):
    if (
        cfg.checkpoint_assumption != "owner_confirmed"
        or cfg.supervision != "oracle_clean_train_labels"
    ):
        raise ValueError(
            "Declare owner checkpoint assumption and oracle detector supervision"
        )
    torch.set_num_threads(4)
    device = choose_device(cfg.device)
    out = Path(HydraConfig.get().runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    with np.load(to_absolute_path(cfg.target_path), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    validate_meta_splits(data, data["sample_ids"])
    if (
        data["dataset_name"].item() != cfg.dataset.name
        or int(data["split_seed"]) != cfg.split_seed
    ):
        raise ValueError("Wrong prepared input dataset/split")
    noisy, clean = load_noisy_labels(
        to_absolute_path(cfg.dataset.data_dir),
        cfg.dataset.name,
        cfg.dataset.noisy_split,
    )
    if not np.array_equal(data["noisy_labels"], noisy[data["sample_ids"]]):
        raise ValueError("Noisy target mismatch")
    detector, noise_labels = fit_oracle_detector(cfg, data, clean, noisy)
    with np.load(
        Path(to_absolute_path(cfg.feature_path)) / "features.npz", allow_pickle=False
    ) as f:
        if not np.array_equal(f["sample_ids"], data["sample_ids"]):
            raise ValueError("Feature IDs differ")
        features = f["features"]
    cache = build_cache(cfg, data, device)
    results = {}
    compact = []
    start = time.perf_counter()
    rows = {int(s): i for i, s in enumerate(data["sample_ids"])}
    paired_ids = np.concatenate([data["detector_fit_ids"], data["detector_eval_ids"]])
    paired_rows = np.array([rows[int(s)] for s in paired_ids])
    pending = {}
    for seed in cfg.meta_seeds:
        run = Path(to_absolute_path(cfg.meta_run_pattern.format(seed=seed)))
        saved = torch.load(
            run / "checkpoints/best.pt", map_location="cpu", weights_only=False
        )
        if (
            saved["config"]["seed"] != seed
            or saved["config"]["dataset"]["name"] != cfg.dataset.name
        ):
            raise ValueError("Meta checkpoint seed/dataset mismatch")
        with np.load(run / "artifacts/split_ids.npz") as f:
            for k in [
                "meta_train_ids",
                "meta_val_ids",
                "detector_fit_ids",
                "detector_eval_ids",
            ]:
                if not np.array_equal(f[k], data[k]):
                    raise ValueError("Training split differs from evaluation inputs")
        model = (
            ConditionedPredictor
            if cfg.dataset.name == "cifar10n"
            else CFMTrajectoryModel
        )(cfg.dataset.num_classes, pretrained=False)
        model.load_state_dict(saved["state_dict"])
        model.to(device).eval()
        original = torch.load(
            Path(to_absolute_path(cfg.feature_path)) / "backbone.pt",
            map_location="cpu",
            weights_only=True,
        )
        if any(
            not torch.equal(v, model.features.state_dict()[k].cpu())
            for k, v in original.items()
        ):
            raise ValueError("Meta checkpoint backbone differs from feature cache")
        labels = data["noisy_labels"][paired_rows]
        dynamic = predict_arrays(model, features[paired_rows], labels, cfg, device)
        emb = (
            model.class_emb
            if cfg.dataset.name == "cifar10n"
            else model.vector_field.class_emb
        )
        with torch.no_grad():
            class_features = emb(torch.as_tensor(labels, device=device)).cpu().numpy()
        static = np.concatenate([features[paired_rows], class_features], axis=1)
        ablation = evaluate_matched_lda(
            static,
            dynamic,
            noise_labels[paired_rows],
            paired_ids,
            data["detector_fit_ids"],
            data["detector_eval_ids"],
            np.concatenate([data["meta_train_ids"], data["meta_val_ids"]]),
        )
        np.savez_compressed(
            out / f"artifacts/paired_seed{seed}.npz",
            static_features=static,
            predicted_trajectories=dynamic,
            labels=noise_labels[paired_rows],
            sample_ids=paired_ids,
            fit_ids=data["detector_fit_ids"],
            eval_ids=data["detector_eval_ids"],
            meta_train_ids=np.concatenate(
                [data["meta_train_ids"], data["meta_val_ids"]]
            ),
            label_kind=np.array("noise_label_detection"),
            meta_checkpoint_path=np.array(str(run / "checkpoints/best.pt")),
            representation_kind=np.array("shared_features_predicted_trajectories"),
        )
        for method, values in ablation.items():
            compact.append(
                dict(seed=seed, split="heldout_noise", method=method, **values)
            )
        val_probs = np.load(cache / "val_probs.npy", mmap_mode="r")
        val_feats = np.load(cache / "val_features.npy", mmap_mode="r")
        test_probs = np.load(cache / "test_probs.npy", mmap_mode="r")
        test_feats = np.load(cache / "test_features.npy", mmap_mode="r")
        if cfg.dataset.name == "cifar10n":
            val_top = np.argsort(val_probs[:, 0], axis=1)[:, -2:][:, ::-1].copy()
            val_b = np.column_stack(
                [
                    predict_arrays(
                        model, val_feats[:, 0], val_top[:, k], cfg, device, detector
                    )
                    for k in range(2)
                ]
            )
            selected, sweep = select_threshold(
                val_top, val_b, noisy[data["base_val_ids"]], cfg.thresholds
            )
        else:
            val_b = predict_arrays(
                model,
                val_feats.reshape(-1, 512),
                val_probs.argmax(2).reshape(-1),
                cfg,
                device,
                detector,
            ).reshape(val_probs.shape[:2])
            selected, sweep = select_temperature(
                val_probs, val_b, noisy[data["base_val_ids"]], cfg.temperatures
            )
        # Selection is durably recorded before any test correctness is inspected.
        (out / f"artifacts/selection_seed{seed}.json").write_text(
            json.dumps(
                dict(
                    selected=selected,
                    validation_label_source="official noisy labels",
                    sweep=sweep,
                ),
                indent=2,
            )
        )
        pending[seed] = (model, ablation, selected)
        print(f"seed {seed}: validation selection frozen: {selected}", flush=True)
    # Freeze the entire campaign's selection before observing any test correctness.
    for seed, (model, ablation, selected) in pending.items():
        test_y = np.load(cache / "test_clean_labels.npy")
        baseline = error_metrics(test_y, test_probs[:, 0])
        per_seed = dict(ablation=ablation, baseline=baseline, selected=selected)
        compact.append(dict(seed=seed, split="test_error", method="msp", **baseline))
        if cfg.dataset.name == "cifar10n":
            top = np.argsort(test_probs[:, 0], axis=1)[:, -2:][:, ::-1].copy()
            b = np.column_stack(
                [
                    predict_arrays(
                        model, test_feats[:, 0], top[:, k], cfg, device, detector
                    )
                    for k in range(2)
                ]
            )
            corrected, mask = top2_correct(top, b, selected)
            correction = dict(
                acc=float(np.mean(corrected == test_y)),
                overrides=int(mask.sum()),
                successful=int(
                    (mask & (top[:, 0] != test_y) & (top[:, 1] == test_y)).sum()
                ),
                harmful=int((mask & (top[:, 0] == test_y)).sum()),
            )
            per_seed["top2"] = correction
            compact.append(
                dict(seed=seed, split="test_accuracy", method="top2", **correction)
            )
            score = b[:, 0]
            np.savez_compressed(
                out / f"artifacts/test_seed{seed}.npz",
                labels=test_y,
                probabilities=test_probs[:, 0],
                badness=b,
                corrected=corrected,
            )
        else:
            b = predict_arrays(
                model,
                test_feats.reshape(-1, 512),
                test_probs.argmax(2).reshape(-1),
                cfg,
                device,
                detector,
            ).reshape(test_probs.shape[:2])
            standard = test_probs.mean(1)
            weighted = (
                standard
                if selected == "uniform"
                else weighted_consensus(test_probs, b, selected)
            )
            for name, p in [("tta", standard), ("uw_tta", weighted)]:
                per_seed[name] = error_metrics(test_y, p)
                compact.append(
                    dict(seed=seed, split="test_error", method=name, **per_seed[name])
                )
            score = b[:, 0]
            np.savez_compressed(
                out / f"artifacts/test_seed{seed}.npz",
                labels=test_y,
                baseline_probabilities=test_probs[:, 0],
                standard_probabilities=standard,
                weighted_probabilities=weighted,
                badness=b,
            )
        for name, s in [
            ("badness", score),
            (
                "entropy",
                -(test_probs[:, 0] * np.log(np.maximum(test_probs[:, 0], 1e-12))).sum(
                    1
                ),
            ),
            (
                "margin",
                -(
                    np.sort(test_probs[:, 0], axis=1)[:, -1]
                    - np.sort(test_probs[:, 0], axis=1)[:, -2]
                ),
            ),
        ]:
            values = error_metrics(test_y, test_probs[:, 0], s)
            per_seed[name] = values
            compact.append(dict(seed=seed, split="test_error", method=name, **values))
        results[str(seed)] = per_seed
        print(
            f"seed {seed}: ablation={ablation}; selected={selected}; test={per_seed}",
            flush=True,
        )
    metrics = dict(
        seeds=results, elapsed_evaluation_seconds=time.perf_counter() - start
    )
    save_metrics(out, metrics)
    meta = dict(
        seed=list(cfg.meta_seeds),
        device=str(device),
        checkpoint_path=to_absolute_path(cfg.base_checkpoint),
        checkpoint_assumption="owner confirmed correctness; not independently established",
        detector_supervision=cfg.supervision,
        selection_labels="noisy base validation",
        base_seed=cfg.base_seed,
        split_seed=cfg.split_seed,
        meta_checkpoints=[
            to_absolute_path(cfg.meta_run_pattern.format(seed=s))
            + "/checkpoints/best.pt"
            for s in cfg.meta_seeds
        ],
    )
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "Oracle clean TRAIN detector supervision. Matched LDA predicts annotation noise, while test badness AUROC detects classifier errors: distinct targets. Static arm shares a temporally trained class embedding. CFM ablation uses mean of 5 Euler samples; primary badness averages detector probabilities over samples. All methods share exact cached views; original view plus 19 crop/flip views for CIFAR-100. Base normalization fixed to historical CIFAR-10 statistics for both datasets before evaluation. ECE describes the classifier probabilities, not calibration of badness. Three meta initialization seeds share one assumed-correct historical base checkpoint, identical splits and identical random flow draws; no independent base-seed claim. Validation selection precedes test correctness. Historical test inspections make this a protocol-fixed replication, not a newly untouched confirmatory test.",
    )
    fields = [
        "seed",
        "split",
        "method",
        "acc",
        "auroc",
        "auprc",
        "aurc",
        "ece",
        "fit_n",
        "eval_n",
        "overrides",
        "successful",
        "harmful",
    ]
    with (out / "tables/summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in compact:
            w.writerow(
                {
                    k: format(v, ".4g") if isinstance(v, float) else v
                    for k, v in row.items()
                }
            )


if __name__ == "__main__":
    main()
