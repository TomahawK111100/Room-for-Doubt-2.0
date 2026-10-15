"""Shared architecture/scoring experiment; no class-count based method switch."""

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
from room_for_doubt.data.cifar_noisy import load_observed_labels, load_noisy_labels
from room_for_doubt.data.meta_splits import validate_meta_splits
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor
from room_for_doubt.models.flow_matching import CFMTrajectoryModel
from room_for_doubt.training.meta_retraining import choose_device
from room_for_doubt.evaluation.corrected_campaign import predict_arrays
from room_for_doubt.evaluation.trajectory_scoring import (
    mean_margin_badness,
    quantile_thresholds,
)
from room_for_doubt.evaluation.selection import (
    select_threshold,
    top2_correct,
    error_metrics,
    paired_accuracy_interval,
    select_temperature,
    weighted_consensus,
)
from room_for_doubt.reporting.artifacts import (
    generate_report,
    save_metrics,
    save_run_metadata,
)
from room_for_doubt.utils.paths import config_directory
from room_for_doubt.reporting.architecture_study import write_study_summary


MODEL_CLASSES = {"conditioned": ConditionedPredictor, "cfm": CFMTrajectoryModel}


def score_pair(paths, scorer):
    if scorer is None:
        return mean_margin_badness(paths)
    return scorer.predict_proba(paths)[:, 1]


def cached_tta_paths(model, cache, split, original_paths, cfg, device):
    probs = np.load(cache / f"{split}_probs.npy", mmap_mode="r")
    if not cfg.evaluate_cached_tta or probs.shape[1] == 1:
        return probs[:, :1], original_paths[:, None, :]
    features = np.load(cache / f"{split}_features.npy", mmap_mode="r")
    paths = predict_arrays(
        model, features.reshape(-1, 512), probs.argmax(2).reshape(-1), cfg, device
    ).reshape(len(probs), probs.shape[1], 50)
    # Use the exact same original-view Monte Carlo estimate in both actions.
    paths[:, 0] = original_paths
    return probs, paths


def load_model(run, architecture, seed, dataset, data, backbone, device):
    saved = torch.load(
        run / "checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    config = saved["config"]
    if (
        config["meta_model"] != architecture
        or config["seed"] != seed
        or config["dataset"]["name"] != dataset
    ):
        raise ValueError("Meta checkpoint architecture/seed/dataset mismatch")
    if (
        architecture == "cfm"
        and config.get("validation_metric", "native") != "trajectory_mse"
    ):
        raise ValueError("CFM study checkpoints must use reconstruction-MSE selection")
    with np.load(run / "artifacts/split_ids.npz", allow_pickle=False) as splits:
        for key in [
            "meta_train_ids",
            "meta_val_ids",
            "detector_fit_ids",
            "detector_eval_ids",
        ]:
            if not np.array_equal(splits[key], data[key]):
                raise ValueError("Meta/evaluation split mismatch")
    model = MODEL_CLASSES[architecture](
        config["dataset"]["num_classes"], pretrained=False
    )
    model.load_state_dict(saved["state_dict"])
    if any(
        not torch.equal(v, model.features.state_dict()[k]) for k, v in backbone.items()
    ):
        raise ValueError("Meta checkpoint frozen backbone differs from cache")
    return model.to(device).eval()


@hydra.main(
    config_path=str(config_directory()),
    config_name="architecture_study",
    version_base=None,
)
def main(cfg):
    torch.set_num_threads(4)
    device = choose_device(cfg.device)
    out = Path(HydraConfig.get().runtime.output_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    pending = []
    fidelity = []
    selection_rows = []
    source_paths = []
    # Freeze every dataset/model/score/seed selection before test correctness.
    for entry in cfg.datasets:
        with np.load(to_absolute_path(entry.targets), allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        validate_meta_splits(data, data["sample_ids"])
        if (
            data["dataset_name"].item() != entry.name
            or int(data["split_seed"]) != cfg.split_seed
        ):
            raise ValueError("Prepared dataset mismatch")
        noisy = load_observed_labels(
            to_absolute_path(cfg.data_dir), entry.name, entry.noisy_split
        )
        if not np.array_equal(data["noisy_labels"], noisy[data["sample_ids"]]):
            raise ValueError("Noisy annotation mismatch")
        feature_dir = Path(to_absolute_path(entry.features))
        cache = Path(to_absolute_path(entry.inference_cache))
        with np.load(feature_dir / "features.npz", allow_pickle=False) as f:
            if not np.array_equal(f["sample_ids"], data["sample_ids"]):
                raise ValueError("Feature ID mismatch")
            features = f["features"]
        if not (cache / "complete.json").exists():
            raise ValueError("Use a complete shared inference cache")
        if not np.array_equal(np.load(cache / "val_ids.npy"), data["base_val_ids"]):
            raise ValueError("Validation IDs differ")
        with (cache / "complete.json").open() as f:
            cache_meta = json.load(f)
        if (
            cache_meta["dataset"] != entry.name
            or cache_meta["inference_seed"] != cfg.inference_seed
            or cache_meta["meta_preprocessing"] != "imagenet_pil_bilinear_224"
        ):
            raise ValueError("Inference cache protocol mismatch")
        backbone = torch.load(
            feature_dir / "backbone.pt", map_location="cpu", weights_only=True
        )
        lookup = {int(s): i for i, s in enumerate(data["sample_ids"])}
        val_rows = [lookup[int(s)] for s in data["meta_val_ids"]]
        scorers = {"mean_margin": None}
        if cfg.run_oracle_reference:
            _, clean = load_noisy_labels(
                to_absolute_path(cfg.data_dir), entry.name, entry.noisy_split
            )
            ids = data["detector_fit_ids"]
            fit_rows = [lookup[int(s)] for s in ids]
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(data["targets"][fit_rows], (noisy[ids] != clean[ids]).astype(int))
            scorers["oracle_lda"] = lda
        val_probs = np.load(cache / "val_probs.npy", mmap_mode="r")[:, 0]
        val_features = np.load(cache / "val_features.npy", mmap_mode="r")[:, 0]
        top = np.argsort(-val_probs, axis=1, kind="stable")[:, :2].copy()
        for architecture, pattern in entry.models.items():
            for seed in cfg.seeds:
                run = Path(to_absolute_path(pattern.format(seed=seed)))
                source_paths.append(str(run))
                model = load_model(
                    run, architecture, seed, entry.name, data, backbone, device
                )
                start = time.perf_counter()
                reconstructed = predict_arrays(
                    model,
                    features[val_rows],
                    data["noisy_labels"][val_rows],
                    cfg,
                    device,
                )
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                mse = float(np.mean((reconstructed - data["targets"][val_rows]) ** 2))
                fidelity.append(
                    dict(
                        dataset=entry.name,
                        architecture=architecture,
                        seed=seed,
                        val_trajectory_mse=mse,
                        seconds=elapsed,
                        samples=len(val_rows),
                        trainable_parameters=sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                    )
                )
                paths = [
                    predict_arrays(model, val_features, top[:, k], cfg, device)
                    for k in range(2)
                ]
                val_tta_probs, val_tta_paths = cached_tta_paths(
                    model, cache, "val", paths[0], cfg, device
                )
                for score_name, scorer in scorers.items():
                    badness = np.column_stack([score_pair(p, scorer) for p in paths])
                    thresholds = quantile_thresholds(badness, cfg.threshold_quantiles)
                    selected, sweep = select_threshold(
                        top, badness, noisy[data["base_val_ids"]], thresholds
                    )
                    val_tta_scores = score_pair(
                        val_tta_paths.reshape(-1, 50), scorer
                    ).reshape(val_tta_paths.shape[:2])
                    selected_t, sweep_t = select_temperature(
                        val_tta_probs,
                        val_tta_scores,
                        noisy[data["base_val_ids"]],
                        cfg.tta_temperatures,
                    )
                    key = f"{entry.name}_{architecture}_{score_name}_{seed}"
                    np.savez_compressed(
                        out / f"artifacts/validation_{key}.npz",
                        ids=data["base_val_ids"],
                        noisy_labels=noisy[data["base_val_ids"]],
                        top=top,
                        badness=badness,
                        tta_badness=val_tta_scores,
                        tta_probabilities=val_tta_probs,
                    )
                    (out / f"artifacts/selection_{key}.json").write_text(
                        json.dumps(
                            dict(
                                selected=selected,
                                sweep=sweep,
                                label_source="noisy validation",
                                tta_temperature=selected_t,
                                tta_sweep=sweep_t,
                            ),
                            indent=2,
                        )
                    )
                    selection_rows.append(
                        dict(
                            dataset=entry.name,
                            architecture=architecture,
                            score=score_name,
                            seed=seed,
                            threshold=selected,
                            val_noisy_acc=max(r["val_noisy_acc"] for r in sweep),
                            tta_temperature=selected_t,
                            tta_val_noisy_acc=max(r["val_noisy_acc"] for r in sweep_t),
                        )
                    )
                    pending.append(
                        (
                            entry.name,
                            architecture,
                            seed,
                            run,
                            cache,
                            score_name,
                            scorer,
                            selected,
                            selected_t,
                            data,
                            backbone,
                        )
                    )
                del model
                print(
                    f"{entry.name} {architecture} seed{seed}: held-out trajectory MSE={mse:.5g}; selections frozen",
                    flush=True,
                )
    (out / "artifacts/all_selections_frozen.json").write_text(
        json.dumps(selection_rows, indent=2)
    )
    # Report reconstruction preference using validation only, never test gains.
    preference = {
        arch: float(
            np.mean(
                [r["val_trajectory_mse"] for r in fidelity if r["architecture"] == arch]
            )
        )
        for arch in MODEL_CLASSES
    }
    validation_preference = min(preference, key=preference.get)
    rows = []
    previous = None
    model = None
    paths = None
    for (
        dataset,
        architecture,
        seed,
        run,
        cache,
        score_name,
        scorer,
        selected,
        selected_t,
        data,
        backbone,
    ) in pending:
        key = (dataset, architecture, seed)
        if key != previous:
            model = load_model(run, architecture, seed, dataset, data, backbone, device)
            probs = np.load(cache / "test_probs.npy", mmap_mode="r")[:, 0]
            f = np.load(cache / "test_features.npy", mmap_mode="r")[:, 0]
            labels = np.load(cache / "test_clean_labels.npy")
            top = np.argsort(-probs, axis=1, kind="stable")[:, :2].copy()
            paths = [predict_arrays(model, f, top[:, k], cfg, device) for k in range(2)]
            tta_probs, tta_paths = cached_tta_paths(
                model, cache, "test", paths[0], cfg, device
            )
            previous = key
        badness = np.column_stack([score_pair(p, scorer) for p in paths])
        corrected, mask = top2_correct(top, badness, selected)
        metrics = error_metrics(labels, probs, badness[:, 0])
        interval = paired_accuracy_interval(labels, top[:, 0], corrected)
        tta_badness = score_pair(tta_paths.reshape(-1, 50), scorer).reshape(
            tta_paths.shape[:2]
        )
        standard = tta_probs.mean(1)
        weighted = (
            standard
            if selected_t == "uniform"
            else weighted_consensus(tta_probs, tta_badness, selected_t)
        )
        tta_interval = paired_accuracy_interval(
            labels, standard.argmax(1), weighted.argmax(1)
        )
        row = dict(
            dataset=dataset,
            architecture=architecture,
            score=score_name,
            seed=seed,
            base_acc=metrics["acc"],
            corrected_acc=float(np.mean(corrected == labels)),
            auroc=metrics["auroc"],
            auprc=metrics["auprc"],
            aurc=metrics["aurc"],
            overrides=int(mask.sum()),
            views=tta_probs.shape[1],
            tta_acc=float(np.mean(standard.argmax(1) == labels)),
            uw_tta_acc=float(np.mean(weighted.argmax(1) == labels)),
            uw_minus_tta_pp=tta_interval["gain_pp"],
            uw_ci_low_pp=tta_interval["ci_low_pp"],
            uw_ci_high_pp=tta_interval["ci_high_pp"],
            **interval,
        )
        rows.append(row)
        np.savez_compressed(
            out / f"artifacts/test_{dataset}_{architecture}_{score_name}_{seed}.npz",
            labels=labels,
            probabilities=probs,
            badness=badness,
            corrected=corrected,
            standard_probabilities=standard,
            weighted_probabilities=weighted,
            tta_badness=tta_badness,
        )
        print(row, flush=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    metrics = dict(
        common_validation_mse_by_architecture=preference,
        validation_reconstruction_preference=validation_preference,
        fidelity=fidelity,
        selections=selection_rows,
        results=rows,
    )
    save_metrics(out, metrics)
    meta = dict(
        device=str(device),
        seed=list(cfg.seeds),
        checkpoint_path=source_paths,
        base_checkpoint_assumption="owner confirmed",
        primary_supervision="no clean training labels used by mean-margin scorer or meta training",
        oracle_reference=bool(cfg.run_oracle_reference),
        class_count_routing=False,
    )
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "Exploratory follow-up after historical test inspection. Same Top-2 policy, quantile calibration and scores for both datasets. Mean-margin score is a fixed heuristic without clean-label detector training; oracle LDA is a separately labeled reference. Clean test labels used for evaluation only. Real trajectory MSE on meta-validation compares conditional means, not flow velocity loss. Reconstruction preference is validation-only, not a claim of best downstream accuracy. Architectures have unequal capacities; parameter count and cost are recorded. Cached original views used; no new image inference. CFM uses five draws/ten steps; oracle LDA scores the mean trajectory for both models, rather than averaging per-draw LDA probabilities. One base seed per dataset, complete-case targets, three meta seeds. No across-dataset weight transfer or universal effectiveness established.",
    )
    for name, table in [
        ("summary", rows),
        ("fidelity", fidelity),
        ("selection", selection_rows),
    ]:
        with (out / f"tables/{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_study_summary(out, config, metrics)


if __name__ == "__main__":
    main()
