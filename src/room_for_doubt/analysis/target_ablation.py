"""Last versus mean versus full-trajectory regression on identical frozen inputs."""

import csv
import json
from pathlib import Path
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from room_for_doubt.models.scalar_margin import ScalarMarginPredictor
from room_for_doubt.analysis.target_contrasts import target_contrasts
from room_for_doubt.training.meta_retraining import (
    read_targets,
    fit_head,
    choose_device,
)
from room_for_doubt.analysis.architecture_study import load_model, cached_tta_paths
from room_for_doubt.evaluation.corrected_campaign import predict_arrays
from room_for_doubt.evaluation.trajectory_scoring import (
    mean_margin_badness,
    quantile_thresholds,
)
from room_for_doubt.evaluation.selection import (
    select_threshold,
    select_temperature,
    top2_correct,
    weighted_consensus,
    error_metrics,
    paired_accuracy_interval,
)
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.utils.reproducibility import set_seed
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory


def regression_targets(targets, mode):
    if mode == "last":
        scalar = targets[:, -1:]
    elif mode == "mean":
        scalar = targets.mean(1, keepdims=True)
    else:
        raise ValueError("Scalar mode must be last or mean")
    return np.repeat(scalar, targets.shape[1], axis=1)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@hydra.main(
    config_path=str(config_directory()),
    config_name="target_ablation",
    version_base=None,
)
def main(cfg):
    torch.set_num_threads(4)
    device = choose_device(cfg.device)
    out = Path(HydraConfig.get().runtime.output_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    data = read_targets(cfg)
    cache_dir = Path(to_absolute_path(cfg.cache_path))
    with np.load(cache_dir / "features.npz") as f:
        if (
            not np.array_equal(f["sample_ids"], data["sample_ids"])
            or f["dataset_name"].item() != cfg.dataset.name
        ):
            raise ValueError("Cached feature identity mismatch")
        if f["preprocessing"].item() != "imagenet_pil_bilinear_224":
            raise ValueError("Unexpected preprocessing")
        features = f["features"].copy()
    backbone = torch.load(
        cache_dir / "backbone.pt", map_location="cpu", weights_only=True
    )
    sources = []
    # Complete all training before downstream validation selection and test evaluation.
    for mode in ["last", "mean"]:
        transformed = dict(data, targets=regression_targets(data["targets"], mode))
        for seed in cfg.seeds:
            local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            local.seed = seed
            local.target_mode = mode
            local.run_id = f"{cfg.run_id}_{mode}_seed{seed}"
            run = out / "heads" / f"{mode}_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(local, run / "config.yaml", resolve=True)
            set_seed(seed)
            model = ScalarMarginPredictor(cfg.dataset.num_classes)
            model.features.load_state_dict(backbone)
            m = fit_head(model, transformed, features, local, run, device)
            meta = dict(
                seed=seed,
                device=str(device),
                checkpoint_path=str(run / "checkpoints/best.pt"),
                target_mode=mode,
                trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            )
            config = OmegaConf.to_container(local, resolve=True)
            save_metrics(run, m)
            save_run_metadata(run, local.run_id, config, **meta)
            generate_report(
                run,
                local.run_id,
                config,
                m,
                meta,
                "Scalar noisy-class regression. Same frozen features, class embedding and hidden width as full trajectory; one output instead of fifty. Repeated scalar targets/outputs make common MSE equal scalar MSE. No clean-label training or test selection.",
            )
            sources.append(meta["checkpoint_path"])

    def model_for(mode, seed):
        if mode == "full":
            run = Path(to_absolute_path(cfg.full_template.format(seed=seed)))
            return load_model(
                run, "conditioned", seed, cfg.dataset.name, data, backbone, device
            )
        saved = torch.load(
            out / "heads" / f"{mode}_{seed}" / "checkpoints/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = ScalarMarginPredictor(cfg.dataset.num_classes)
        model.load_state_dict(saved["state_dict"])
        return model.to(device).eval()

    cache = Path(to_absolute_path(cfg.inference_cache))
    val_probs = np.load(cache / "val_probs.npy", mmap_mode="r")
    val_features = np.load(cache / "val_features.npy", mmap_mode="r")
    val_ids = np.load(cache / "val_ids.npy")
    if not np.array_equal(val_ids, data["base_val_ids"]):
        raise ValueError("Base validation IDs differ")
    noisy = load_observed_labels(
        to_absolute_path(cfg.dataset.data_dir),
        cfg.dataset.name,
        cfg.dataset.noisy_split,
    )[val_ids]
    top = np.argsort(-val_probs[:, 0], axis=1, kind="stable")[:, :2]
    selections = []
    for mode in ["last", "mean", "full"]:
        for seed in cfg.seeds:
            model = model_for(mode, seed)
            paths = [
                predict_arrays(model, val_features[:, 0], top[:, k], cfg, device)
                for k in range(2)
            ]
            badness = np.column_stack([mean_margin_badness(p) for p in paths])
            tta_probs, tta_paths = cached_tta_paths(
                model, cache, "val", paths[0], cfg, device
            )
            tta_badness = mean_margin_badness(tta_paths.reshape(-1, 50)).reshape(
                tta_paths.shape[:2]
            )
            threshold, sweep = select_threshold(
                top,
                badness,
                noisy,
                quantile_thresholds(badness, cfg.threshold_quantiles),
            )
            temperature, tta_sweep = select_temperature(
                tta_probs, tta_badness, noisy, cfg.tta_temperatures
            )
            record = dict(
                mode=mode,
                seed=seed,
                threshold=threshold,
                temperature=temperature,
                trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                top2_sweep=sweep,
                tta_sweep=tta_sweep,
            )
            selections.append(record)
            np.savez_compressed(
                out / f"artifacts/validation_{mode}_{seed}.npz",
                ids=val_ids,
                noisy_labels=noisy,
                top=top,
                badness=badness,
                tta_badness=tta_badness,
            )
            print("selected", mode, seed, threshold, temperature, flush=True)
    (out / "artifacts/all_selections_before_test.json").write_text(
        json.dumps(selections, indent=2)
    )
    # First clean-test load occurs only after all nine policies have been frozen.
    test_probs = np.load(cache / "test_probs.npy", mmap_mode="r")
    test_features = np.load(cache / "test_features.npy", mmap_mode="r")
    labels = np.load(cache / "test_clean_labels.npy")
    top = np.argsort(-test_probs[:, 0], axis=1, kind="stable")[:, :2]
    rows = []
    for record in selections:
        mode, seed = record["mode"], record["seed"]
        model = model_for(mode, seed)
        paths = [
            predict_arrays(model, test_features[:, 0], top[:, k], cfg, device)
            for k in range(2)
        ]
        badness = np.column_stack([mean_margin_badness(p) for p in paths])
        tta_probs, tta_paths = cached_tta_paths(
            model, cache, "test", paths[0], cfg, device
        )
        tta_badness = mean_margin_badness(tta_paths.reshape(-1, 50)).reshape(
            tta_paths.shape[:2]
        )
        standard = tta_probs.mean(1)
        weighted = (
            standard
            if record["temperature"] == "uniform"
            else weighted_consensus(tta_probs, tta_badness, record["temperature"])
        )
        corrected, _ = top2_correct(top, badness, record["threshold"])
        m = error_metrics(labels, test_probs[:, 0], badness[:, 0])
        interval = paired_accuracy_interval(
            labels, standard.argmax(1), weighted.argmax(1)
        )
        row = dict(
            mode=mode,
            seed=seed,
            base_acc=m["acc"],
            auroc=m["auroc"],
            auprc=m["auprc"],
            aurc=m["aurc"],
            top2_acc=float((corrected == labels).mean()),
            tta_acc=float((standard.argmax(1) == labels).mean()),
            uw_acc=float((weighted.argmax(1) == labels).mean()),
            uw_minus_tta_pp=interval["gain_pp"],
            uw_ci_low_pp=interval["ci_low_pp"],
            uw_ci_high_pp=interval["ci_high_pp"],
            parameters=record["trainable_parameters"],
        )
        rows.append(row)
        np.savez_compressed(
            out / f"artifacts/test_{mode}_{seed}.npz",
            labels=labels,
            badness=badness,
            corrected=corrected,
            tta_badness=tta_badness,
            weighted_probabilities=weighted,
        )
        print("result", row, flush=True)
    msp = error_metrics(labels, test_probs[:, 0], 1 - test_probs[:, 0].max(1))
    aggregate = []
    for mode in ["last", "mean", "full"]:
        group = [r for r in rows if r["mode"] == mode]
        a = dict(mode=mode, meta_seeds=len(group))
        for key in [
            "auroc",
            "auprc",
            "aurc",
            "top2_acc",
            "tta_acc",
            "uw_acc",
            "uw_minus_tta_pp",
        ]:
            a[key] = float(np.mean([r[key] for r in group]))
            a[key + "_sd"] = float(np.std([r[key] for r in group], ddof=1))
        aggregate.append(a)
    contrasts = target_contrasts(out, cfg.seeds)
    write_csv(out / "tables/paired_contrasts.csv", contrasts)
    metrics = dict(
        paired_contrasts=contrasts,
        results=rows,
        aggregate=aggregate,
        msp_error_auroc=msp["auroc"],
        base_acc=msp["acc"],
    )
    config = OmegaConf.to_container(cfg, resolve=True)
    meta = dict(
        device=str(device),
        seed=list(cfg.seeds),
        checkpoint_path=sources
        + [
            cfg.full_template.format(seed=s) + "/checkpoints/best.pt" for s in cfg.seeds
        ],
        base_checkpoint_assumption="owner confirmed",
        clean_training_supervision=False,
    )
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "Exploratory reused-test CIFAR-100N study, one base seed, three meta seeds. Same frozen inputs/partitions/hidden layers/Adam/batch/25 epochs; scalar controls have 1 output and fewer parameters than the reused 50-output head. Checkpoints minimize held-out target MSE within each arm; absolute MSE across different targets is not an architecture ranking. All downstream policy selections use noisy validation and are saved before test loads. Scalar mean still uses training-history supervision, not a purely static no-history baseline. Last margin still uses a teacher target from the final training epoch. Does not establish temporal ordering importance or new-base transfer.",
    )
    write_csv(out / "tables/summary.csv", rows)
    write_csv(out / "tables/aggregate.csv", aggregate)
    with (out / "report.md").open("a") as report:
        report.write(
            "\n## Target comparison\n\n| Target | Error AUROC | Top-2 acc, % | UW-TTA acc, % | UW minus TTA, pp |\n|---|---:|---:|---:|---:|\n"
        )
        for a in aggregate:
            report.write(
                f"| {a['mode']} | {a['auroc']:.4f} | {100 * a['top2_acc']:.2f} | {100 * a['uw_acc']:.2f} | {a['uw_minus_tta_pp']:.2f} |\n"
            )
        report.write(
            f"\nMSP error AUROC: {msp['auroc']:.4f}. Base accuracy: {100 * msp['acc']:.2f}%. Standard TTA: {100 * aggregate[0]['tta_acc']:.2f}%.\n"
        )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
