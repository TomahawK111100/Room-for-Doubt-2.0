"""Evaluate matched static/predicted-trajectory LDA from paired exported arrays."""

from pathlib import Path
import csv
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from room_for_doubt.analysis.matched_lda import evaluate_matched_lda
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory


@hydra.main(
    config_path=str(config_directory()),
    config_name="matched_lda_ablation",
    version_base=None,
)
def main(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if not cfg.input_path:
        raise ValueError(
            "Provide input_path to a paired array export from a corrected meta-model; see docs/matched_ablation.md"
        )
    if cfg.label_kind not in {"noise_label_detection", "classifier_error_detection"}:
        raise ValueError(
            "Declare whether target 1 means label corruption or classifier error"
        )
    if cfg.supervision != "oracle_clean_labels":
        raise ValueError(
            "This supervised comparison uses clean-label targets and must be marked oracle"
        )
    if (cfg.detector.name, cfg.detector.solver, cfg.detector.shrinkage) != (
        "lda",
        "lsqr",
        "auto",
    ):
        raise ValueError("This controlled comparison fixes LDA lsqr/auto in both arms")
    cfg.input_path = to_absolute_path(cfg.input_path)
    with np.load(cfg.input_path, allow_pickle=False) as arrays:
        if str(arrays["label_kind"].item()) != cfg.label_kind:
            raise ValueError("Exported label meaning does not match configured target")
        if (
            str(arrays["representation_kind"].item())
            != "shared_features_predicted_trajectories"
        ):
            raise ValueError(
                "Use paired shared-backbone predictions, not real training trajectories"
            )
        checkpoint = str(arrays["meta_checkpoint_path"].item())
        if not checkpoint:
            raise ValueError("Paired arrays must identify their source meta checkpoint")
        metrics = evaluate_matched_lda(
            *(
                arrays[k]
                for k in [
                    "static_features",
                    "predicted_trajectories",
                    "labels",
                    "sample_ids",
                    "fit_ids",
                    "eval_ids",
                    "meta_train_ids",
                ]
            )
        )
    out = Path(HydraConfig.get().runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    save_metrics(out, metrics)
    metadata = dict(
        seed=cfg.seed,
        checkpoint_path=checkpoint,
        device="CPU LDA",
        supervision=cfg.supervision,
        label_kind=cfg.label_kind,
        source_arrays=cfg.input_path,
    )
    save_run_metadata(out, cfg.run_id, config, **metadata)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        metadata,
        "Matched shared image features and class embedding; identical fit/evaluation IDs, labels and LDA settings. Both splits are disjoint from meta-model training. The static_shared baseline shares a learned class embedding, so this comparison tests the trajectory representation on a common trained input, not absence of all temporal supervision. Array provenance must be verified by the producing pipeline.",
    )
    with (out / "tables/summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "acc", "auroc", "fit_n", "eval_n"])
        for name, values in metrics.items():
            writer.writerow(
                [
                    name,
                    f"{values['acc']:.4g}",
                    f"{values['auroc']:.4g}",
                    values["fit_n"],
                    values["eval_n"],
                ]
            )
    print(f"Matched comparison saved to {out}")


if __name__ == "__main__":
    main()
