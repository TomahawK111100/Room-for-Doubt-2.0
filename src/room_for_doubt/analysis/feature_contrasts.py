"""Recompute direct paired target comparisons from a complete Kaggle Output."""

import json
import zipfile
from pathlib import Path
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from room_for_doubt.data.feature_artifacts import completed_study, shared_split
from room_for_doubt.evaluation.paired_targets import target_contrasts
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory


def run_contrasts(cfg, out):
    if not cfg.study_run:
        raise ValueError("study_run must be a full completed feature-study Output")
    study, out = Path(cfg.study_run).resolve(), Path(out).resolve()
    saved, frozen = completed_study(study)
    post = Path(cfg.post_run).resolve() if cfg.post_run else study / "shared"
    if any(
        out == root or root in out.parents or out in root.parents
        for root in (study, post)
    ):
        raise ValueError("Keep contrast output separate from source study/cache")
    for name in ("artifacts", "tables"):
        (out / name).mkdir(parents=True, exist_ok=True)
    effective = OmegaConf.merge(saved, cfg)
    effective.base_digest = saved.base_digest
    old = out / "config.yaml"
    if old.exists() and OmegaConf.to_container(
        OmegaConf.load(old), resolve=True
    ) != OmegaConf.to_container(effective, resolve=True):
        raise ValueError("Existing contrast configuration differs")
    OmegaConf.save(effective, old, resolve=True)
    p, labels, ids = shared_split(study, post, saved, "test")
    # Preserve the original frozen policy, not a selection made by this analysis.
    (out / "artifacts/policies_before_test.json").write_text(
        json.dumps(frozen, indent=2)
    )
    for name in frozen["selected"]:
        if "__" in name:
            value = np.load(study / "artifacts" / f"test_{name}.npy")
            np.save(out / "artifacts" / f"test_{name}.npy", value)
    rows = target_contrasts(
        saved,
        out,
        p,
        labels,
        ids,
        frozen,
        cfg.auroc_bootstrap_draws,
        cfg.bootstrap_seed,
    )
    metrics = dict(
        status="complete",
        comparisons=len(rows),
        samples=len(ids),
        base_retrained=False,
        heads_retrained=False,
    )
    metadata = dict(
        base_digest=saved.base_digest,
        source_study=str(study),
        statistical_unit="test example conditional on a fitted base and frozen policies",
    )
    save_metrics(out, metrics)
    save_run_metadata(
        out, cfg.run_id, OmegaConf.to_container(effective, resolve=True), **metadata
    )
    generate_report(
        out,
        cfg.run_id,
        OmegaConf.to_container(effective, resolve=True),
        metrics,
        metadata,
        "Exploratory post-selection comparisons. Accuracy intervals use 2000 paired draws. "
        "AUROC intervals use paired, class-stratified draws on fixed original-view base errors. "
        "No multiplicity correction; no inference about base-training variance. No new policy selection.",
    )
    with zipfile.ZipFile(
        out / "summary_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in (
            "config.yaml",
            "metrics.json",
            "run_metadata.json",
            "report.md",
            "artifacts/policies_before_test.json",
        ):
            archive.write(out / name, name)
        for path in (out / "tables").glob("*.csv"):
            archive.write(path, str(path.relative_to(out)))
    return metrics


@hydra.main(
    config_path=str(config_directory()),
    config_name="feature_contrasts",
    version_base=None,
)
def main(cfg):
    run_contrasts(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
