"""Prepare both frozen S4 heads and benchmark on noisy validation only."""

from pathlib import Path
from time import monotonic

from omegaconf import OmegaConf
from room_for_doubt.analysis.temporal_distillation import prepare, source_data
from room_for_doubt.data.confirmation import (
    CLASSES,
    METHODS,
    PROTOCOL,
    check_base_config,
    seal_frozen,
)
from room_for_doubt.data.temporal_evidence import load_evidence
from room_for_doubt.evaluation.posthoc_runtime import benchmark
from room_for_doubt.inference.confirmation import make_confirmation_plan
from room_for_doubt.training.posthoc_controls import fit_head
from room_for_doubt.training.temporal_snapshots import file_digest


def fit_and_freeze(root, data, seed, protocol_hash, device):
    root = Path(root).resolve()
    source, cache = root / "base", root / "evidence"
    check_base_config(
        OmegaConf.to_container(OmegaConf.load(source / "config.yaml")), seed
    )
    cfg = OmegaConf.create(
        dict(
            run_id=f"s4_fit_seed{seed}",
            source_run=str(source),
            cache_dir=str(cache),
            data_dir=str(Path(data).resolve()),
            teacher_epochs=[1, 2, 3, 4, 5],
            image_batch_size=64,
            protocol=PROTOCOL,
            protocol_sha256=protocol_hash,
        )
    )
    bundle, raw, noisy, _ = source_data(cfg)
    if tuple(raw.classes) != CLASSES:
        raise ValueError("Unexpected CIFAR-10 class order")
    del bundle
    started = monotonic()
    prepared = prepare(cfg, root, device)
    extraction_seconds = monotonic() - started
    manifest, arrays = load_evidence(cache)
    if manifest["feature_width"] != 2048 or manifest["classes"] != 10:
        raise ValueError("Unexpected feature or class dimensions")
    for branch, parametrization in (("early5", "residual"), ("direct_soft5", "direct")):
        contract = dict(
            protocol=PROTOCOL,
            protocol_sha256=protocol_hash,
            recipient=seed,
            branch=branch,
            parametrization=parametrization,
            classes=10,
            hidden_dim=256,
            evidence_sha256=file_digest(cache / "manifest.json"),
            base_digest=manifest["base_digest"],
            target_sha256={
                split: manifest["arrays"][name]["sha256"]
                for split, name in (
                    ("train", "teacher_probs"),
                    ("val", "val_teacher_probs"),
                )
            },
            head_seed=42,
            dataset=manifest["dataset"],
        )
        fit_head(
            root / branch,
            arrays,
            arrays["teacher_probs"],
            arrays["val_teacher_probs"],
            contract,
            device,
        )
    del arrays
    # The same forward plans are used in timing and final prediction.
    plans = {
        name: lambda bundle, method=name: make_confirmation_plan(
            root, bundle, method, device
        )
        for name in METHODS
    }
    runtime = benchmark(
        root / "runtime",
        {seed: dict(base=source, cache=cache, manifest=manifest)},
        raw.data,
        noisy,
        device,
        None,
        plans=plans,
        protocol=PROTOCOL,
    )
    seal_frozen(root, seed, protocol_hash)
    return {
        "status": "frozen",
        "test_accessed": False,
        "runtime": runtime,
        "extraction_seconds": (
            extraction_seconds if prepared["status"] == "complete" else None
        ),
    }
