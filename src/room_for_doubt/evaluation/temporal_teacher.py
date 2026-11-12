"""Evaluate the preregistered direct-distillation contrast on CIFAR-10.2."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10

from room_for_doubt.analysis.temporal_teacher_fit import BRANCHES, FROZEN_FILES
from room_for_doubt.data.cifar10_1 import load_v6
from room_for_doubt.data.cifar10_2 import BLOB, load_test
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.confirmation import CLASSES, SEEDS, verify_frozen, write_json
from room_for_doubt.data.kaggle_artifacts import load_completed_run, restore_base
from room_for_doubt.evaluation.confirmation import paired_stratified_interval, predict, predict_batches
from room_for_doubt.evaluation.diagnostic_metrics import probability_metrics
from room_for_doubt.evaluation.posthoc_runtime import ForwardPlan
from room_for_doubt.evaluation.tie_aware_risk import expected_tie_aurc
from room_for_doubt.inference.static_teacher import STATIC_VIEWS, static_teacher_probabilities
from room_for_doubt.reporting.artifacts import write_csv
from room_for_doubt.training.posthoc_controls import load_head
from room_for_doubt.training.temporal_snapshots import file_digest, verify_snapshots

METHODS = ("base", "direct_soft5", *BRANCHES, "early5_teacher", "static5_teacher")


def verify_inputs(sources, controls, s4_hash, protocol_hash):
    if set(sources) != set(SEEDS) or set(controls) != set(SEEDS):
        raise ValueError("All three source and control manifests are required")
    lineage, base_digests, epochs = {}, set(), {}
    for seed in SEEDS:
        source, control = Path(sources[seed]), Path(controls[seed])
        source_manifest = verify_frozen(source, seed, s4_hash)
        base_digest = source_manifest["files"]["base/checkpoints/best.pt"]
        base_digests.add(base_digest)
        source_hash = file_digest(source / "frozen.json")
        path = control / "frozen.json"
        seal = json.loads(path.read_text())
        config = seal["config"]
        required = dict(seed=seed, source_frozen_sha256=source_hash,
                        protocol_sha256=protocol_hash, s4_protocol_sha256=s4_hash,
                        views=list(STATIC_VIEWS), test_accessed=False)
        if (
            any(config.get(key) != value for key, value in required.items())
            or config != OmegaConf.to_container(OmegaConf.load(control / "config.yaml"))
            or set(seal["files"]) != set(FROZEN_FILES)
        ):
            raise ValueError(f"Control manifest does not match seed {seed}")
        for name, digest in seal["files"].items():
            if file_digest(control / name) != digest:
                raise ValueError(f"Changed control file for seed {seed}: {name}")
        metrics = json.loads((control / "metrics.json").read_text())
        epoch = metrics["selected_epoch"]
        if metrics["status"] != "frozen" or metrics["test_accessed"] is not False or epoch not in range(1, 6):
            raise ValueError(f"Invalid frozen metrics for seed {seed}")
        teacher_digest = source_manifest["files"][f"base/checkpoints/temporal/epoch_{epoch:03}.pt"]
        for branch in BRANCHES:
            _, policy = load_head(control / branch, "cpu")
            expected = dict(protocol="temporal_teacher_v1", protocol_sha256=protocol_hash,
                            source_frozen_sha256=source_hash, recipient=seed, branch=branch,
                            parametrization="direct", base_digest=base_digest,
                            teacher_epoch=epoch, teacher_sha256=teacher_digest)
            if (any(policy["contract"].get(key) != value for key, value in expected.items())
                    or policy["test_accessed"] is not False
                    or policy["policy"] != metrics["policies"][branch]
                    or policy["policy"]["alpha"] not in (0, 0.25, 0.5, 1)):
                raise ValueError(f"Control policy does not match seed {seed}: {branch}")
        _, direct = load_head(source / "direct_soft5", "cpu")
        if (direct["contract"]["base_digest"] != base_digest
                or direct["contract"]["recipient"] != seed
                or direct["contract"]["protocol_sha256"] != s4_hash
                or direct["contract"]["branch"] != "direct_soft5"
                or direct["test_accessed"] is not False):
            raise ValueError(f"S4 direct head does not match seed {seed}")
        lineage[str(seed)] = dict(source=source_hash, control=file_digest(path))
        epochs[seed] = epoch
    if len(base_digests) != len(SEEDS):
        raise ValueError("Expected three distinct base checkpoints")
    return lineage, epochs


def predict_control(source, control, bundle, images, method, epoch, device, scratch):
    model = restore_base(bundle, device)
    if method == "static5_teacher":
        path = source / f"base/checkpoints/temporal/epoch_{epoch:03}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        return static_teacher_probabilities(model, images, np.arange(len(images)),
                                            224, 32, device, scratch)
    head, policy = load_head(control / method, device)
    plan = ForwardPlan({"base": model}, "base", "direct",
                       alpha=policy["policy"]["alpha"], head=head)
    return predict_batches(plan, images, device)


def contrast(correctness, method, baseline, labels):
    delta = np.stack(correctness[method]).astype(float) - np.stack(correctness[baseline])
    gains = delta.mean(1) * 100
    return dict(per_seed_pp=gains.tolist(), mean_pp=float(gains.mean()),
                between_seed_sd_pp=float(gains.std(ddof=1)),
                paired_ci95_pp=paired_stratified_interval(delta, labels))


def evaluate(sources, controls, data, out, scratch, s4_hash, protocol_hash, device):
    sources = {seed: Path(path) for seed, path in sources.items()}
    controls = {seed: Path(path) for seed, path in controls.items()}
    lineage, epochs = verify_inputs(sources, controls, s4_hash, protocol_hash)
    contract = dict(protocol_sha256=protocol_hash, s4_protocol_sha256=s4_hash,
                    frozen_manifests=lineage, dataset_blob=BLOB)
    out, scratch = Path(out), Path(scratch)
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    path = out / "evaluation_contract.json"
    if path.exists() and (json.loads(path.read_text()) != contract or (out / "metrics.json").exists()):
        raise ValueError("Output already belongs to another cohort or a completed evaluation")
    write_json(path, contract)

    train = CIFAR10(str(data), train=True, download=True)
    original_test = CIFAR10(str(data), train=False, download=False)
    if tuple(train.classes) != CLASSES or tuple(original_test.classes) != CLASSES:
        raise ValueError("Unexpected CIFAR-10 class order")
    noisy = load_observed_labels(str(data), "cifar10n", "worse")
    for source in sources.values():
        bundle = load_completed_run(source / "base", noisy)
        verify_snapshots(bundle["source"], bundle["config"], [1, 2, 3, 4, 5])
    del bundle
    v6, _, v6_hashes = load_v6(scratch / "cifar10_1_v6", train.data)
    images, labels, source_ids, digest = load_test({
        "CIFAR-10 train/val": train.data, "CIFAR-10 test": original_test.data,
        "CIFAR-10.1 v6": v6,
    })
    config = dict(**contract, dataset="CIFAR-10.2 test", dataset_sha256=digest,
                  overlap_reference_v6_sha256=v6_hashes, seeds=list(SEEDS),
                  methods=list(METHODS), device=str(device), bootstrap_seed=20261015,
                  bootstrap_repeats=10000, selected_epochs={str(k): v for k, v in epochs.items()})
    OmegaConf.save(OmegaConf.create(config), out / "config.yaml", resolve=True)
    rows, correctness = [], {method: [] for method in METHODS}
    for seed in SEEDS:
        source, control = sources[seed], controls[seed]
        bundle = load_completed_run(source / "base", noisy)
        saved = {}
        with TemporaryDirectory(dir=scratch) as work:
            for method in METHODS:
                if method in (*BRANCHES, "static5_teacher"):
                    probs = predict_control(source, control, bundle, images, method,
                                            epochs[seed], device, work)
                else:
                    probs = predict(source, bundle, images, method, device)
                correct = probs.argmax(1) == labels
                correctness[method].append(correct)
                base_correct = correctness["base"][-1]
                rows.append(dict(seed=seed, method=method, **probability_metrics(labels, probs),
                                 aurc=expected_tie_aurc(probs.max(1), ~correct),
                                 corrected=int((correct & ~base_correct).sum()),
                                 harmed=int((~correct & base_correct).sum())))
                saved[method] = probs
        np.savez_compressed(out / f"seed{seed}_predictions.npz", ids=np.arange(len(labels)),
                            source_ids=source_ids, labels=labels, **saved)
    primary = contrast(correctness, "direct_soft5", "static5_direct", labels)
    versus_base = contrast(correctness, "direct_soft5", "base", labels)
    primary_pass = primary["mean_pp"] >= 0.5 and primary["paired_ci95_pp"][0] > 0
    metrics = dict(
        status="complete", examples=len(labels), primary=primary, direct_vs_base=versus_base,
        primary_pass=primary_pass,
        practical_pass=primary_pass and versus_base["paired_ci95_pp"][0] > 0,
        secondary={
            "direct_vs_selected1": contrast(correctness, "direct_soft5", "selected1_direct", labels),
            "static5_vs_selected1": contrast(correctness, "static5_direct", "selected1_direct", labels),
            "early_teacher_vs_static_teacher": contrast(correctness, "early5_teacher", "static5_teacher", labels),
        },
    )
    write_csv(out / "metrics.csv", rows)
    write_json(out / "run_metadata.json", config)
    (out / "report.md").write_text(
        "# CIFAR-10.2 teacher comparison\n\n"
        f"Early-5 direct minus Static-5 direct: {primary['mean_pp']:+.2f} pp. "
        f"Prespecified primary criterion passed: {primary_pass}.\n\n"
        "See metrics.csv, metrics.json and the three prediction archives. "
        "Paired intervals condition on the fitted bases.\n"
    )
    write_json(out / "metrics.json", metrics)
    return metrics
