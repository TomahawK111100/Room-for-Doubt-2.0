"""Staged S2 controls: preflight, fit all, freeze all, evaluate; no base training."""

import json
import shutil
import time
import zipfile
from pathlib import Path
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
from torchvision.datasets import CIFAR100
from room_for_doubt.utils.paths import config_directory
from room_for_doubt.data.posthoc_controls import (
    inspect_inputs,
    verify_data,
    aligned_rows,
    mean_probabilities,
    REFERENCE_DIGESTS,
)
from room_for_doubt.data.temporal_evidence import load_evidence, probabilities
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.kaggle_artifacts import load_completed_run, restore_base
from room_for_doubt.inference.temporal_evidence import extract
from room_for_doubt.training.temporal_snapshots import file_digest
from room_for_doubt.training.posthoc_controls import fit_head, load_head, head_residual
from room_for_doubt.evaluation.temporal_residual import (
    corrected_probabilities,
    evaluate_probabilities,
)
from room_for_doubt.evaluation.diagnostic_metrics import probability_metrics
from room_for_doubt.evaluation.selection import paired_accuracy_interval
from room_for_doubt.evaluation.tie_aware_risk import expected_tie_aurc
from room_for_doubt.reporting.artifacts import write_csv


def branches(package):
    if package == "A":
        return [
            (s, b)
            for s in (42, 43, 44)
            for b in ("early2", "supervised_head", "direct_soft5")
        ] + [(42, "selected1")]
    if package == "B":
        return [(s, "independent2") for s in (42, 43, 44)]
    raise ValueError("Package must be A or B")


def choose_epoch(base, candidates, labels):
    rows = []
    for epoch, q in sorted(candidates.items()):
        h = np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(base, 1e-12, 1))
        for alpha in [0.0, 0.25, 0.5, 1.0]:
            m = probability_metrics(labels, corrected_probabilities(base, h, alpha))
            rows.append(dict(epoch=epoch, alpha=alpha, acc=m["acc"], nll=m["nll"]))
    return (
        max(rows, key=lambda r: (r["acc"], -r["nll"], -r["alpha"], -r["epoch"])),
        rows,
    )


def training_data(cfg):
    """No clean targets returned; only official observed noisy labels are used."""
    root = Path(cfg.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Copy only a verified torchvision CIFAR folder and official annotations, never full Outputs.
    if not (root / "cifar-100-python/train").exists():
        folders = sorted(Path(cfg.input_root).rglob("cifar-100-python"))
        if folders:
            shutil.copytree(folders[0], root / "cifar-100-python")
    if not (root / "CIFAR-100_human.pt").exists():
        labels = sorted(Path(cfg.input_root).rglob("CIFAR-100_human.pt"))
        if labels:
            shutil.copyfile(labels[0], root / "CIFAR-100_human.pt")
    raw = CIFAR100(str(root), train=True, download=True)
    noisy = load_observed_labels(str(root), "cifar100n", "noisy100")
    return raw.data, noisy


def targets(seed, branch, sources, arrays, noisy):
    a, s = arrays[seed], sources[seed]
    selected, grid = None, []
    if branch == "selected1":
        selected, grid = choose_epoch(
            a["val_probs"],
            {
                e: np.load(
                    s["cache"] / f"val_epoch{e:03d}_probs.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for e in range(1, 6)
            },
            a["val_noisy_labels"],
        )
    qs = []
    for split in ("train", "val"):
        if branch == "supervised_head":
            q = np.eye(100, dtype=np.float32)[noisy[a[f"{split}_ids"]]]
        elif branch == "direct_soft5":
            q = np.array(
                a["teacher_probs" if split == "train" else "val_teacher_probs"],
                dtype=np.float32,
            )
        elif branch in ("early2", "selected1"):
            epochs = [1, 5] if branch == "early2" else [selected["epoch"]]
            q = mean_probabilities(
                [
                    np.load(
                        s["cache"] / f"{split}_epoch{e:03d}_probs.npy",
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    for e in epochs
                ]
            )
        elif branch == "independent2":
            donors = [d for d in (42, 43, 44) if d != seed]
            q = mean_probabilities(
                [arrays[d][f"{split}_probs"] for d in donors],
                [
                    aligned_rows(arrays[d][f"{split}_ids"], a[f"{split}_ids"])
                    for d in donors
                ],
            )
        else:
            raise ValueError("Unknown branch")
        probabilities(q)
        qs.append(q)
    return qs, selected, grid


def target_contract(out, seed, branch, sources, arrays, noisy, cfg):
    root = Path(out) / "targets" / f"base{seed}_{branch}"
    root.mkdir(parents=True, exist_ok=True)
    qs, selected, grid = targets(seed, branch, sources, arrays, noisy)
    for split, q in zip(("train", "val"), qs):
        path = root / f"{split}_target.npy"
        if path.exists():
            if not np.array_equal(np.load(path, mmap_mode="r", allow_pickle=False), q):
                raise ValueError("Existing target differs; preserve the previous run")
        else:
            np.save(path, q)
    if grid:
        write_csv(root / "selection_grid.csv", grid)
    contract = dict(
        schema=1,
        protocol="s2_v1",
        recipient=seed,
        branch=branch,
        parametrization="direct" if branch == "direct_soft5" else "residual",
        classes=100,
        hidden_dim=cfg.hidden_dim,
        evidence_sha256=file_digest(sources[seed]["cache"] / "manifest.json"),
        base_digest=sources[seed]["manifest"]["base_digest"],
        target_sha256={
            split: file_digest(root / f"{split}_target.npy")
            for split in ("train", "val")
        },
        donors=(
            [d for d in (42, 43, 44) if d != seed] if branch == "independent2" else []
        ),
        donor_base_digests=(
            {
                str(d): sources[d]["manifest"]["base_digest"]
                for d in (42, 43, 44)
                if d != seed
            }
            if branch == "independent2"
            else {}
        ),
        selected_teacher=selected,
        head_seed=42,
    )
    path = root / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Target contract changed")
    path.write_text(json.dumps(contract, indent=2))
    return qs, contract


def freeze_group(out, jobs):
    out = Path(out)
    entries = {}
    for key, head in jobs.items():
        path = Path(head) / "artifacts/policy_before_test.json"
        if not path.is_file():
            raise ValueError("Every package policy must exist before test access")
        entries[key] = dict(head=str(head), policy_sha256=file_digest(path))
    path = out / "all_policies_before_test.json"
    if path.exists() and json.loads(path.read_text()) != entries:
        raise ValueError("Frozen package policies changed")
    path.write_text(json.dumps(entries, indent=2))
    return entries


def verify_group(out, jobs):
    path = Path(out) / "all_policies_before_test.json"
    if not path.is_file():
        raise ValueError("No frozen package; test is forbidden")
    entries = json.loads(path.read_text())
    if set(entries) != set(jobs):
        raise ValueError("Package membership differs")
    for key, head in jobs.items():
        if (
            entries[key]["head"] != str(head)
            or file_digest(Path(head) / "artifacts/policy_before_test.json")
            != entries[key]["policy_sha256"]
        ):
            raise ValueError("A package policy changed after freezing")
    return entries


def evaluate_group(cfg, sources, arrays, noisy, jobs):
    out = Path(cfg.output_dir)
    group = verify_group(
        out, jobs
    )  # Gate BEFORE instantiating clean test or opening raw test predictions.
    raw = CIFAR100(cfg.data_dir, train=False, download=True)
    ids, labels = np.arange(len(raw.data), dtype=np.int64), np.array(
        raw.targets, dtype=np.int64
    )
    refs = {}
    for seed, s in sources.items():
        if file_digest(s["reference"]) != REFERENCE_DIGESTS[seed]:
            raise ValueError("Original reference changed since preflight")
        with np.load(s["reference"], allow_pickle=False) as f:
            refs[seed] = {k: f[k] for k in f.files}
        if not np.array_equal(refs[seed]["ids"], ids) or not np.array_equal(
            refs[seed]["labels"], labels
        ):
            raise ValueError("Official test IDs/labels differ from original export")
    rows = []
    for seed in (42, 43, 44):
        # A single recipient feature extraction is shared by all its new heads.
        data = load_completed_run(sources[seed]["base"], noisy)
        f, p = extract(
            restore_base(data, cfg.device),
            raw.data,
            ids,
            224,
            cfg.image_batch_size,
            cfg.device,
            out / "evaluation_cache" / f"base{seed}",
            "test",
            True,
        )
        if not np.array_equal(p, refs[seed]["base"]):
            raise ValueError(
                "Canonical test base probabilities differ from pinned original; no silent tolerance"
            )
        early2 = None
        seed_jobs = [
            (key, head) for key, head in jobs.items() if key.startswith(f"base{seed}_")
        ]
        if any(key.endswith("_early2") for key, _ in seed_jobs):
            teacher_arrays = []
            for epoch in (1, 5):
                model = restore_base(data, cfg.device)
                state = torch.load(
                    sources[seed]["base"]
                    / f"checkpoints/temporal/epoch_{epoch:03d}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(state["state_dict"], strict=True)
                _, q = extract(
                    model,
                    raw.data,
                    ids,
                    224,
                    cfg.image_batch_size,
                    cfg.device,
                    out / "evaluation_cache" / f"base{seed}",
                    f"epoch{epoch}",
                    False,
                )
                teacher_arrays.append(q)
                del model, state
            early2 = mean_probabilities(teacher_arrays)
        for key, head_path in seed_jobs:
            verify_group(out, jobs)
            head_path = Path(head_path)
            head, policy = load_head(head_path, cfg.device)
            result_path = head_path / "final_evaluation"
            contract = dict(
                policy_sha256=group[key]["policy_sha256"],
                reference_sha256=file_digest(sources[seed]["reference"]),
            )
            done = result_path / "metrics.json"
            if done.exists():
                if (
                    json.loads((result_path / "evaluation_contract.json").read_text())
                    != contract
                ):
                    raise ValueError("Completed evaluation belongs to another policy")
                old = json.loads(done.read_text())
                if (
                    file_digest(result_path / "predictions.npz")
                    != old["prediction_sha256"]
                ):
                    raise ValueError("Completed predictions changed")
                rows.extend(old["compact_rows"])
                continue
            result_path.mkdir(exist_ok=True)
            cp = result_path / "evaluation_contract.json"
            if cp.exists() and json.loads(cp.read_text()) != contract:
                raise ValueError("Interrupted evaluation contract differs")
            cp.write_text(json.dumps(contract, indent=2))
            branch = policy["contract"]["branch"]
            h = head_residual(
                head,
                f,
                p,
                policy["contract"]["parametrization"],
                cfg.head_batch_size,
                cfg.device,
            )
            corrected = corrected_probabilities(p, h, policy["policy"]["alpha"])
            values = dict(base=p, corrected=corrected)
            q = None
            if branch == "early2":
                q = early2
            elif branch == "direct_soft5":
                q = refs[seed]["snapshot_mean"]
            elif branch == "selected1":
                epoch = policy["contract"]["selected_teacher"]["epoch"]
                model = restore_base(data, cfg.device)
                state = torch.load(
                    sources[seed]["base"]
                    / f"checkpoints/temporal/epoch_{epoch:03d}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(state["state_dict"], strict=True)
                _, q = extract(
                    model,
                    raw.data,
                    ids,
                    224,
                    cfg.image_batch_size,
                    cfg.device,
                    out / "evaluation_cache" / f"base{seed}",
                    "selected1",
                    False,
                )
                del model, state
            elif branch == "independent2":
                q = mean_probabilities(
                    [refs[d]["base"] for d in (42, 43, 44) if d != seed]
                )
            if q is not None:
                values["teacher"] = q
                th = np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1))
                values["teacher_interpolation"] = corrected_probabilities(
                    p, th, policy["teacher_policy"]["alpha"]
                )
            methods = {}
            for name, pred in values.items():
                methods[name] = evaluate_probabilities(labels, p, pred)[
                    "temporal_residual"
                ]
                methods[name]["aurc_expected_ties"] = expected_tie_aurc(
                    pred.max(1), pred.argmax(1) != labels
                )
            baseline_ok, corrected_ok = (
                p.argmax(1) == labels,
                corrected.argmax(1) == labels,
            )
            metrics = dict(
                status="complete",
                methods=methods,
                paired_vs_base=paired_accuracy_interval(
                    labels, p.argmax(1), corrected.argmax(1)
                ),
                corrected=int((~baseline_ok & corrected_ok).sum()),
                harmed=int((baseline_ok & ~corrected_ok).sum()),
                changed=int((p.argmax(1) != corrected.argmax(1)).sum()),
                unchanged=int((p.argmax(1) == corrected.argmax(1)).sum()),
                exploratory=True,
            )
            compact = [
                dict(
                    base_seed=seed,
                    branch=branch,
                    method=name,
                    **{
                        k: None if v is None else float(f"{v:.4g}")
                        for k, v in m.items()
                    },
                )
                for name, m in methods.items()
            ]
            metrics["compact_rows"] = compact
            rows.extend(compact)
            np.savez_compressed(
                result_path / "predictions.npz", ids=ids, labels=labels, **values
            )
            metrics["prediction_sha256"] = file_digest(result_path / "predictions.npz")
            (result_path / "tables").mkdir(exist_ok=True)
            OmegaConf.save(
                OmegaConf.create(dict(stage="evaluate", **contract)),
                result_path / "config.yaml",
            )
            (result_path / "run_metadata.json").write_text(
                json.dumps(contract, indent=2)
            )
            (result_path / "report.md").write_text(
                f"# Experiment Report: {key}\n\n## Configuration\nFrozen S2 evaluation, see config.yaml.\n\n## Dataset\nOfficial CIFAR100 test; exploratory reuse.\n\n## Checkpoint\nBound to policy SHA256.\n\n## Metrics\nSee metrics.json.\n\n## Compact Results\nSee tables/methods.csv.\n\n## Generated Artifacts\nRaw predictions and frozen contract.\n\n## Notes\nConditional paired intervals; no independent confirmation.\n"
            )
            write_csv(result_path / "tables/methods.csv", compact)
            done.write_text(json.dumps(metrics, indent=2))
        del f, p, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (out / "tables").mkdir(exist_ok=True)
    write_csv(out / "tables/methods.csv", rows)
    return rows


def exports(out, metrics, cfg, report=None):
    out = Path(out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                protocol="s2_v1",
                package=cfg.package,
                device=cfg.device,
                test_scope="post-pilot exploratory reuse",
                base_training=False,
            ),
            indent=2,
        )
    )
    (out / "report.md").write_text(
        "# Experiment Report: "
        + cfg.run_id
        + "\n\n## Configuration\nSee config.yaml.\n\n## Dataset\nCIFAR100N noisy100; known test reused only for frozen evaluation.\n\n## Checkpoint\nNo base training; existing bases pinned. New heads min reserved meta-val KL.\n\n## Metrics\nSee metrics.json.\n\n## Compact Results\nSee tables/.\n\n## Generated Artifacts\nPreflight, targets, heads, policies, raw predictions and stage status.\n\n## Notes\n"
        + metrics["status"]
        + "; benchmark is a separate stage and is not implied by accuracy completion.\n"
    )
    (out / "morning_summary.txt").write_text(
        "Status: "
        + metrics["status"]
        + "\nPackage: "
        + cfg.package
        + "\n"
        + str(metrics.get("error", ""))
        + "\n"
    )
    for full in (False, True):
        dest = out / ("posthoc_results.zip" if full else "posthoc_summary.zip")
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob("*")):
                if not p.is_file() or any(
                    n in p.relative_to(out).parts
                    for n in ("evaluation_cache", "matplotlib")
                ):
                    continue
                if p.suffix in {".json", ".yaml", ".md", ".txt", ".csv", ".log"} or (
                    full and p.name == "predictions.npz"
                ):
                    z.write(p, str(p.relative_to(out)))


def canonical_witness(cfg, sources, images, noisy, arrays):
    """Check a full canonical source batch on noisy-validation BEFORE any fitting."""
    report = []
    for seed, source in sources.items():
        bundle = load_completed_run(source["base"], noisy)
        ids = arrays[seed]["val_ids"][: cfg.image_batch_size]
        directory = Path(cfg.output_dir) / "artifacts/canonical_witness" / str(seed)
        f, p = extract(
            restore_base(bundle, cfg.device),
            images,
            ids,
            224,
            cfg.image_batch_size,
            cfg.device,
            directory,
            "val",
            True,
        )
        if not np.array_equal(
            p, arrays[seed]["val_probs"][: len(ids)]
        ) or not np.array_equal(f, arrays[seed]["val_features"][: len(ids)]):
            raise ValueError(
                f"Canonical source validation forward changed on base{seed}; review environment before fit"
            )
        report.append(
            dict(
                seed=seed,
                images=len(ids),
                base_digest=bundle["checkpoint_digest"],
                features_and_probabilities="bit-identical to sealed source validation batch",
            )
        )
        del bundle, f, p
    (Path(cfg.output_dir) / "artifacts/canonical_witness.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


def resource_smoke(cfg, sources, arrays):
    """Bounded train-cache GPU witness; estimates compute component, not total runtime."""
    from room_for_doubt.models.temporal_residual import (
        TemporalResidualHead,
        distillation_kl,
    )

    device = torch.device(cfg.device)
    a = arrays[42]
    torch.manual_seed(42)
    model = TemporalResidualHead(
        np.zeros(2048, dtype=np.float32), np.ones(2048, dtype=np.float32), 100, 256
    ).to(device)
    rows = a["meta_train_rows"][:256]
    f, p, q = [
        torch.from_numpy(np.array(a[name][rows], dtype=np.float32)).to(device)
        for name in ("train_features", "train_probs", "teacher_probs")
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    times = []
    for step in range(15):
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = distillation_kl(p, model(f), q)
        if not torch.isfinite(loss):
            raise FloatingPointError("S2 resource smoke is nonfinite")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        if step >= 5:
            times.append(time.perf_counter() - began)
    compute_seconds = (
        float(np.median(times)) * 25 * int(np.ceil(len(a["meta_train_rows"]) / 256))
    )
    info = dict(
        scope="15 train-cache optimizer steps, first5 warmup; no test",
        head_compute_component_seconds=compute_seconds,
        warning="Excludes CPU data loading, full feature stats, meta-validation, export, target generation and runtime benchmark; not a wall-time guarantee",
        packages_new_head_fits=dict(A=10, B=3),
        effective_training_device=str(device),
        source_costs={},
    )
    import csv

    for seed, source in sources.items():
        with (source["base"] / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        info["source_costs"][str(seed)] = dict(
            base_epoch_seconds=sum(float(r["seconds"]) for r in history),
            base_checkpoint_bytes=(source["base"] / "checkpoints/best.pt")
            .stat()
            .st_size,
            snapshot_bytes=sum(
                (source["base"] / f"checkpoints/temporal/epoch_{e:03d}.pt")
                .stat()
                .st_size
                for e in range(1, 6)
            ),
            sealed_cache_bytes=sum(
                (source["cache"] / v["file"]).stat().st_size
                for v in source["manifest"]["arrays"].values()
            ),
            prior_extraction_seconds=None,
            prior_snapshot_write_seconds=None,
            unavailable_cost_policy="null means unmeasured, not zero",
        )
    return info


def execute(cfg):
    branches(cfg.package)
    if cfg.stage not in ("preflight", "fit", "evaluate", "all", "benchmark"):
        raise ValueError("stage must be preflight,fit,evaluate,all,benchmark")
    if (
        cfg.head_epochs,
        cfg.head_batch_size,
        cfg.hidden_dim,
        cfg.head_seed,
        cfg.protocol,
    ) != (25, 256, 256, 42, "s2_v1"):
        raise ValueError(
            "Frozen S2 budget/protocol differs; explicitly version protocol first"
        )
    for key in ("input_root", "output_dir", "data_dir"):
        cfg[key] = to_absolute_path(cfg[key])
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    # Stages may change on resume; scientific/runtime settings may not.
    previous = out / "config.yaml"
    if previous.exists():
        old = OmegaConf.to_container(OmegaConf.load(previous), resolve=True)
        if {k: v for k, v in old.items() if k != "stage"} != {
            k: v for k, v in config.items() if k != "stage"
        }:
            raise ValueError("Resume settings changed")
    OmegaConf.save(cfg, previous, resolve=True)
    metrics = dict(status="preflight", package=cfg.package)
    try:
        report, sources = inspect_inputs(cfg.input_root)
        (out / "preflight.json").write_text(json.dumps(report, indent=2))
        if report["status"] != "verified_files":
            metrics = dict(status="missing_input", preflight=report)
            return metrics
        if cfg.stage == "preflight":
            metrics = dict(status="preflight_files_verified", test_opened=False)
            return metrics
        if cfg.require_cuda and not str(cfg.device).startswith("cuda"):
            raise RuntimeError("S2 real runs require CUDA")
        if cfg.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("Select CUDA accelerator")
        images, noisy = training_data(cfg)
        verify_data(sources, noisy)
        arrays = {seed: load_evidence(s["cache"])[1] for seed, s in sources.items()}
        jobs = {
            f"base{seed}_{branch}": out / "heads" / f"base{seed}_{branch}"
            for seed, branch in branches(cfg.package)
        }
        if cfg.stage in ("fit", "all"):
            canonical_witness(cfg, sources, images, noisy, arrays)
            smoke = resource_smoke(cfg, sources, arrays)
            (out / "artifacts").mkdir(exist_ok=True)
            (out / "artifacts/resource_smoke.json").write_text(
                json.dumps(smoke, indent=2)
            )
            for seed, branch in branches(cfg.package):
                q, contract = target_contract(
                    out, seed, branch, sources, arrays, noisy, cfg
                )
                began = time.perf_counter()
                print("FIT", seed, branch, flush=True)
                fit_head(
                    jobs[f"base{seed}_{branch}"],
                    arrays[seed],
                    *q,
                    contract,
                    cfg.device,
                    epochs=cfg.head_epochs,
                    batch_size=cfg.head_batch_size,
                )
                print(
                    "FROZEN",
                    seed,
                    branch,
                    round(time.perf_counter() - began, 2),
                    flush=True,
                )
            freeze_group(out, jobs)
            metrics = dict(status="policies_frozen", heads=len(jobs), test_opened=False)
        if cfg.stage in ("evaluate", "all"):
            evaluate_group(cfg, sources, arrays, noisy, jobs)
            metrics = dict(
                status="accuracy_controls_complete",
                heads=len(jobs),
                runtime_benchmark="pending_separate_stage",
                exploratory=True,
            )
        if cfg.stage == "benchmark" or (cfg.stage == "all" and cfg.benchmark):
            from room_for_doubt.evaluation.posthoc_runtime import benchmark

            runtime = benchmark(
                out / "runtime", sources, images, noisy, cfg.device, choose_epoch
            )
            metrics.update(
                runtime=runtime,
                status=(
                    "accuracy_and_runtime_complete"
                    if cfg.stage == "all"
                    else "runtime_complete"
                ),
                runtime_benchmark=runtime["status"],
            )
        return metrics
    except Exception as exc:
        metrics = dict(status="failed", error=str(exc), package=cfg.package)
        raise
    finally:
        exports(out, metrics, cfg)


@hydra.main(
    config_path=str(config_directory()),
    config_name="posthoc_controls",
    version_base=None,
)
def main(cfg):
    print(json.dumps(execute(cfg), indent=2), flush=True)
