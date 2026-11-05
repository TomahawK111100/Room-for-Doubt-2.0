"""Isolated resident-weight CUDA timings for the declared S2 inference plans."""

import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from room_for_doubt.data.kaggle_artifacts import load_completed_run, restore_base
from room_for_doubt.training.temporal_residual import load_frozen
from room_for_doubt.data.observed_images import imagenet_transform
from room_for_doubt.reporting.artifacts import write_csv


class ForwardPlan:
    """One call per unique weights state; feature hook shares the base forward."""

    def __init__(self, models, base_key, kind, teacher_keys=(), alpha=1.0, head=None):
        self.models, self.base_key, self.kind = models, base_key, kind
        self.teacher_keys, self.alpha, self.head = tuple(teacher_keys), alpha, head

    def __call__(self, x):
        captured = []
        hook = None
        if self.head is not None:
            hook = self.models[self.base_key].fc.register_forward_pre_hook(
                lambda _, args: captured.append(args[0])
            )
        try:
            outputs = {
                key: model(x).float().softmax(1) for key, model in self.models.items()
            }
        finally:
            if hook is not None:
                hook.remove()
        if self.kind == "base":
            return outputs[self.base_key]
        if self.kind == "student":
            p = outputs[self.base_key]
            if self.alpha == 0:
                return p
            return (
                p.clamp_min(1e-12).log() + self.alpha * self.head(captured[0])
            ).softmax(1)
        q = torch.stack([outputs[k] for k in self.teacher_keys]).mean(0)
        q = q / q.sum(1, keepdim=True)
        if self.kind == "teacher":
            return q
        p = outputs[self.base_key]
        if self.alpha == 0:
            return p
        return (
            (1 - self.alpha) * p.clamp_min(1e-12).log()
            + self.alpha * q.clamp_min(1e-12).log()
        ).softmax(1)


def make_plan(seed, kind, sources, bundles, selected, device):
    s = sources[seed]
    b = bundles[seed]
    states = {}
    teacher_keys = []
    base_key = f"base{seed}"

    def base(which):
        key = f"base{which}"
        states[key] = bundles[which]["checkpoint"]["state_dict"]
        return key

    def epoch(e):
        saved = torch.load(
            s["base"] / f"checkpoints/temporal/epoch_{e:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        key = f"base{seed}_epoch{e}"
        # File digests differ because checkpoint metadata differ; verify actual weight equality.
        if b["checkpoint"]["epoch"] == e - 1:
            reference = b["checkpoint"]["state_dict"]
            if set(reference) != set(saved["state_dict"]) or any(
                not torch.equal(reference[k], saved["state_dict"][k]) for k in reference
            ):
                raise ValueError(
                    "Selected base/snapshot epoch claims equality but weights differ"
                )
            key = base_key
        states[key] = saved["state_dict"]
        return key

    head = None
    alpha = 1.0
    mode = "teacher"
    if kind in ("base", "student", "selected1+base"):
        base(seed)
    if kind == "base":
        mode = "base"
    elif kind == "student":
        head, policy = load_frozen(s["head"], s["cache"], device)
        alpha = policy["policy"]["alpha"]
        mode = "student"
    elif kind == "selected1+base":
        teacher_keys = [epoch(selected["epoch"])]
        alpha = selected["alpha"]
        mode = "mixture"
    elif kind in ("early2", "early5"):
        teacher_keys = [
            epoch(e) for e in ((1, 5) if kind == "early2" else (1, 2, 3, 4, 5))
        ]
    elif kind == "independent2":
        teacher_keys = [base(d) for d in (42, 43, 44) if d != seed]
    else:
        raise ValueError("Unknown runtime plan")
    models = {}
    for key, state in states.items():
        model = restore_base(b, device)
        model.load_state_dict(state, strict=True)
        models[key] = model
    return ForwardPlan(models, base_key, mode, teacher_keys, alpha, head), len(states)


def benchmark(
    out,
    sources,
    images,
    noisy,
    device,
    choose_epoch,
    *,
    warmup=20,
    blocks=10,
    repeats=50,
):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "Real S2 timing requires CUDA; CPU timings cannot substitute"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(
        protocol="s2_v1",
        warmup=warmup,
        blocks=blocks,
        repeats=repeats,
        batch_sizes=[1, 32],
        precision="float32",
        selection_inputs="first32 original noisy-val IDs",
        base_digests={str(k): v["manifest"]["base_digest"] for k, v in sources.items()},
    )
    cp = out / "config.yaml"
    from omegaconf import OmegaConf

    if cp.exists() and OmegaConf.to_container(OmegaConf.load(cp)) != contract:
        raise ValueError("Runtime scientific settings changed")
    OmegaConf.save(OmegaConf.create(contract), cp)
    from room_for_doubt.data.temporal_evidence import load_evidence

    arrays = {s: load_evidence(v["cache"])[1] for s, v in sources.items()}
    bundles = {s: load_completed_run(v["base"], noisy) for s, v in sources.items()}
    selections = {}
    for seed, s in sources.items():
        selections[seed], grid = choose_epoch(
            arrays[seed]["val_probs"],
            {
                e: np.load(
                    s["cache"] / f"val_epoch{e:03d}_probs.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for e in range(1, 6)
            },
            arrays[seed]["val_noisy_labels"],
        )
        selection_dir = out / "selection_replay" / f"base{seed}"
        selection_dir.mkdir(parents=True, exist_ok=True)
        write_csv(selection_dir / "selection_grid.csv", grid)
        (selection_dir / "selection.json").write_text(
            json.dumps(selections[seed], indent=2)
        )
    progress = out / "samples.json"
    rows = json.loads(progress.read_text()) if progress.exists() else []
    keys = {
        (r["seed"], r["batch_size"], r["scope"], r["block"], r["method"]) for r in rows
    }
    session = f"{time.time_ns()}"
    env = dict(
        session=session,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(torch.device(device)),
        previous_blocks=len(rows),
    )
    (out / f"environment_{session}.json").write_text(json.dumps(env, indent=2))
    transform = imagenet_transform(224)
    names = ["base", "student", "selected1+base", "early2", "early5", "independent2"]
    rng = np.random.default_rng(20261015)
    for seed in (42, 43, 44):
        decoded = [
            Image.fromarray(images[int(i)]) for i in arrays[seed]["val_ids"][:32]
        ]
        for bs in (1, 32):
            cpu = torch.stack([transform(im) for im in decoded[:bs]])
            for scope in ("model_only", "pipeline"):
                for block in range(blocks):
                    order = rng.permutation(names).tolist()
                    for name in order:
                        key = (seed, bs, scope, block, name)
                        if key in keys:
                            continue
                        plan = None
                        x = None
                        try:
                            gc.collect()
                            torch.cuda.empty_cache()
                            plan, passes = make_plan(
                                seed, name, sources, bundles, selections[seed], device
                            )
                            x = cpu.to(device) if scope == "model_only" else None

                            def call():
                                xx = (
                                    x
                                    if x is not None
                                    else torch.stack(
                                        [transform(im) for im in decoded[:bs]]
                                    ).to(device)
                                )
                                return plan(xx)

                            with torch.inference_mode():
                                for _ in range(warmup):
                                    result = call()
                                torch.cuda.synchronize(device)
                                torch.cuda.reset_peak_memory_stats(device)
                                times = []
                                for _ in range(repeats):
                                    torch.cuda.synchronize(device)
                                    began = time.perf_counter()
                                    result = call()
                                    torch.cuda.synchronize(device)
                                    times.append(time.perf_counter() - began)
                                elapsed = sum(times)
                                block_median = float(np.median(times))
                                if not torch.isfinite(result).all():
                                    raise FloatingPointError("Nonfinite runtime output")
                            row = dict(
                                seed=seed,
                                batch_size=bs,
                                scope=scope,
                                block=block,
                                method=name,
                                status="complete",
                                median_block_ms=block_median * 1000,
                                ms_per_image=block_median / bs * 1000,
                                images_per_second=bs * repeats / elapsed,
                                unique_backbone_passes=passes,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(
                                    device
                                ),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(
                                    device
                                ),
                                session=session,
                            )
                        except torch.cuda.OutOfMemoryError:
                            row = dict(
                                seed=seed,
                                batch_size=bs,
                                scope=scope,
                                block=block,
                                method=name,
                                status="oom",
                                session=session,
                            )
                        finally:
                            plan = None
                            x = None
                            if "result" in locals():
                                del result
                            gc.collect()
                            torch.cuda.empty_cache()
                        rows.append(row)
                        keys.add(key)
                        temp = progress.with_suffix(".tmp")
                        temp.write_text(json.dumps(rows, indent=2))
                        temp.replace(progress)
    compact = []
    for seed in (42, 43, 44):
        for bs in (1, 32):
            for scope in ("model_only", "pipeline"):
                for name in names:
                    sample = [
                        r
                        for r in rows
                        if (r["seed"], r["batch_size"], r["scope"], r["method"])
                        == (seed, bs, scope, name)
                    ]
                    valid = [r for r in sample if r["status"] == "complete"]
                    values = np.array([r["median_block_ms"] for r in valid])
                    compact.append(
                        dict(
                            seed=seed,
                            batch_size=bs,
                            scope=scope,
                            method=name,
                            blocks=len(valid),
                            oom_blocks=len(sample) - len(valid),
                            median_ms_per_batch=(
                                float(np.median(values)) if len(values) else None
                            ),
                            p10=(
                                float(np.quantile(values, 0.1)) if len(values) else None
                            ),
                            p90=(
                                float(np.quantile(values, 0.9)) if len(values) else None
                            ),
                        )
                    )
    (out / "tables").mkdir(exist_ok=True)
    write_csv(out / "tables/runtime.csv", compact)
    metrics = dict(
        status=(
            "complete"
            if all(r["blocks"] == blocks for r in compact)
            else "complete_with_oom"
        ),
        measurement_rows=len(rows),
        independent_hardware_sessions=len({r["session"] for r in rows}),
        timing_scope="synchronized block medians; p10/p90 across block medians, not pooled per-request quantiles",
    )
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                config=contract,
                environments=[p.name for p in out.glob("environment_*.json")],
            ),
            indent=2,
        )
    )
    (out / "report.md").write_text(
        "# Experiment Report: S2 runtime\n\n## Configuration\nSee config.yaml.\n\n## Dataset\n32 fixed noisy-validation images per base; no test access.\n\n## Checkpoint\nPinned original bases and head; duplicate base/snapshot weight states share a forward.\n\n## Metrics\nSee metrics.json.\n\n## Compact Results\nSee tables/runtime.csv.\n\n## Generated Artifacts\nPer-block samples and environment metadata.\n\n## Notes\nWarmup20,50 measurements per block,10 blocks. Block medians are not independent GPU sessions or per-request percentiles. Weights resident per method; startup excluded.\n"
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
