"""Read-only, pinned full-output preflight for the S2 post-hoc controls."""

import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from room_for_doubt.data.temporal_evidence import load_evidence, probabilities
from room_for_doubt.data.kaggle_artifacts import load_completed_run
from room_for_doubt.training.temporal_snapshots import file_digest, verify_snapshots

PINNED = {
    42: (
        "528f45f2d36efabed062716b0f4e6690d396ca4370dba26497cb4ea5ebff0a6a",
        "1c45838acbdd07d101195e4a51e11e47982bced8c02282b3b9242d8538c89aad",
        "ed159cfa8ba0a9ee3f421851e14044b7bad03be132befdbe12e20c3c5a3bdc05",
    ),
    43: (
        "a666fad014059bad8c3f206b9035c1a3377110ea23bf2f40fdcd4e35470c95f8",
        "98b112d4633ccd06cb089e030e2721736153cc03e2e52f266031dffe03596e20",
        "32704a02dd73b788813aefd19184f8efc9a81baf76a32b9054db265ee15c93a3",
    ),
    44: (
        "95d61ea33484eab3784f44a2b6b7341ea794f8212fd6b485be665006dafef656",
        "3702cca52cafc5ea1e84b287d271fe29c11efef67a9a095667fc87d7421e0fea",
        "2f35a68b094e04a106487bf7a4838435289dcfbdf809e48edba5ddcded271966",
    ),
}

POLICY_DIGESTS = {
    42: "e1dd54ce3d1fd630d8b7fb7d67f9f9f92fdfad4e1854787332b1dfef47354f57",
    43: "4cf312dba6e20f39ea7df27b2360e328952e658600f48f0405a0b9dd379d7c23",
    44: "9881bc9e22d0f2290e50bf915e6dbd77da249837e8aeaa45dd397cc974d1dab5",
}
REFERENCE_DIGESTS = {
    42: "860d39995e280275838b1830c39241c0963cdeb816abb7009c72a57d51d37433",
    43: "fea296ee41a3033b20d99ed7d532469b7c26546e6517ef089ec7eaf8c0d22416",
    44: "a4290f2a8ef5c82a5df053f73771f3b8dc2c8550e9cc9b58f55e9c656c478e4f",
}


def aligned_rows(source_ids, requested_ids):
    source_ids, requested_ids = np.asarray(source_ids), np.asarray(requested_ids)
    for ids in (source_ids, requested_ids):
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Unique integer sample IDs required")
    if len(source_ids) != len(requested_ids) or set(source_ids.tolist()) != set(
        requested_ids.tolist()
    ):
        raise ValueError("Donor/recipient sample membership differs")
    lookup = {int(sid): i for i, sid in enumerate(source_ids)}
    return np.array([lookup[int(sid)] for sid in requested_ids], dtype=np.int64)


def mean_probabilities(arrays, rows=None):
    """Bounded float64 accumulation; never stack all donor arrays."""
    if not arrays:
        raise ValueError("At least one teacher required")
    shape = arrays[0].shape
    result = np.empty(shape, dtype=np.float32)
    if rows is None:
        rows = [np.arange(shape[0]) for _ in arrays]
    if len(rows) != len(arrays):
        raise ValueError("One row alignment per teacher required")
    for a, r in zip(arrays, rows):
        probabilities(a)
        if a.shape != shape or np.asarray(r).shape != (shape[0],):
            raise ValueError("Teacher class/row dimensions differ")
    for start in range(0, shape[0], 1024):
        stop = min(start + 1024, shape[0])
        total = np.zeros((stop - start, shape[1]), dtype=np.float64)
        for a, r in zip(arrays, rows):
            total += a[r[start:stop]]
        q = (total / len(arrays)).astype(np.float32)
        q /= q.sum(1, keepdims=True)
        result[start:stop] = q
    probabilities(result)
    return result


def head_name(seed):
    return (
        "temporal_residual_seed42_v1" if seed == 42 else f"temporal_base{seed}_head42"
    )


def inspect_inputs(input_root):
    """CPU-only: verify sealed files/weights, never open clean test predictions."""
    inputs = Path(input_root)
    reports, sources = [], {}
    prediction_paths = list(inputs.rglob("predictions.npz")) if inputs.exists() else []
    for seed, pinned in PINNED.items():
        candidates = []
        errors = []
        for base in sorted(inputs.rglob(f"temporal_cifar100n_seed{seed}_v1")):
            if not base.is_dir() or base.parent.name != "bases":
                continue
            root = base.parent.parent
            head = root / "heads" / head_name(seed)
            for cache_name in ("evidence", "evidence_temporal"):
                cache = root / cache_name
                if not cache.is_dir():
                    continue
                required = [
                    base / n
                    for n in (
                        "config.yaml",
                        "metrics.json",
                        "history.csv",
                        "artifacts/meta_targets.npz",
                        "artifacts/split_ids.npz",
                        "checkpoints/best.pt",
                        "checkpoints/temporal/manifest.json",
                    )
                ]
                required += [
                    base / f"checkpoints/temporal/epoch_{e:03d}.pt" for e in range(1, 6)
                ]
                required += [
                    head / "artifacts/policy_before_test.json",
                    head / "checkpoints/best.pt",
                    cache / "manifest.json",
                ]
                required += [
                    cache / f"{s}_epoch{e:03d}_probs.npy"
                    for s in ("train", "val")
                    for e in range(1, 6)
                ]
                missing = [str(p) for p in required if not p.is_file()]
                if missing:
                    errors.append(dict(base=str(base), missing_files=missing))
                    continue
                try:
                    cfg = OmegaConf.to_container(
                        OmegaConf.load(base / "config.yaml"), resolve=True
                    )
                    policy = json.loads(
                        (head / "artifacts/policy_before_test.json").read_text()
                    )
                    found = (
                        file_digest(base / "checkpoints/best.pt"),
                        file_digest(cache / "manifest.json"),
                        file_digest(head / "checkpoints/best.pt"),
                    )
                    if (
                        file_digest(head / "artifacts/policy_before_test.json")
                        != POLICY_DIGESTS[seed]
                    ):
                        raise ValueError("Original frozen policy changed")
                    if (
                        found != pinned
                        or policy["base_digest"] != found[0]
                        or policy["cache_manifest_sha256"] != found[1]
                        or policy["checkpoint_sha256"] != found[2]
                    ):
                        raise ValueError(
                            "Source is not the pinned original temporal base/cache/head"
                        )
                    if (
                        cfg["seed"],
                        cfg["base_split_seed"],
                        cfg["split_seed"],
                        cfg["dataset"]["name"],
                        cfg["dataset"]["noisy_split"],
                        cfg["image_size"],
                    ) != (seed, 42, 20261015, "cifar100n", "noisy100", 224):
                        raise ValueError("Source protocol differs")
                    manifest, a = load_evidence(cache)
                    snapshots = verify_snapshots(base, cfg, [1, 2, 3, 4, 5])
                    if manifest["snapshots"] != [
                        {k: s[k] for k in ("epoch", "sha256")} for s in snapshots
                    ]:
                        raise ValueError("Teacher/cache identity mismatch")
                    for split, target in (
                        ("train", "teacher_probs"),
                        ("val", "val_teacher_probs"),
                    ):
                        qs = [
                            np.load(
                                cache / f"{split}_epoch{e:03d}_probs.npy",
                                mmap_mode="r",
                                allow_pickle=False,
                            )
                            for e in range(1, 6)
                        ]
                        if not np.array_equal(mean_probabilities(qs), a[target]):
                            raise ValueError(
                                "Per-epoch probabilities do not reconstruct original sealed mean"
                            )
                    refs = [
                        p
                        for p in prediction_paths
                        if p.parent.name == "final_evaluation"
                        and p.parent.parent.name == head_name(seed)
                    ]
                    if not refs:
                        raise ValueError(
                            "Missing original temporal final_evaluation/predictions.npz; mount frozen-test42 Output as well as full bases"
                        )
                    # Equal duplicate mounts accepted only when bytes are identical.
                    if {file_digest(p) for p in refs} != {REFERENCE_DIGESTS[seed]}:
                        raise ValueError("Conflicting original prediction exports")
                    candidates.append(
                        dict(
                            seed=seed,
                            root=root,
                            base=base,
                            head=head,
                            cache=cache,
                            config=cfg,
                            reference=refs[0],
                            manifest=manifest,
                        )
                    )
                except (ValueError, KeyError, OSError, RuntimeError) as exc:
                    errors.append(dict(base=str(base), reason=str(exc)))
        if candidates:
            # Every candidate has matched the same pinned base/cache/head identities.
            sources[seed] = candidates[0]
            reports.append(
                dict(
                    seed=seed,
                    status="verified_files",
                    base=str(candidates[0]["base"]),
                    cache=str(candidates[0]["cache"]),
                    reference=str(candidates[0]["reference"]),
                    identical_mounts=len(candidates),
                    rejected_candidates=errors,
                )
            )
        else:
            reports.append(
                dict(
                    seed=seed,
                    status="missing_input",
                    candidates=errors,
                    expected_run=f"temporal_cifar100n_seed{seed}_v1",
                    required="Full original base, early-five evidence/head and original final predictions; summaries do not contain weights or training evidence",
                )
            )
    return (
        dict(
            status="verified_files" if len(sources) == 3 else "missing_input",
            studies=reports,
            test_predictions_opened=False,
        ),
        sources,
    )


def verify_data(sources, noisy):
    """Bind all donor labels/splits to official observed labels before fitting."""
    baseline = None
    for seed, s in sources.items():
        b = load_completed_run(s["base"], noisy)
        manifest, a = load_evidence(s["cache"])
        if not np.array_equal(
            b["data"]["base_train_ids"], a["train_ids"]
        ) or not np.array_equal(b["data"]["base_val_ids"], a["val_ids"]):
            raise ValueError("Evidence/base sample IDs differ")
        if not np.array_equal(noisy[a["val_ids"]], a["val_noisy_labels"]):
            raise ValueError("Noisy holdout labels differ")
        if baseline is not None:
            for name in ("train_ids", "val_ids"):
                aligned_rows(baseline[name], a[name])
            if (
                manifest["class_axis"] != list(range(100))
                or manifest["image_size"] != 224
            ):
                raise ValueError("Class axis or transforms differ")
        baseline = a
        del b
    return True
