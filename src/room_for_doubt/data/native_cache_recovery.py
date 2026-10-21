"""Recover shared probability inputs from verified, already saved native caches."""

import csv
import json
from pathlib import Path

import numpy as np

from room_for_doubt.data.feature_artifacts import completed_study
from room_for_doubt.evaluation.feature_study import weighted
from room_for_doubt.evaluation.diagnostic_metrics import static_badness


def recover_native_cache(study, destination, expected_digest, labels_by_split):
    """No inference, fitting or policy selection; reject changed saved accuracies.

    Native FP32 probabilities can have small numerical differences from the
    original shared cache. Preserve this fact and require the original recorded
    alignment to have zero per-view prediction disagreements, plus reproduced
    frozen-policy test accuracies. This is not a bit-identical cache claim.
    """
    study, destination = Path(study), Path(destination)
    cfg, frozen = completed_study(study)
    if cfg.base_digest != expected_digest:
        raise ValueError("Native recovery base identity differs")
    prepared, checks = [], []
    for split in ("val", "test"):
        source = study / "native_features"
        marker = json.loads((source / f"{split}_complete.json").read_text())
        alignment = json.loads((source / f"{split}_alignment.json").read_text())
        ids = np.asarray(marker["ids"], dtype=np.int64)
        labels = np.asarray(labels_by_split[split], dtype=np.int64)
        p = np.load(source / f"{split}_probs.npy", mmap_mode="r")
        if (marker["base_digest"] != expected_digest
                or marker["split"] != split or marker["precision"] != "fp32"
                or marker["inference_seed"] != cfg.inference_seed
                or marker["views"] != max(cfg.view_counts)
                or alignment["prediction_disagreements"] != 0
                or not np.isfinite(alignment["max_probability_difference"])
                or alignment["max_probability_difference"] > 1e-3
                or len(np.unique(ids)) != len(ids) or labels.shape != ids.shape
                or p.shape != (len(ids), marker["views"], cfg.dataset.num_classes)):
            raise ValueError("Native cache identity/alignment differs: " + split)
        for start in range(0, len(ids), 256):
            block = p[start:start + 256]
            if (not np.isfinite(block).all() or np.any(block < 0)
                    or np.any(block > 1)
                    or not np.allclose(block.sum(-1), 1, atol=1e-5)):
                raise ValueError("Invalid native probabilities")
        prepared.append((split, source, marker, ids, labels, p))

    # This verifies the frozen results, never searches a policy or tunes a head.
    for split, _, _, _, labels, p in prepared:
        scores = {name: static_badness(p, name) for name in ("msp", "gap", "entropy")}
        for name in frozen["selected"]:
            if "__" in name:
                path = study / "artifacts" / f"{split}_{name}.npy"
                if not path.exists():
                    raise ValueError("Saved scores needed for numerical verification: " + str(path))
                scores[name] = np.load(path, mmap_mode="r")
                if scores[name].shape != p.shape[:2] or not np.isfinite(scores[name]).all():
                    raise ValueError("Invalid saved native-recovery scores")
        if split == "test":
            with (study / "tables/test_validation_selected.csv").open() as stream:
                expected = {row["method"]: float(row["acc"]) for row in csv.DictReader(stream)}
            if set(expected) != {"base", *frozen["selected"]}:
                raise ValueError("Frozen result table methods differ")
        for name, policy in {"base": None, **frozen["selected"]}.items():
            if name == "base" and split == "val":
                continue
            predicted = p[:, 0] if name == "base" else weighted(p, scores, policy)
            actual = float((predicted.argmax(-1) == labels).mean())
            reference = expected[name] if split == "test" else policy["val_noisy_acc"]
            if not np.isclose(actual, reference, atol=1e-10, rtol=0):
                raise ValueError(f"Frozen {split} accuracy changed for {name}: {actual} != {reference}")
            checks.append(dict(split=split, method=name, acc=actual))

    cache = destination / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    provenance = dict(source_study=str(study.resolve()), base_digest=expected_digest,
                      operation="reuse_native_fp32_probabilities", fitting=False,
                      inference=False, bit_identical_to_original_shared_cache=False,
                      frozen_accuracy_checks=checks)
    record = destination / "recovery_provenance.json"
    if record.exists() and json.loads(record.read_text()) != provenance:
        raise ValueError("Existing cache recovery provenance differs")
    for split, source, marker, ids, labels, _ in prepared:
        target = cache / f"{split}_probs.npy"
        original = (source / f"{split}_probs.npy").resolve()
        if target.exists() and target.resolve() != original:
            raise ValueError("Refuse to replace unrelated recovered probabilities")
        if not target.exists():
            target.symlink_to(original)
        np.save(cache / f"{split}_ids.npy", ids)
        np.save(cache / f"{split}_labels.npy", labels)
        spec = dict(base_digest=expected_digest, seed=cfg.inference_seed,
                    views=marker["views"], image_size=marker["image_size"],
                    ids=ids.tolist(), recovery_source="native_fp32_probabilities")
        (cache / f"{split}_complete.json").write_text(json.dumps(spec))
    record.write_text(json.dumps(provenance, indent=2))
    return destination
