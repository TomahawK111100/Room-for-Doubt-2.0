"""Read-only replay of three completed Animal Outputs; never fit or reselect a method.

Replays saved validation grids to verify existing choices, all scalar heads from
saved projected features, and selected metrics/paired intervals from raw caches.
Only a fixed eight-validation-image base forward check is performed; full base
feature extraction, PCA fitting and an independent semantic audit are out of scope.
"""

import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from room_for_doubt.data.animal10n import Animal10N
from room_for_doubt.data.feature_artifacts import shared_split
from room_for_doubt.data.cifar_noisy import get_train_val_indices
from room_for_doubt.data.meta_splits import reserve_meta_splits
from room_for_doubt.data.observed_images import imagenet_transform
from room_for_doubt.evaluation.diagnostic_metrics import (
    probability_metrics,
    detection_metrics,
    static_badness,
)
from room_for_doubt.evaluation.selection import (
    select_temperature,
    weighted_consensus,
    paired_accuracy_interval,
)
from room_for_doubt.evaluation.paired_targets import paired_auroc_interval
from room_for_doubt.models.feature_margin import FeatureMarginHead
from room_for_doubt.models.robust_classifier import make_pretrained_classifier
from room_for_doubt.reporting.artifacts import (
    generate_report,
    save_metrics,
    save_run_metadata,
    write_csv,
)

BASES = {
    42: "f160a3c58149324ba6935c985def7d4b53ae3d973aacbf61d811229b8db7e8e9",
    43: "4f0b19b228570796929192b3787db1484737d91b3607cc84e8cd421934449ae3",
    44: "508fd8f3e39bcb3a5ab33767dc88cb556925a83d68da8f87e3ed2700c33e7f85",
}
DATA = {
    "data_batch_1.bin": "c8409fa2b1b60f3b896afbb590d75e3975214da2ee1151977f885d50340ae123",
    "test_batch.bin": "006a389eeed34154c15b19b129ce41a906ed5cf5f6348b30ba3a3d139270fe97",
    "source_provenance.json": "b5e5b5b456b2cc8c9ba0c4dd464bdef29ff6e91e994273823235e7eb430cc868",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def complete_rows(rows, expected, key):
    actual = [key(row) for row in rows]
    require(
        len(actual) == len(expected) and set(actual) == set(expected),
        "Missing, duplicate or unexpected result rows",
    )


def scientific_config(root, cfg):
    protocol_path = Path(__file__).resolve().parents[3] / "configs/animal10n_pilot.yaml"
    require(
        (root / "artifacts/frozen_protocol.yaml").read_bytes()
        == protocol_path.read_bytes(),
        "Frozen protocol snapshot differs",
    )
    protocol = OmegaConf.load(protocol_path)
    for key in [
        "model",
        "optimizer",
        "loss",
        "epochs",
        "batch_size",
        "base_split_seed",
        "split_seed",
        "image_size",
        "expected_cuda_devices",
    ]:
        require(cfg[key] == protocol.base[key], "Frozen base config differs " + key)
    for key in ["name", "noisy_split", "num_classes", "val_ratio", "native_size"]:
        require(
            cfg.dataset[key] == protocol.base.dataset[key],
            "Dataset config differs " + key,
        )
    return protocol


def checkpoint(path, epochs):
    """Best files contain tensors and primitive config only; no unrestricted pickle."""
    saved = torch.load(
        path / "checkpoints/best.pt", map_location="cpu", weights_only=True
    )
    history = csv_rows(path / "history.csv")
    require(
        [int(r["epoch"]) for r in history] == list(range(epochs)),
        "Incomplete ordered checkpoint history",
    )
    losses = np.array([float(r["val_loss"]) for r in history])
    require(np.isfinite(losses).all(), "Nonfinite checkpoint losses")
    best = int(losses.argmin())
    require(
        saved["epoch"] == best
        and np.isclose(saved["val_loss"], losses[best], atol=1e-10, rtol=0),
        "Checkpoint is not minimum validation loss",
    )
    require(
        all(torch.isfinite(t).all().item() for t in saved["state_dict"].values()),
        "Nonfinite checkpoint weights",
    )
    return saved


def compare_numeric(saved, actual, keys, tolerance=1e-8):
    for key in keys:
        require(key in saved and key in actual, "Missing metric " + key)
        require(
            np.isclose(float(saved[key]), float(actual[key]), atol=tolerance, rtol=0),
            f"Metric mismatch {key}: saved={saved[key]}, replay={actual[key]}",
        )


def validation_replay(cfg, probabilities, labels, scores, frozen):
    """Reproduce the saved grid on noisy val, without choosing any NEW policy."""
    selected = {}
    for count in cfg.view_counts:
        p = probabilities[:, :count]
        rows = [
            dict(
                method="tta",
                views=int(count),
                temperature="uniform",
                val_noisy_acc=float((p.mean(1).argmax(1) == labels).mean()),
            )
        ]
        for name, score in scores.items():
            temperature, sweep = select_temperature(
                p, score[:, :count], labels, cfg.temperatures
            )
            best = next(r for r in sweep if r["temperature"] == temperature)
            rows.append(dict(method=name, views=int(count), **best))
        for row in rows:
            name = row["method"]
            if (
                name not in selected
                or row["val_noisy_acc"] > selected[name]["val_noisy_acc"]
            ):
                selected[name] = row
    require(
        selected == frozen["selected"],
        "Saved policies do not replay from noisy-validation caches",
    )
    return selected


def consensus(p, scores, policy):
    q = p[:, : policy["views"]]
    return (
        q.mean(1)
        if policy["temperature"] == "uniform"
        else weighted_consensus(
            q, scores[policy["method"]][:, : policy["views"]], policy["temperature"]
        )
    )


def replay_head(
    run, cfg, encoder, mode, seed, features, probabilities, saved_scores, deadline
):
    saved = checkpoint(run, int(cfg.epochs))
    local = saved["config"]
    for key, value in dict(
        base_digest=cfg.base_digest,
        encoder=encoder,
        target_mode=mode,
        seed=seed,
        feature_dim=cfg.feature_dim,
        pca_seed=cfg.pca_seed,
        early_epochs=cfg.early_epochs,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        split_seed=cfg.split_seed,
    ).items():
        require(local.get(key) == value, "Head config mismatch: " + key)
    if mode == "best_augmented_mean":
        for key in ["augmentation_views", "augmentation_seed", "static_target_digest"]:
            require(local.get(key) == cfg[key], "Augmented head config mismatch " + key)
    model = FeatureMarginHead(int(cfg.dataset.num_classes), int(cfg.feature_dim)).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    maximum = 0.0
    with torch.inference_mode():
        for split in ["val", "test"]:
            f, p = features[split], probabilities[split]
            require(
                f.shape == (*p.shape[:2], cfg.feature_dim),
                "Projected feature dimensions differ",
            )
            flat, classes = f.reshape(-1, cfg.feature_dim), p.argmax(-1).reshape(-1)
            score = np.empty(len(flat), dtype=np.float32)
            for start in range(0, len(flat), 1024):
                require(time.monotonic() < deadline, "CPU audit time limit reached")
                x = torch.from_numpy(np.array(flat[start : start + 1024], copy=True))
                y = torch.from_numpy(classes[start : start + 1024].astype(np.int64))
                trajectory = model.predict_from_features(x, y).numpy()
                score[start : start + 1024] = np.clip(
                    (1 - trajectory.mean(1)) / 2, 0, 1
                )
            score = score.reshape(p.shape[:2])
            expected = saved_scores[split]
            require(
                score.shape == expected.shape and np.isfinite(expected).all(),
                "Invalid saved view score",
            )
            delta = float(np.abs(score - expected).max())
            maximum = max(maximum, delta)
            require(delta <= 1e-5, f"CPU head score mismatch {split}: maxabs={delta}")
    del model, saved
    return maximum


def locate(input_root, seeds):
    root = Path(input_root)
    result, missing = {}, []
    for seed in seeds:
        candidates = [
            p.parent
            for p in root.rglob("metrics.json")
            if p.parent.name == f"rfd_animal10n_seed{seed}"
            and read(p).get("base_digest") == BASES[seed]
        ]
        if len(candidates) != 1:
            missing.append(
                f"seed{seed}: expected one full Output, found {len(candidates)}"
            )
            continue
        run = candidates[0]
        required = [
            "base/checkpoints/best.pt",
            "base/artifacts/meta_targets.npz",
            "base/artifacts/split_ids.npz",
        ]
        for split in ["val", "test"]:
            required += [
                f"feature_study/shared/cache/{split}_{name}.npy"
                for name in ["probs", "labels", "ids"]
            ]
            for encoder in ["resnet18", "base_resnet50"]:
                required += [f"feature_study/projected/{encoder}/{split}.npy"]
        for stage, encoders, modes in [
            (
                "feature_study",
                ["resnet18", "base_resnet50"],
                ["early_mean", "history_mean", "best_margin"],
            ),
            ("augmentation", ["base_resnet50"], ["best_augmented_mean"]),
        ]:
            for encoder in encoders:
                for mode in modes:
                    for head in [42, 43, 44]:
                        required += [
                            f"{stage}/heads/{encoder}_{mode}_{head}/checkpoints/best.pt"
                        ]
                        required += [
                            f"{stage}/artifacts/{split}_{encoder}__{mode}__{head}.npy"
                            for split in ["val", "test"]
                        ]
        absent = [name for name in required if not (run / name).is_file()]
        if absent:
            missing.append(dict(seed=seed, missing_files=absent))
        else:
            result[seed] = run
    data = [
        p.parent
        for p in root.rglob("source_provenance.json")
        if all((p.parent / name).is_file() for name in DATA)
    ]
    if len(data) != 1:
        missing.append(
            f"Animal dataset: expected one binary data folder, found {len(data)}"
        )
    return result, data[0] if len(data) == 1 else None, missing


def audit_base(root, seed, data, out, deadline):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(root / "base/config.yaml")
    require(
        read(root / "metrics.json")["status"] == "complete", "Source root is incomplete"
    )
    require(
        cfg.seed == seed and cfg.dataset.name == "animal10n",
        "Wrong source seed or dataset",
    )
    require(
        sha(root / "base/checkpoints/best.pt") == BASES[seed],
        "Base checkpoint SHA256 differs",
    )
    train = Animal10N(data, train=True)
    test = Animal10N(data, train=False)
    protocol = scientific_config(root, cfg)
    saved = checkpoint(root / "base", 50)
    for key in [
        "seed",
        "dataset",
        "model",
        "optimizer",
        "loss",
        "epochs",
        "batch_size",
        "base_split_seed",
        "split_seed",
        "image_size",
    ]:
        require(
            saved["config"][key] == OmegaConf.to_container(cfg, resolve=True)[key],
            "Base checkpoint config mismatch " + key,
        )
    with np.load(root / "base/artifacts/meta_targets.npz") as archive:
        targets = {key: archive[key] for key in archive.files}
    train_ids, val_ids = get_train_val_indices(
        50000, cfg.dataset.val_ratio, cfg.base_split_seed
    )
    require(
        np.array_equal(targets["sample_ids"], train_ids)
        and np.array_equal(targets["base_val_ids"], val_ids)
        and np.array_equal(targets["base_train_ids"], train_ids),
        "Base train/val identities differ",
    )
    require(
        np.array_equal(targets["noisy_labels"], train.targets[train_ids]),
        "Trajectory observed labels differ",
    )
    require(
        targets["targets"].shape == (45000, 50)
        and np.isfinite(targets["targets"]).all()
        and np.max(np.abs(targets["targets"])) <= 1.00001,
        "Trajectory dimensions or range differ",
    )
    require(
        np.array_equal(targets["epochs"], np.arange(50))
        and len(targets["excluded_ids"]) == 0,
        "Trajectory epochs/exclusions differ",
    )
    for key, value in reserve_meta_splits(train_ids, cfg.split_seed).items():
        require(np.array_equal(targets[key], value), "Meta split differs " + key)
    with np.load(root / "base/artifacts/split_ids.npz") as ids:
        require(
            np.array_equal(ids["train_ids"], train_ids)
            and np.array_equal(ids["val_ids"], val_ids),
            "Saved base split differs",
        )
    study = root / "feature_study"
    study_cfg = OmegaConf.load(study / "config.yaml")
    probability, labels = {}, {}
    for split, reference_ids, reference_labels in [
        ("val", val_ids, train.targets[val_ids]),
        ("test", np.arange(5000), test.targets),
    ]:
        p, y, ids = shared_split(study, study / "shared", study_cfg, split)
        require(
            np.array_equal(ids, reference_ids) and np.array_equal(y, reference_labels),
            "Cache IDs/labels differ from supplied dataset " + split,
        )
        require(
            p.shape == (5000, 10, 10)
            and np.isfinite(p).all()
            and np.all(p >= 0)
            and np.all(p <= 1)
            and np.allclose(p.sum(-1), 1, atol=1e-5),
            "Invalid cached probabilities",
        )
        probability[split], labels[split] = p, y
    # Fixed validation-only smoke; not a full re-extraction of base features.
    model = make_pretrained_classifier(10, weights=None).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    transform = imagenet_transform(cfg.image_size)
    with torch.inference_mode():
        images = torch.stack(
            [transform(Image.fromarray(train.data[int(i)])) for i in val_ids[:8]]
        )
        replay = model(images).softmax(-1).numpy()
    base_error = float(np.abs(replay - probability["val"][:8, 0]).max())
    require(base_error <= 1e-4, f"Base8-validation-image replay mismatch: {base_error}")
    del model, saved, images, targets
    results, head_rows = [], []
    for stage in ["feature_study", "augmentation"]:
        folder = root / stage
        scfg = OmegaConf.load(folder / "config.yaml")
        frozen = read(folder / "artifacts/policies_before_test.json")
        require(
            read(folder / "metrics.json")["status"] == "complete"
            and frozen["base_digest"] == BASES[seed]
            and scfg.base_digest == BASES[seed],
            "Study status/base identity differs",
        )
        require(time.monotonic() < deadline, "CPU audit time limit reached")
        for key in [
            "epochs",
            "batch_size",
            "learning_rate",
            "meta_model",
            "view_counts",
            "temperatures",
            "feature_dim",
            "pca_seed",
            "early_epochs",
            "inference_seed",
            "bootstrap_seed",
            "split_seed",
            "seeds",
            "optimizer",
        ]:
            require(
                scfg[key] == protocol.study[key],
                "Frozen study grid/config differs " + key,
            )
        expected_encoders = (
            list(protocol.study.encoders)
            if stage == "feature_study"
            else ["base_resnet50"]
        )
        expected_modes = list(protocol.study.target_modes) + (
            [] if stage == "feature_study" else ["best_augmented_mean"]
        )
        require(
            list(scfg.encoders) == expected_encoders
            and list(scfg.target_modes) == expected_modes,
            "Frozen encoder/target scope differs",
        )
        if stage == "augmentation":
            for key in ["augmentation_views", "augmentation_seed"]:
                require(
                    scfg[key] == protocol.augmentation[key],
                    "Frozen augmentation config differs " + key,
                )
        scores = {
            split: {
                name: static_badness(probability[split], name)
                for name in ["msp", "gap", "entropy"]
            }
            for split in ["val", "test"]
        }
        for encoder in scfg.encoders:
            branch = study / "projected" / encoder
            marker = read(branch / "projection.json")
            with np.load(root / "base/artifacts/meta_targets.npz") as meta:
                lookup = {int(s): i for i, s in enumerate(meta["sample_ids"])}
                fit = [lookup[int(s)] for s in meta["meta_train_ids"]]
            require(
                marker["fit_rows"] == fit
                and marker["dimensions"] == 512
                and marker["seed"] == scfg.pca_seed,
                "PCA fit rows/dimensions differ",
            )
            features = {
                split: np.load(branch / f"{split}.npy", mmap_mode="r")
                for split in ["val", "test"]
            }
            for mode in scfg.target_modes:
                for head_seed in scfg.seeds:
                    name = f"{encoder}__{mode}__{head_seed}"
                    cached = {
                        split: np.load(folder / "artifacts" / f"{split}_{name}.npy")
                        for split in ["val", "test"]
                    }
                    for split in ["val", "test"]:
                        require(
                            cached[split].shape == (5000, 10)
                            and np.isfinite(cached[split]).all(),
                            "Invalid head score cache " + name,
                        )
                        scores[split][name] = cached[split]
                    head = folder / "heads" / f"{encoder}_{mode}_{head_seed}"
                    if stage == "augmentation" and mode != "best_augmented_mean":
                        original = study / "heads" / head.name / "checkpoints/best.pt"
                        require(
                            sha(original) == sha(head / "checkpoints/best.pt"),
                            "Reused head weight differs",
                        )
                        for split in ["val", "test"]:
                            require(
                                np.array_equal(
                                    cached[split],
                                    np.load(
                                        study / "artifacts" / f"{split}_{name}.npy"
                                    ),
                                ),
                                "Reused head score differs",
                            )
                    else:
                        error = replay_head(
                            head,
                            scfg,
                            encoder,
                            mode,
                            int(head_seed),
                            features,
                            probability,
                            cached,
                            deadline,
                        )
                        head_rows.append(
                            dict(stage=stage, method=name, score_max_abs_error=error)
                        )
            del features
        validation_replay(
            scfg, probability["val"], labels["val"], scores["val"], frozen
        )
        baseline = probability["test"][:, 0]
        events = baseline.argmax(1) != labels["test"]
        tta = consensus(probability["test"], scores["test"], frozen["selected"]["tta"])
        saved_rows = csv_rows(folder / "tables/test_validation_selected.csv")
        complete_rows(
            saved_rows, set(frozen["selected"]) | {"base"}, lambda row: row["method"]
        )
        predictions = {}
        for row in saved_rows:
            name = row["method"]
            policy = (
                dict(method="base", views=1, temperature="uniform")
                if name == "base"
                else frozen["selected"][name]
            )
            p = (
                baseline
                if name == "base"
                else consensus(probability["test"], scores["test"], policy)
            )
            score = scores["test"].get(name, scores["test"]["msp"])[:, 0]
            metrics = probability_metrics(labels["test"], p)
            metrics.update(
                {
                    "base_error_" + key: value
                    for key, value in detection_metrics(events, score).items()
                }
            )
            metrics.update(
                {
                    key + "_vs_tta_selected": value
                    for key, value in paired_accuracy_interval(
                        labels["test"],
                        tta.argmax(1),
                        p.argmax(1),
                        seed=scfg.bootstrap_seed,
                    ).items()
                }
            )
            compare_numeric(
                row,
                metrics,
                [
                    "acc",
                    "ece",
                    "nll",
                    "brier",
                    "base_error_auroc",
                    "base_error_auprc",
                    "gain_pp_vs_tta_selected",
                    "ci_low_pp_vs_tta_selected",
                    "ci_high_pp_vs_tta_selected",
                ],
            )
            require(
                int(row["views"]) == policy["views"]
                and str(row["temperature"]) == str(policy["temperature"]),
                "Selected table policy differs",
            )
            predictions[name] = p.argmax(1)
            results.append(dict(stage=stage, method=name, **metrics))
        pair_folder = root / "paired" if stage == "feature_study" else folder
        require(
            read(pair_folder / "metrics.json")["status"] == "complete",
            "Paired stage incomplete",
        )
        require(
            read(pair_folder / "artifacts/policies_before_test.json") == frozen,
            "Paired frozen policies differ",
        )
        pair_rows = csv_rows(pair_folder / "tables/paired_target_contrasts.csv")
        expected_pairs = {
            (encoder, int(head), mode)
            for encoder in scfg.encoders
            for head in scfg.seeds
            for mode in ["best_margin", "history_mean", "best_augmented_mean"]
            if mode in scfg.target_modes
        }
        complete_rows(
            pair_rows,
            expected_pairs,
            lambda row: (row["encoder"], int(row["head_seed"]), row["reference"]),
        )
        with np.load(
            pair_folder / "artifacts/paired_selected_predictions.npz",
            allow_pickle=False,
        ) as archive:
            require(
                np.array_equal(archive["ids"], np.arange(5000))
                and np.array_equal(archive["labels"], labels["test"]),
                "Paired prediction IDs/GT differ",
            )
            for name in frozen["selected"]:
                if "__" in name:
                    require(
                        np.array_equal(archive[name], predictions[name]),
                        "Stored paired prediction differs " + name,
                    )
        for row in pair_rows:
            require(time.monotonic() < deadline, "CPU audit time limit reached")
            require(
                row["proposed"] == "early_mean"
                and int(row["samples"]) == 5000
                and row["error_event"] == "original_view_base_error",
                "Paired comparison scope differs",
            )
            first = f"{row['encoder']}__{row['reference']}__{row['head_seed']}"
            second = f"{row['encoder']}__early_mean__{row['head_seed']}"
            require(
                int(row["reference_views"]) == frozen["selected"][first]["views"]
                and int(row["proposed_views"]) == frozen["selected"][second]["views"],
                "Paired view budgets differ",
            )
            actual = paired_accuracy_interval(
                labels["test"],
                predictions[first],
                predictions[second],
                scfg.bootstrap_seed,
            )
            actual.update(
                paired_auroc_interval(
                    events,
                    scores["test"][first][:, 0],
                    scores["test"][second][:, 0],
                    draws=2000,
                    seed=scfg.bootstrap_seed,
                )
            )
            compare_numeric(
                row,
                actual,
                [
                    "gain_pp",
                    "ci_low_pp",
                    "ci_high_pp",
                    "auroc_difference",
                    "auroc_ci_low",
                    "auroc_ci_high",
                ],
            )
    write_csv(out / "recomputed_selected.csv", results)
    write_csv(out / "head_replay.csv", head_rows)
    return dict(
        seed=seed,
        status="passed_replay_checks",
        heads_replayed=len(head_rows),
        selected_rows_checked=len(results),
        base_validation_smoke_images=8,
        base_probability_max_abs_error=base_error,
        all_test_images_metric_replay=5000,
        model_fitting_performed=False,
        full_base_feature_extraction_replayed=False,
    )


def run(input_root, output, max_minutes=45, inspect_only=False):
    """Always export an inspectable status, including missing inputs/failures."""
    output = Path(output).resolve()
    input_root = Path(input_root).resolve()
    require(
        not output.is_relative_to(input_root) and not input_root.is_relative_to(output),
        "Keep audit output separate from source mounts",
    )
    require(max_minutes > 0, "Positive CPU audit budget required")
    output.mkdir(parents=True, exist_ok=True)
    source_snapshot = output / "artifacts" / "animal10n_audit.py"
    source_snapshot.parent.mkdir(parents=True, exist_ok=True)
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    config = dict(
        dataset=dict(name="animal10n", noisy_split="human"),
        base_digests=BASES,
        input_root=str(input_root),
        max_minutes=max_minutes,
        inspect_only=inspect_only,
        base_forward_sample_size=8,
        head_replay_tolerance=1e-5,
        base_smoke_tolerance=1e-4,
        metric_tolerance=1e-8,
        bootstrap_draws=2000,
        evaluation_type="real_gt",
        source_provenance="Activeloop mirror, original KAIST archive equality unverified",
        no_new_policy_selection=True,
        audit_source_sha256=sha(source_snapshot),
    )
    OmegaConf.save(OmegaConf.create(config), output / "config.yaml", resolve=True)
    started = time.monotonic()
    state = dict(
        status="inspecting",
        runs=[],
        training_performed=False,
        source_files_modified=False,
        semantic_audit_acceptance="not_assessed_by_replay",
    )
    try:
        sources, data, missing = locate(input_root, [42, 43, 44])
        (output / "input_inventory.json").write_text(
            json.dumps(
                dict(
                    sources={str(k): str(v) for k, v in sources.items()},
                    data=str(data),
                    missing=missing,
                ),
                indent=2,
            )
            + "\n"
        )
        if missing:
            state["status"] = "blocked_missing_inputs"
            state["missing"] = missing
            return state
        for name, expected in DATA.items():
            require(sha(data / name) == expected, "Dataset checksum mismatch " + name)
        if inspect_only:
            state["status"] = "inputs_ready_not_replayed"
            return state
        for seed, source in sources.items():
            print("AUDIT seed", seed, flush=True)
            try:
                result = audit_base(
                    source,
                    seed,
                    data,
                    output / f"seed{seed}",
                    started + max_minutes * 60,
                )
            except Exception as exc:
                result = dict(
                    seed=seed,
                    status="failed_replay_check",
                    error=f"{type(exc).__name__}: {exc}",
                )
            state["runs"].append(result)
            (output / f"seed{seed}_status.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result), flush=True)
        state["status"] = (
            "complete_replay_checks"
            if all(r["status"] == "passed_replay_checks" for r in state["runs"])
            else "replay_discrepancy_or_runtime_failure"
        )
        return state
    except Exception as exc:
        state["status"] = "failed_input_or_runtime_check"
        state["error"] = f"{type(exc).__name__}: {exc}"
        return state
    finally:
        state["elapsed_seconds"] = time.monotonic() - started
        save_metrics(output, state)
        save_run_metadata(
            output,
            output.name,
            config,
            device="cpu",
            torch=torch.__version__,
            source_versions="identified by supplied base SHA256; no API downloads",
        )
        generate_report(
            output,
            output.name,
            config,
            state,
            notes="Read-only recorded-artifact replay, not a full semantic integrity verdict or a new experiment. Exact dataset bytes anchor cached labels; policy grid is replayed only on noisy validation and must match already saved choices. Scalar heads are replayed on all stored projected val/test features; base weights receive a fixed8-validation-image smoke only. Full base feature extraction and actual PCA fitting are not rerun. All compared intervals condition on fitted models; no between-base confidence interval is inferred. A replay failure is not success and needs inspection of its saved error.",
        )
        with zipfile.ZipFile(
            output / "animal10n_audit_summary.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for p in sorted(output.rglob("*")):
                if p.is_file() and p.suffix in {
                    ".json",
                    ".yaml",
                    ".csv",
                    ".md",
                    ".py",
                    ".txt",
                }:
                    archive.write(p, str(p.relative_to(output)))
        print(
            "Status:",
            state["status"],
            "Summary:",
            output / "animal10n_audit_summary.zip",
            flush=True,
        )
