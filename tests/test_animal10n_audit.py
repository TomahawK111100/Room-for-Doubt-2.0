"""Fragile replay/identity and failure boundaries, without downloading anything."""

import json
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from room_for_doubt.analysis.animal10n_audit import (
    checkpoint,
    compare_numeric,
    validation_replay,
    replay_head,
    run,
)
from room_for_doubt.models.feature_margin import FeatureMarginHead


def test_checkpoint_rejects_last_or_nonminimum_validation(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "history.csv").write_text("epoch,val_loss\n0,0.2\n1,0.1\n2,0.3\n")
    path = tmp_path / "checkpoints/best.pt"
    torch.save(
        dict(
            epoch=2, val_loss=0.3, state_dict={"weight": torch.tensor([1.0])}, config={}
        ),
        path,
    )
    with pytest.raises(ValueError, match="minimum"):
        checkpoint(tmp_path, 3)
    torch.save(
        dict(
            epoch=1, val_loss=0.1, state_dict={"weight": torch.tensor([1.0])}, config={}
        ),
        path,
    )
    assert checkpoint(tmp_path, 3)["epoch"] == 1


def test_validation_ties_require_saved_fewer_views_and_uniform():
    p = np.array([[[0.8, 0.2], [0.8, 0.2]], [[0.1, 0.9], [0.1, 0.9]]], dtype=np.float32)
    cfg = OmegaConf.create(dict(view_counts=[1, 2], temperatures=[0.1, 1.0]))
    policy = {
        name: dict(method=name, views=1, temperature="uniform", val_noisy_acc=1.0)
        for name in ["tta", "head"]
    }
    assert (
        validation_replay(
            cfg, p, np.array([0, 1]), {"head": np.ones((2, 2))}, {"selected": policy}
        )
        == policy
    )
    policy["head"]["views"] = 2
    with pytest.raises(ValueError, match="policies"):
        validation_replay(
            cfg, p, np.array([0, 1]), {"head": np.ones((2, 2))}, {"selected": policy}
        )


def test_metric_replay_rejects_changed_number():
    compare_numeric({"acc": 0.8}, {"acc": 0.8}, ["acc"])
    with pytest.raises(ValueError, match="mismatch"):
        compare_numeric({"acc": 0.8}, {"acc": 0.7}, ["acc"])


def test_full_head_replay_uses_class_conditioned_weights(tmp_path):
    torch.manual_seed(7)
    cfg = OmegaConf.create(
        dict(
            dataset=dict(num_classes=2),
            epochs=2,
            base_digest="abc",
            feature_dim=4,
            pca_seed=9,
            early_epochs=5,
            batch_size=8,
            learning_rate=0.001,
            split_seed=11,
        )
    )
    local = OmegaConf.to_container(cfg)
    local.update(encoder="base_resnet50", target_mode="early_mean", seed=42)
    model = FeatureMarginHead(2, 4).eval()
    features = {
        s: np.random.default_rng(i).normal(size=(3, 2, 4)).astype("float32")
        for i, s in enumerate(["val", "test"])
    }
    probabilities = {
        s: np.array([[[0.8, 0.2], [0.2, 0.8]]] * 3, dtype="float32") for s in features
    }
    scores = {}
    with torch.inference_mode():
        for s in features:
            x = torch.from_numpy(features[s].reshape(-1, 4))
            c = torch.from_numpy(probabilities[s].argmax(-1).reshape(-1))
            scores[s] = np.clip(
                (1 - model.predict_from_features(x, c).numpy().mean(1)) / 2, 0, 1
            ).reshape(3, 2)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "history.csv").write_text("epoch,val_loss\n0,0.2\n1,0.1\n")
    torch.save(
        dict(epoch=1, val_loss=0.1, state_dict=model.state_dict(), config=local),
        tmp_path / "checkpoints/best.pt",
    )
    args = (
        tmp_path,
        cfg,
        "base_resnet50",
        "early_mean",
        42,
        features,
        probabilities,
        scores,
        float("inf"),
    )
    assert replay_head(*args) < 1e-6
    scores["test"][1, 1] += 0.05
    with pytest.raises(ValueError, match="score mismatch"):
        replay_head(*args)


def test_missing_raw_outputs_never_claim_complete(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    out = tmp_path / "audit"
    result = run(inputs, out, inspect_only=True)
    assert result["status"] == "blocked_missing_inputs"
    assert len(result["missing"]) == 4
    assert (out / "animal10n_audit_summary.zip").is_file()
    assert json.loads((out / "metrics.json").read_text())["training_performed"] is False


def test_output_cannot_overlap_source(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        run(tmp_path, tmp_path / "audit", inspect_only=True)


def test_empty_and_truncated_result_tables_are_rejected():
    from room_for_doubt.analysis.animal10n_audit import complete_rows

    expected = {"base", "tta", "head"}
    complete_rows([dict(method=k) for k in expected], expected, lambda r: r["method"])
    for rows in [
        [],
        [dict(method="base")],
        [dict(method="base"), dict(method="tta"), dict(method="tta")],
    ]:
        with pytest.raises(ValueError, match="result rows"):
            complete_rows(rows, expected, lambda r: r["method"])
