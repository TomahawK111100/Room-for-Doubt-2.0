"""Recovered inputs preserve sample identity and all frozen results."""

import json
import numpy as np
import pytest
from omegaconf import OmegaConf

from room_for_doubt.data.feature_artifacts import shared_split
from room_for_doubt.data.native_cache_recovery import recover_native_cache


def fixture(study):
    (study / "native_features").mkdir(parents=True)
    (study / "artifacts").mkdir()
    (study / "tables").mkdir()
    cfg = OmegaConf.create(dict(
        base_digest="sha", inference_seed=123, view_counts=[1, 2],
        dataset=dict(num_classes=2),
    ))
    OmegaConf.save(cfg, study / "config.yaml")
    (study / "metrics.json").write_text(json.dumps(dict(status="complete")))
    policy = dict(method="tta", views=2, temperature="uniform", val_noisy_acc=1.)
    frozen = dict(base_digest="sha", selected=dict(tta=policy))
    (study / "artifacts/policies_before_test.json").write_text(json.dumps(frozen))
    (study / "tables/test_validation_selected.csv").write_text("method,acc\nbase,1\ntta,1\n")
    probs = np.array([[[.8, .2], [.6, .4]], [[.1, .9], [.2, .8]]], dtype=np.float32)
    for split in ("val", "test"):
        marker = dict(base_digest="sha", split=split, ids=[3, 8], views=2,
                      inference_seed=123, image_size=224, precision="fp32")
        source = study / "native_features"
        (source / f"{split}_complete.json").write_text(json.dumps(marker))
        (source / f"{split}_alignment.json").write_text(json.dumps(
            dict(prediction_disagreements=0, max_probability_difference=1e-7)))
        np.save(source / f"{split}_probs.npy", probs)
    return cfg, dict(val=np.array([0, 1]), test=np.array([0, 1]))


def test_recovery_readable_without_copying_arrays(tmp_path):
    study, out = tmp_path / "study", tmp_path / "recovered"
    cfg, labels = fixture(study)
    recover_native_cache(study, out, "sha", labels)
    p, y, ids = shared_split(study, out, cfg, "test")
    assert p.shape == (2, 2, 2)
    assert ids.tolist() == [3, 8] and y.tolist() == [0, 1]
    assert (out / "cache/test_probs.npy").is_symlink()
    assert not json.loads((out / "recovery_provenance.json").read_text())["bit_identical_to_original_shared_cache"]
    recover_native_cache(study, out, "sha", labels)


def test_failed_accuracy_does_not_publish_recovery(tmp_path):
    study, out = tmp_path / "study", tmp_path / "recovered"
    _, labels = fixture(study)
    labels["val"] = np.array([1, 0])
    with pytest.raises(ValueError, match="Frozen val accuracy changed"):
        recover_native_cache(study, out, "sha", labels)
    assert not (out / "cache/test_complete.json").exists()


def test_different_original_predictions_rejected(tmp_path):
    study, out = tmp_path / "study", tmp_path / "recovered"
    _, labels = fixture(study)
    (study / "native_features/test_alignment.json").write_text(json.dumps(
        dict(prediction_disagreements=1, max_probability_difference=.0001)))
    with pytest.raises(ValueError, match="identity/alignment"):
        recover_native_cache(study, out, "sha", labels)
    assert not out.exists()


def test_wrong_base_rejected(tmp_path):
    study = tmp_path / "study"
    _, labels = fixture(study)
    with pytest.raises(ValueError, match="base identity"):
        recover_native_cache(study, tmp_path / "recovered", "wrong", labels)
