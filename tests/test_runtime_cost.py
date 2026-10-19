import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from tests.test_followup_controls import complete_features, source  # noqa: F401
from tests.test_post_kaggle import TinyBase
from room_for_doubt.evaluation.runtime_cost import measure
from room_for_doubt.evaluation.selection import weighted_consensus
from room_for_doubt.inference.runtime_tta import FrozenViewPredictor
from room_for_doubt.models.feature_margin import FeatureMarginHead

ROOT = Path(__file__).parents[1]


def test_runtime_matches_cached_formula_and_cleans_up_feature_hooks():
    torch.manual_seed(7)
    base = TinyBase(classes=3)
    with torch.no_grad():
        base.fc.weight.copy_(torch.randn_like(base.fc.weight))
    head = FeatureMarginHead(3, 2)
    projection = dict(
        mean=np.array([0.1, 0.2, 0.3], dtype="float32"),
        components=np.eye(3, dtype="float32")[:2],
    )
    runtime = FrozenViewPredictor(base, head, projection).eval()
    x = torch.randn(4, 2, 3, 8, 8)
    with torch.inference_mode():
        flat = x.flatten(0, 1)
        p = base(flat).softmax(-1)
        features = (flat.mean((2, 3)) - runtime.center) @ runtime.components.T
        badness = (
            (1 - head.predict_from_features(features, p.argmax(-1)).mean(-1)) / 2
        ).clamp(0, 1)
        expected = weighted_consensus(
            p.numpy().reshape(4, 2, 3), badness.numpy().reshape(4, 2), 0.05
        )
    actual = runtime(x, "early", 0.05).numpy()
    assert np.allclose(actual, expected, atol=1e-6)
    assert not base.fc._forward_pre_hooks
    uniform = FrozenViewPredictor(base).eval()
    assert torch.allclose(uniform(x, "uniform"), p.reshape(4, 2, 3).mean(1))
    assert torch.allclose(uniform(x, "early", "uniform"), uniform(x, "uniform"))
    with pytest.raises(ValueError, match="fitted head"):
        uniform(x, "early", 0.05)
    with pytest.raises(ValueError, match="one view"):
        uniform(x, "base")
    calls = []
    result, values = measure(lambda: calls.append(1), torch.device("cpu"), 2, 4, 2)
    assert len(calls) == 6 and len(values) == 4
    assert result["peak_cuda_allocated_mb"] is None
    assert result["amortized_ms_per_image"] == result["batch_ms_median"] / 2


def test_runtime_driver_never_accesses_clean_labels(
    complete_features,  # noqa: F811
    tmp_path,
    monkeypatch,
):  # noqa: F811
    import room_for_doubt.analysis.runtime_cost as runner

    base, noisy, study = complete_features

    class ObservedOnly:
        def __init__(self, root, train, download):
            assert train
            self.data = np.random.default_rng(8).integers(
                0, 255, (40, 32, 32, 3), dtype="uint8"
            )

        @property
        def targets(self):
            raise AssertionError("Timing does not need clean labels")

    monkeypatch.setattr(runner, "CIFAR100", ObservedOnly)
    monkeypatch.setattr(runner, "load_observed_labels", lambda *args: noisy)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="runtime_cost")
    cfg.source_run, cfg.study_run, cfg.device = str(base), str(study), "cpu"
    cfg.batch_images, cfg.view_counts, cfg.warmup, cfg.repetitions = [1], [1, 2], 1, 2
    out = tmp_path / "cost"
    result = runner.run_benchmark(cfg, out)
    assert result["configurations"] == 10 and result["clean_test_accessed"] is False
    rows = list(csv.DictReader((out / "tables/runtime_cost.csv").open()))
    assert {r["scope"] for r in rows} == {"model_only", "with_preprocessing"}
    assert all(r["peak_cuda_allocated_mb"] == "" for r in rows)
    assert (out / "config.yaml").exists() and (out / "report.md").exists()
    with pytest.raises(ValueError, match="new run_id"):
        runner.run_benchmark(cfg, out)
