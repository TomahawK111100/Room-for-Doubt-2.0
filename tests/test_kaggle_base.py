import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from pathlib import Path
from room_for_doubt.models.robust_classifier import robust_loss, probability_margin
from room_for_doubt.training.kaggle_base import record_margins, train


def test_robust_loss_and_margin():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    assert torch.allclose(
        robust_loss(logits, labels, 0.7, True),
        torch.nn.functional.cross_entropy(logits, labels),
    )
    expected = ((1 - logits.softmax(1)[range(2), labels] ** 0.7) / 0.7).mean()
    assert torch.allclose(robust_loss(logits, labels, 0.7), expected)
    robust_loss(logits, labels, 0.7).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(
        probability_margin(logits, labels), torch.full((2,), 0.761594), atol=1e-6
    )


def test_trajectory_coverage_rejects_duplicates():
    values = np.full(3, np.nan)
    seen = np.zeros(3, dtype=bool)
    record_margins(values, seen, [2, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="Duplicate"):
        record_margins(values, seen, [2], [0.3])
    record_margins(values, seen, [1], [0.4])
    assert seen.all() and np.isfinite(values).all()


def test_training_resume_selects_noisy_val_and_exports_complete_targets(
    tmp_path, monkeypatch
):
    import room_for_doubt.training.kaggle_base as module

    events = []

    class FakeCIFAR:
        def __init__(self, root, train, download):
            events.append("train_data" if train else "test_data")
            self.data = np.random.default_rng(5).integers(
                0, 255, (40 if train else 8, 32, 32, 3), dtype=np.uint8
            )
            # Training clean labels are deliberately unusable.
            self.targets = None if train else [0] * 8

    monkeypatch.setattr(module, "CIFAR100", FakeCIFAR)
    monkeypatch.setattr(
        module, "load_observed_labels", lambda *args: np.arange(40) % 100
    )
    monkeypatch.setattr(
        module,
        "make_pretrained_classifier",
        lambda *args: torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 100)
        ),
    )
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/kaggle_base.yaml")
    cfg.epochs = 2
    cfg.image_size = 8
    cfg.batch_size = 16
    cfg.num_workers = 0
    cfg.require_cuda = False
    cfg.max_hours = -1
    # Force the offline test to CPU even on a CUDA runner.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    out = tmp_path / "run"
    first = train(cfg, out)
    assert first["status"] == "paused_time_budget" and "test_data" not in events
    (out / "checkpoints/best.pt").unlink()  # last.pt must recover the best model.
    cfg.max_hours = 10
    result = train(cfg, out)
    assert result["status"] == "complete" and events.count("test_data") == 1
    with np.load(out / "artifacts/meta_targets.npz") as data:
        assert data["targets"].shape == (36, 2)
        assert np.isfinite(data["targets"]).all()
        assert not set(data["base_train_ids"]) & set(data["base_val_ids"])
    last = torch.load(out / "checkpoints/last.pt", weights_only=False)
    best = torch.load(out / "checkpoints/best.pt", weights_only=False)
    assert best["val_loss"] == min(row["val_loss"] for row in last["history"])
    cfg.optimizer.lr = 0.002
    with pytest.raises(ValueError, match="Resume config differs"):
        train(cfg, out)
