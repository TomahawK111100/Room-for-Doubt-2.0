import json
import numpy as np
import torch
from omegaconf import OmegaConf


def test_shared_cache_bounds_expanded_views_and_reuses_completed_cache(tmp_path, monkeypatch):
    import room_for_doubt.inference.post_kaggle_cache as cache

    batches = []

    class Base(torch.nn.Module):
        def forward(self, x):
            batches.append(len(x))
            return torch.zeros(len(x), 2)

    class Encoder(torch.nn.Module):
        def extract_features(self, x):
            return torch.zeros(len(x), 512)

    monkeypatch.setattr(cache, "restore_base", lambda *a: Base())
    monkeypatch.setattr(cache, "frozen_encoder", lambda *a: Encoder())
    monkeypatch.setattr(cache, "imagenet_transform", lambda *a: lambda im: torch.zeros(3, 8, 8))
    monkeypatch.setattr(cache, "meta_transform", lambda: lambda im: torch.zeros(3, 8, 8))
    monkeypatch.setattr(cache, "make_views", lambda im, count, seed: [im] * count)
    (tmp_path / "features").mkdir()
    torch.save({}, tmp_path / "features/backbone.pt")
    cfg = OmegaConf.create(dict(views=10, image_batch_size=64, inference_seed=123))
    bundle = dict(checkpoint_digest="test", config=dict(image_size=8, dataset=dict(num_classes=2)))
    images = np.zeros((14, 8, 8, 3), dtype=np.uint8)
    ids, labels = np.arange(14), np.zeros(14, dtype=int)
    args = (bundle, images, ids, labels, "val", tmp_path / "cache", tmp_path / "features", cfg, "cpu")
    cache.split_cache(*args)
    assert batches == [60, 60, 20]
    assert np.load(tmp_path / "cache/val_features.npy").shape == (14, 10, 512)
    cache.split_cache(*args)
    assert batches == [60, 60, 20]


def test_campaign_stage_uses_worker_process(tmp_path, monkeypatch):
    import room_for_doubt.analysis.kaggle_campaign as campaign

    calls = []

    def worker(command, check):
        calls.append(command)
        assert check
        assert OmegaConf.load(command[-2]).seed == 43
        (tmp_path / "metrics.json").write_text(json.dumps(dict(status="complete")))

    monkeypatch.setattr(campaign.subprocess, "run", worker)
    assert campaign.train_base(OmegaConf.create(dict(seed=43)), tmp_path)["status"] == "complete"
    assert calls[0][1:4] == ["-m", "room_for_doubt.analysis.campaign_stage", "base"]
