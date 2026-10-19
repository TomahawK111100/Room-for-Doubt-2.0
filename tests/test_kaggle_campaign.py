import json
from pathlib import Path
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tests.test_post_kaggle import source  # noqa: F401
from room_for_doubt.training.kaggle_base import base_split_seed, resume_contract
from room_for_doubt.evaluation.campaign_ensemble import evaluate_ensembles
from room_for_doubt.training.feature_heads import target_values

ROOT = Path(__file__).parents[1]


def test_fixed_data_split_independent_of_model_seed_and_window_targets():
    old = dict(seed=42, epochs=50)
    assert resume_contract(old) == resume_contract(dict(old, base_split_seed=None))
    assert base_split_seed(dict(seed=43, base_split_seed=42)) == 42
    assert base_split_seed(dict(seed=43)) == 43
    data = dict(targets=np.broadcast_to(np.arange(50), (2, 50)))
    for mode, expected in (
        ("early_mean_2", 0.5),
        ("early_mean_10", 4.5),
        ("early_mean_20", 9.5),
        ("late_mean_5", 47),
    ):
        assert np.all(target_values(data, None, mode, 5) == expected)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="feature_study_extended")
    assert (
        OmegaConf.is_struct(cfg)
        and len(cfg.target_modes) * len(cfg.encoders) * len(cfg.seeds) == 42
    )


def test_fixed_ensemble_members_and_aligned_ids(tmp_path):
    reference = tmp_path / "reference/cache"
    reference.mkdir(parents=True)
    labels, ids = np.array([0, 1, 0, 1]), np.arange(4)
    a = np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.2, 0.8]])
    np.save(reference / "test_ids.npy", ids)
    np.save(reference / "test_labels.npy", labels)
    np.save(reference / "test_probs.npy", a[:, None])
    new = tmp_path / "new/shared/cache"
    new.mkdir(parents=True)
    np.save(new / "test_ids.npy", ids)
    np.save(new / "test_labels.npy", labels)
    np.save(new / "test_probs.npy", a[:, None])
    output = tmp_path / "out"
    (output / "tables").mkdir(parents=True)
    (output / "artifacts").mkdir()
    jobs = [dict(dataset="cifar100n", seed=43, study_run=str(tmp_path / "new"))]
    rows = evaluate_ensembles(reference.parent, jobs, output)
    assert [r["base_seeds"] for r in rows] == ["42", "42/43"]
    assert all(r["gain_pp"] == 0 for r in rows)


def test_campaign_budget_failure_recovery_and_completed_job_reuse(
    source,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    import room_for_doubt.analysis.kaggle_campaign as campaign

    base, _ = source
    calls, fail_cifar10 = [], [True]

    def fake_train(cfg, out):
        calls.append((cfg.dataset.name, cfg.seed))
        assert cfg.base_split_seed == 42 and cfg.epochs == 50
        if cfg.dataset.name == "cifar10n" and fail_cifar10[0]:
            raise RuntimeError("Synthetic independent dataset failure")
        out.mkdir(parents=True, exist_ok=True)
        result = dict(status="complete", test_acc=0.7)
        (out / "metrics.json").write_text(json.dumps(result))
        return result

    def fake_study(cfg, out):
        assert cfg.post_run is None and cfg.scope == "standard"
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "tables/aggregate.csv").write_text("method,acc\nbase,0.7\n")
        result = dict(status="complete")
        (out / "metrics.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(campaign, "train_base", fake_train)
    monkeypatch.setattr(campaign, "run_study", fake_study)
    monkeypatch.setattr(campaign, "evaluate_ensembles", lambda *args: [])
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="kaggle_campaign")
    cfg.source_run, cfg.post_run, cfg.data_dir = (
        str(base),
        str(tmp_path / "post"),
        str(tmp_path / "data"),
    )
    cfg.min_free_gb, cfg.minimum_start_hours = 0, 9
    out = tmp_path / "campaign"
    paused = campaign.run_campaign(cfg, out)
    assert paused["status"] == "partial" and not calls
    cfg.minimum_start_hours = 2.5
    partial = campaign.run_campaign(cfg, out)
    assert [job["status"] for job in partial["jobs"]] == [
        "complete",
        "failed",
        "complete",
    ]
    fail_cifar10[0] = False
    complete = campaign.run_campaign(cfg, out)
    assert complete["status"] == "complete"
    assert calls.count(("cifar100n", 43)) == calls.count(("cifar100n", 44)) == 1
    assert calls.count(("cifar10n", 42)) == 2
    assert (out / "campaign_summary.zip").exists()


def test_replication_profiles_only_request_missing_seeds():
    for name, run_id, expected in [
        ('campaign_cifar100', 'replication_cifar100', [('cifar100n', 44)]),
        ('campaign_cifar10', 'replication_cifar10', [('cifar10n', 43), ('cifar10n', 44)]),
    ]:
        with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
            cfg = compose(config_name=name)
        assert cfg.run_id == run_id
        assert [(j.dataset, j.seed) for j in cfg.jobs] == expected
        assert cfg.base_split_seed == 42
        assert cfg.jobs[0].optional is False
