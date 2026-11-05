"""Resumable S2 heads on immutable evidence, with frozen policy contracts."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from room_for_doubt.models.temporal_residual import (
    TemporalResidualHead,
    distillation_kl,
)
from room_for_doubt.training.temporal_residual import feature_stats, residuals
from room_for_doubt.training.temporal_snapshots import file_digest
from room_for_doubt.training.kaggle_base import atomic_save
from room_for_doubt.evaluation.temporal_residual import select_alpha
from room_for_doubt.reporting.artifacts import write_csv


class TargetRows(Dataset):
    def __init__(self, arrays, q, rows):
        self.arrays, self.q, self.rows = arrays, q, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return tuple(
            torch.from_numpy(np.array(x[r], dtype=np.float32))
            for x in (self.arrays["train_features"], self.arrays["train_probs"], self.q)
        )


def input_base(p, parametrization):
    if parametrization == "residual":
        return p
    if parametrization != "direct":
        raise ValueError("Unknown head parametrization")
    return torch.full_like(p, 1 / p.shape[1])


def head_residual(model, features, base, parametrization, batch_size, device):
    h = residuals(model, features, batch_size, device)
    return h if parametrization == "residual" else h - np.log(np.clip(base, 1e-12, 1))


@torch.no_grad()
def loss_value(model, loader, device, parametrization):
    model.eval()
    total, n = 0.0, 0
    for f, p, q in loader:
        f, p, q = f.to(device), p.to(device), q.to(device)
        total += float(
            distillation_kl(input_base(p, parametrization), model(f), q)
        ) * len(f)
        n += len(f)
    return total / n


def load_head(root, device, expected_contract=None):
    root = Path(root)
    policy_path = root / "artifacts/policy_before_test.json"
    policy = json.loads(policy_path.read_text())
    if expected_contract is not None and policy["contract"] != expected_contract:
        raise ValueError("Frozen head belongs to another target/protocol")
    if file_digest(root / "checkpoints/best.pt") != policy["checkpoint_sha256"]:
        raise ValueError("Frozen head checkpoint changed")
    saved = torch.load(
        root / "checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    if saved["contract"] != policy["contract"]:
        raise ValueError("Head contract differs from policy")
    state = saved["state_dict"]
    head = TemporalResidualHead(
        state["mean"],
        state["scale"],
        policy["contract"]["classes"],
        policy["contract"]["hidden_dim"],
    )
    head.load_state_dict(state, strict=True)
    return head.to(device).eval(), policy


def fit_head(
    out, arrays, q_train, q_val, contract, device, *, epochs=25, batch_size=256
):
    out = Path(out)
    for name in ("artifacts", "checkpoints", "tables"):
        (out / name).mkdir(parents=True, exist_ok=True)
    if (out / "artifacts/policy_before_test.json").exists():
        _, policy = load_head(out, device, contract)
        return policy
    cfg = dict(
        run_id=out.name,
        contract=contract,
        epochs=epochs,
        batch_size=batch_size,
        optimizer="AdamW",
        learning_rate=0.001,
        weight_decay=0.0001,
        seed=42,
        checkpoint="min meta-val KL, including epoch0",
        test_evaluation=False,
    )
    config_path = out / "config.yaml"
    if (
        config_path.exists()
        and OmegaConf.to_container(OmegaConf.load(config_path), resolve=True) != cfg
    ):
        raise ValueError("Resume config differs")
    OmegaConf.save(OmegaConf.create(cfg), config_path)
    mean, scale = feature_stats(arrays["train_features"], arrays["meta_train_rows"])
    torch.manual_seed(42)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(42)
    head = TemporalResidualHead(
        mean, scale, contract["classes"], contract["hidden_dim"]
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
    train = TargetRows(arrays, q_train, arrays["meta_train_rows"])
    val = DataLoader(
        TargetRows(arrays, q_train, arrays["meta_val_rows"]), batch_size=batch_size
    )
    param = contract["parametrization"]
    initial = loss_value(head, val, device, param)
    best, best_epoch, start, history, elapsed = initial, 0, 1, [], 0.0
    best_path, last_path = out / "checkpoints/best.pt", out / "checkpoints/last.pt"

    def snapshot(epoch, loss):
        return dict(
            state_dict={
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            },
            contract=contract,
            epoch=epoch,
            val_loss=loss,
        )

    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("Resume head contract differs")
        head.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        best, best_epoch, history, elapsed = (
            saved["best"],
            saved["best_epoch"],
            saved["history"],
            saved["elapsed"],
        )
        start = saved["epoch"] + 1
        if not best_path.exists():
            raise ValueError("Resume best checkpoint missing")
    else:
        # An interrupted first epoch restarts from the same deterministic identity/uniform init.
        atomic_save(snapshot(0, initial), best_path)
    for epoch in range(start, epochs + 1):
        began = time.perf_counter()
        loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(42 + epoch),
        )
        head.train()
        total = 0.0
        for f, p, q in loader:
            f, p, q = f.to(device), p.to(device), q.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = distillation_kl(input_base(p, param), head(f), q)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite S2 head loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(f)
        value = loss_value(head, val, device, param)
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite S2 meta-validation loss")
        if value < best:
            best, best_epoch = value, epoch
            atomic_save(snapshot(epoch, value), best_path)
        seconds = time.perf_counter() - began
        elapsed += seconds
        history.append(
            dict(
                epoch=epoch,
                train_kl=total / len(train),
                meta_val_kl=value,
                seconds=seconds,
            )
        )
        last = snapshot(epoch, value)
        last.update(
            optimizer=optimizer.state_dict(),
            best=best,
            best_epoch=best_epoch,
            history=history,
            elapsed=elapsed,
        )
        atomic_save(last, last_path)
        write_csv(out / "tables/training.csv", history)
    head.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["state_dict"]
    )
    h = head_residual(
        head, arrays["val_features"], arrays["val_probs"], param, batch_size, device
    )
    selected, grid = select_alpha(
        arrays["val_probs"], h, arrays["val_noisy_labels"], [0.0, 0.25, 0.5, 1.0]
    )
    teacher_policy = None
    teacher_grid = []
    if contract["branch"] != "supervised_head":
        th = np.log(np.clip(q_val, 1e-12, 1)) - np.log(
            np.clip(arrays["val_probs"], 1e-12, 1)
        )
        teacher_policy, teacher_grid = select_alpha(
            arrays["val_probs"], th, arrays["val_noisy_labels"], [0.0, 0.25, 0.5, 1.0]
        )
    policy = dict(
        schema=1,
        scope="posthoc_controls_s2_v1",
        contract=contract,
        checkpoint_sha256=file_digest(best_path),
        policy=selected,
        teacher_policy=teacher_policy,
        test_accessed=False,
    )
    write_csv(out / "tables/validation_policies.csv", grid)
    if teacher_grid:
        write_csv(out / "tables/validation_teacher_interpolation.csv", teacher_grid)
    metrics = dict(
        status="policy_frozen",
        initial_meta_val_kl=initial,
        best_meta_val_kl=best,
        best_epoch=best_epoch,
        fit_seconds=elapsed,
        selected_alpha=selected["alpha"],
        test_evaluated=False,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                config=cfg,
                device=str(device),
                feature_normalization="meta-fit rows only",
                label_usage="observed noisy labels only for supervised branch; teacher probabilities otherwise",
            ),
            indent=2,
        )
    )
    (out / "report.md").write_text(
        f"# Experiment Report: {out.name}\n\n## Configuration\nSee config.yaml.\n\n## Dataset\nCIFAR-100N noisy100; shared frozen meta partitions.\n\n## Checkpoint\nBest epoch {best_epoch}, meta-val KL {best:.6g}.\n\n## Metrics\nSee metrics.json.\n\n## Compact Results\nSee validation tables.\n\n## Generated Artifacts\nCheckpoints, frozen policy and selection grids.\n\n## Notes\nExploratory S2 control; no test accessed during fitting.\n"
    )
    # Commit policy only after all required fit outputs are inspectable.
    (out / "artifacts/policy_before_test.json").write_text(json.dumps(policy, indent=2))
    return policy
