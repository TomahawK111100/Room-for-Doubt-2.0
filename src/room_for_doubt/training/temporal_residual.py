"""Fit a small residual student; clean labels and base holdout never fit weights."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from room_for_doubt.data.temporal_evidence import load_evidence
from room_for_doubt.models.temporal_residual import (
    TemporalResidualHead,
    distillation_kl,
)
from room_for_doubt.evaluation.temporal_residual import select_alpha
from room_for_doubt.training.temporal_snapshots import file_digest
from room_for_doubt.training.kaggle_base import atomic_save
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
    write_csv,
)


class EvidenceRows(Dataset):
    def __init__(self, arrays, rows):
        self.arrays, self.rows = arrays, np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return tuple(
            torch.from_numpy(np.array(self.arrays[key][row], dtype=np.float32))
            for key in ("train_features", "train_probs", "teacher_probs")
        )


def feature_stats(features, rows):
    total = np.zeros(features.shape[1], dtype=np.float64)
    squares = total.copy()
    for start in range(0, len(rows), 256):
        part = np.asarray(features[rows[start : start + 256]], dtype=np.float64)
        total += part.sum(0)
        squares += np.square(part).sum(0)
    mean = total / len(rows)
    scale = np.sqrt(np.maximum(squares / len(rows) - mean**2, 0))
    return mean.astype(np.float32), np.maximum(scale, 1e-3).astype(np.float32)


@torch.no_grad()
def residuals(model, features, batch_size, device):
    model.eval()
    value = np.empty((len(features), model.net[-1].out_features), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(np.array(features[start : start + batch_size])).to(device)
        value[start : start + len(x)] = model(x).cpu().numpy()
    return value


@torch.no_grad()
def validation_kl(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for f, p, q in loader:
        f, p, q = f.to(device), p.to(device), q.to(device)
        total += float(distillation_kl(p, model(f), q)) * len(f)
        count += len(f)
    return total / count


def fit(cfg, directory, device):
    out = Path(directory)
    if (out / "checkpoints").exists() or (
        out / "artifacts/policy_before_test.json"
    ).exists():
        raise ValueError("Residual run already exists; choose a new run_id")
    if (
        cfg.epochs < 1
        or cfg.batch_size < 1
        or cfg.learning_rate <= 0
        or cfg.weight_decay < 0
    ):
        raise ValueError("Invalid residual training settings")
    manifest, arrays = load_evidence(cfg.cache_dir)
    if list(cfg.teacher_epochs) != [s["epoch"] for s in manifest["snapshots"]]:
        raise ValueError("Teacher epochs differ from sealed evidence")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "artifacts", "tables"):
        (out / name).mkdir(exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    mean, scale = feature_stats(arrays["train_features"], arrays["meta_train_rows"])
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    model = TemporalResidualHead(mean, scale, manifest["classes"], cfg.hidden_dim).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train = EvidenceRows(arrays, arrays["meta_train_rows"])
    validation = EvidenceRows(arrays, arrays["meta_val_rows"])
    val_loader = DataLoader(validation, batch_size=cfg.batch_size, shuffle=False)
    initial = validation_kl(model, val_loader, device)
    best, best_epoch, history = initial, 0, []
    checkpoint_path = out / "checkpoints/best.pt"
    contract = dict(
        cache_manifest_sha256=file_digest(Path(cfg.cache_dir) / "manifest.json"),
        base_digest=manifest["base_digest"],
        teacher_epochs=list(cfg.teacher_epochs),
        classes=manifest["classes"],
        hidden_dim=cfg.hidden_dim,
        feature_width=len(mean),
    )

    def save(epoch, loss):
        atomic_save(
            dict(
                state_dict={
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                epoch=epoch,
                val_loss=loss,
                config=config,
                contract=contract,
            ),
            checkpoint_path,
        )

    save(0, initial)  # The identity initialization may itself be the best KL fit.
    for epoch in range(1, cfg.epochs + 1):
        loader = DataLoader(
            train,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(cfg.seed + epoch),
        )
        model.train()
        total = 0.0
        for f, p, q in loader:
            f, p, q = f.to(device), p.to(device), q.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = distillation_kl(p, model(f), q)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite temporal distillation loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(f)
        val = validation_kl(model, val_loader, device)
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite meta-validation KL")
        history.append(dict(epoch=epoch, train_kl=total / len(train), meta_val_kl=val))
        if val < best:
            best, best_epoch = val, epoch
            save(epoch, val)
        # Keep a restart/audit artifact; this small head currently restarts with a new ID.
        atomic_save(
            dict(
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                config=config,
                contract=contract,
            ),
            out / "checkpoints/last.pt",
        )
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["state_dict"])
    h = residuals(model, arrays["val_features"], cfg.batch_size, device)
    policy, candidates = select_alpha(
        arrays["val_probs"], h, arrays["val_noisy_labels"], list(cfg.alphas)
    )
    # Diagnostic upper reference for this target, not an oracle using clean labels.
    teacher_residual = np.log(np.clip(arrays["val_teacher_probs"], 1e-12, 1)) - np.log(
        np.clip(arrays["val_probs"], 1e-12, 1)
    )
    teacher_policy, teacher_candidates = select_alpha(
        arrays["val_probs"],
        teacher_residual,
        arrays["val_noisy_labels"],
        list(cfg.alphas),
    )
    identity_acc = candidates[0]["val_noisy_acc"]
    teacher_gain = 100 * (teacher_policy["val_noisy_acc"] - identity_acc)
    student_gain = 100 * (policy["val_noisy_acc"] - identity_acc)
    diagnosis = (
        "no_noisy_validation_accuracy_complementarity"
        if teacher_gain <= 0
        else (
            "teacher_complementarity_not_transferred"
            if student_gain <= 0
            else "student_preserves_noisy_validation_accuracy_improvement"
        )
    )
    frozen = dict(
        schema=1,
        scope="temporal_residual_pilot",
        **contract,
        checkpoint_sha256=file_digest(checkpoint_path),
        policy=policy,
        teacher_policy=teacher_policy,
        selection="noisy-base-validation accuracy, then CE, then smaller alpha",
        test_accessed=False,
    )
    (out / "artifacts/policy_before_test.json").write_text(json.dumps(frozen, indent=2))
    write_csv(out / "tables/training.csv", history)
    write_csv(out / "tables/validation_policies.csv", candidates)
    write_csv(out / "tables/validation_teacher_interpolation.csv", teacher_candidates)
    metrics = dict(
        status="policy_frozen",
        initial_meta_val_kl=initial,
        best_meta_val_kl=best,
        best_epoch=best_epoch,
        selected_alpha=policy["alpha"],
        val_noisy_acc=policy["val_noisy_acc"],
        teacher_val_noisy_acc=teacher_policy["val_noisy_acc"],
        teacher_selected_alpha=teacher_policy["alpha"],
        teacher_val_gain_pp=teacher_gain,
        student_val_gain_pp=student_gain,
        pilot_diagnosis=diagnosis,
        test_evaluated=False,
    )
    info = dict(
        checkpoint_path=str(checkpoint_path),
        best_epoch=best_epoch,
        best_val_loss=best,
        device=str(device),
        **contract,
        label_usage="head KL uses teacher probabilities only; alpha uses noisy base holdout",
        policy_sha256=file_digest(out / "artifacts/policy_before_test.json"),
    )
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **info)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        info,
        "Experimental compressed snapshot mean. No clean labels, no test evaluation. "
        "Same meta partitions as scalar protocol; unused detector partitions stay reserved. "
        "Initial zero head is checkpoint epoch0. No novelty or benchmark gain established.",
    )
    return metrics


def load_frozen(directory, cache_directory, device):
    root = Path(directory)
    policy = json.loads((root / "artifacts/policy_before_test.json").read_text())
    manifest, _ = load_evidence(cache_directory)
    metadata = json.loads((root / "run_metadata.json").read_text())
    checkpoint = root / "checkpoints/best.pt"
    if (
        policy.get("scope") != "temporal_residual_pilot"
        or policy.get("schema") != 1
        or policy["checkpoint_sha256"] != file_digest(checkpoint)
        or policy["cache_manifest_sha256"]
        != file_digest(Path(cache_directory) / "manifest.json")
        or policy["base_digest"] != manifest["base_digest"]
        or metadata["policy_sha256"]
        != file_digest(root / "artifacts/policy_before_test.json")
    ):
        raise ValueError("Frozen policy/checkpoint/cache identity mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key, value in saved["contract"].items():
        if policy[key] != value:
            raise ValueError("Frozen policy/checkpoint scientific contract differs")
    state = saved["state_dict"]
    head = TemporalResidualHead(
        state["mean"], state["scale"], policy["classes"], policy["hidden_dim"]
    )
    head.load_state_dict(state, strict=True)
    return head.to(device).eval(), policy
