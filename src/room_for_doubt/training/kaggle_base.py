"""Resumable stronger-base pilot; observed-label selection, complete trajectories.

Uses PyTorch DataParallel for a single-process two-T4 run. Trajectories are
recorded once per example at its augmented training forward, before the update.
"""

import csv
import json
import math
import os
import random
import time
from pathlib import Path
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from room_for_doubt.data.cifar_noisy import get_train_val_indices, load_observed_labels
from room_for_doubt.data.animal10n import Animal10N, verify_identity
from room_for_doubt.data.meta_splits import reserve_meta_splits
from room_for_doubt.data.observed_images import ObservedImages, imagenet_transform
from room_for_doubt.models.robust_classifier import (
    make_pretrained_classifier,
    robust_loss,
    probability_margin,
)
from room_for_doubt.reporting.artifacts import (
    generate_report,
    save_metrics,
    save_run_metadata,
)
from room_for_doubt.utils.paths import config_directory


def atomic_save(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def record_margins(margins, seen, positions, values):
    positions = np.asarray(positions)
    if len(np.unique(positions)) != len(positions) or seen[positions].any():
        raise ValueError("Duplicate trajectory sample in epoch")
    margins[positions] = values
    seen[positions] = True


def base_split_seed(config):
    value = config.get("base_split_seed")
    return config["seed"] if value is None else int(value)


def resume_contract(config):
    config = dict(config, base_split_seed=base_split_seed(config))
    return {
        k: v
        for k, v in config.items()
        if k
        not in {
            "resume",
            "max_hours",
            "output_dir",
            "data_dir",
            "num_workers",
            "require_cuda",
        }
    }


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    probabilities, labels, ids = [], [], []
    total_loss = 0.0
    for x, y, sid in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        total_loss += torch.nn.functional.cross_entropy(
            logits.float(), y, reduction="sum"
        ).item()
        probabilities.append(logits.float().softmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
        ids.append(sid.numpy())
    p, y, sid = (
        np.concatenate(probabilities),
        np.concatenate(labels),
        np.concatenate(ids),
    )
    return dict(loss=total_loss / len(y), accuracy=float((p.argmax(1) == y).mean())), (
        p,
        y,
        sid,
    )


def train(cfg, out):
    config = OmegaConf.to_container(cfg, resolve=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    artifacts = out / "artifacts"
    artifacts.mkdir(exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    checkpoint = None
    if cfg.resume == "auto" and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    elif cfg.resume not in ("auto", None):
        checkpoint = torch.load(
            to_absolute_path(cfg.resume), map_location="cpu", weights_only=False
        )
    elif last_path.exists():
        raise ValueError("Run already exists; use resume=auto or a new run_id")
    if checkpoint is not None and resume_contract(
        checkpoint["config"]
    ) != resume_contract(config):
        raise ValueError(
            "Resume config differs from the saved scientific configuration"
        )
    if checkpoint is None and (out / "config.yaml").exists():
        raise ValueError(
            "Existing run has no resumable checkpoint; choose a new run_id"
        )
    if cfg.model.name != "resnet50" or cfg.epochs < 1 or not 0 < cfg.loss.q <= 1:
        raise ValueError("Invalid model, epoch count or GCE q")
    expected = {"cifar10n": 10, "cifar100n": 100, "animal10n": 10}
    if cfg.dataset.num_classes != expected[cfg.dataset.name]:
        raise ValueError("Dataset class count mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.require_cuda and device.type != "cuda":
        raise RuntimeError("Enable a CUDA GPU before running")
    if (
        cfg.get("expected_cuda_devices") is not None
        and torch.cuda.device_count() != cfg.expected_cuda_devices
    ):
        raise RuntimeError(
            "CUDA device count differs from this run's frozen base protocol"
        )
    amp = device.type == "cuda"
    torch.set_num_threads(4)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if amp:
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    cls = {"cifar10n": CIFAR10, "cifar100n": CIFAR100, "animal10n": Animal10N}[
        cfg.dataset.name
    ]
    root = to_absolute_path(cfg.data_dir)
    raw = cls(root, train=True, download=True)
    verify_identity(raw, config)
    noisy = load_observed_labels(root, cfg.dataset.name, cfg.dataset.noisy_split)
    train_ids, val_ids = get_train_val_indices(
        len(raw.data), cfg.dataset.val_ratio, base_split_seed(config)
    )
    np.savez_compressed(
        artifacts / "split_ids.npz", train_ids=train_ids, val_ids=val_ids
    )
    train_set = ObservedImages(
        raw.data,
        noisy,
        train_ids,
        imagenet_transform(cfg.image_size, True, cfg.dataset.get("native_size", 32)),
    )
    val_set = ObservedImages(
        raw.data, noisy, val_ids, imagenet_transform(cfg.image_size)
    )
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=amp,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    base = make_pretrained_classifier(
        cfg.dataset.num_classes, None if checkpoint else cfg.model.weights
    ).to(device)
    optimizer = torch.optim.SGD(
        base.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    model = torch.nn.DataParallel(base) if torch.cuda.device_count() > 1 else base
    history, start, best_loss, best_epoch = [], 0, float("inf"), None
    best_state = None
    targets = np.full((len(train_ids), cfg.epochs), np.nan, dtype=np.float32)
    if checkpoint is not None:
        base.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, start, best_loss, best_epoch = (
            checkpoint["history"],
            checkpoint["epoch"] + 1,
            checkpoint["best_val_loss"],
            checkpoint["best_epoch"],
        )
        if not np.array_equal(checkpoint["train_ids"], train_ids) or not np.array_equal(
            checkpoint["val_ids"], val_ids
        ):
            raise ValueError("Resume split mismatch")
        targets = checkpoint["targets"]
        best_state = checkpoint["best_state_dict"]
        # last.pt is the atomic epoch commit; recover even if best.pt was
        # overwritten just before a crash interrupted the next last.pt save.
        atomic_save(
            dict(
                state_dict=best_state,
                epoch=best_epoch,
                val_loss=best_loss,
                config=config,
            ),
            best_path,
        )
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if amp and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    id_to_row = np.full(len(noisy), -1, dtype=np.int64)
    id_to_row[train_ids] = np.arange(len(train_ids))
    info = dict(
        checkpoint_path=str(best_path),
        seed=cfg.seed,
        base_split_seed=base_split_seed(config),
        dataset_split=cfg.dataset.noisy_split,
        device=str(device),
        gpu_names=[
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        pretraining="ImageNet-1K; external supervised data",
        trajectory_definition="augmented train forward before update; observed-class probability margin",
        checkpoint_selection="minimum noisy-validation cross entropy",
        clean_training_labels_used=False,
    )
    (artifacts / "environment.json").write_text(
        json.dumps(
            dict(torch=torch.__version__, cuda=torch.version.cuda, **info), indent=2
        )
    )
    started = time.monotonic()
    for epoch in range(start, cfg.epochs):
        tick = time.monotonic()
        generator = torch.Generator().manual_seed(cfg.seed + epoch)
        train_loader = DataLoader(
            train_set, shuffle=True, generator=generator, **loader_kwargs
        )
        model.train()
        seen = np.zeros(len(train_ids), dtype=bool)
        total = 0.0
        lr = cfg.optimizer.lr * 0.5 * (1 + math.cos(math.pi * epoch / cfg.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr
        for x, y, sid in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
            loss = robust_loss(logits, y, cfg.loss.q, epoch < cfg.loss.ce_warmup_epochs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            values = probability_margin(logits.detach(), y).cpu().numpy()
            record_margins(targets[:, epoch], seen, id_to_row[sid.numpy()], values)
            total += loss.item() * len(y)
        if not seen.all():
            raise ValueError("Incomplete trajectory coverage")
        val, _ = evaluate(model, val_loader, device, amp)
        if not np.isfinite(val["loss"]):
            raise FloatingPointError("Nonfinite validation loss")
        row = dict(
            epoch=epoch,
            train_loss=total / len(train_ids),
            val_loss=val["loss"],
            val_noisy_acc=val["accuracy"],
            lr=lr,
            seconds=time.monotonic() - tick,
        )
        history.append(row)
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in base.state_dict().items()
            }
            atomic_save(
                dict(
                    state_dict=best_state,
                    epoch=epoch,
                    val_loss=best_loss,
                    config=config,
                ),
                best_path,
            )
        state = dict(
            state_dict=base.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_loss=best_loss,
            best_state_dict=best_state,
            config=config,
            history=history,
            targets=targets,
            train_ids=train_ids,
            val_ids=val_ids,
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if amp else None,
        )
        atomic_save(state, last_path)
        with (out / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        metrics = dict(
            status="training",
            completed_epochs=epoch + 1,
            best_epoch=best_epoch,
            best_val_loss=best_loss,
        )
        save_metrics(out, metrics)
        save_run_metadata(
            out,
            cfg.run_id,
            config,
            **info,
            best_epoch=best_epoch,
            best_val_loss=best_loss,
        )
        generate_report(
            out,
            cfg.run_id,
            config,
            metrics,
            info,
            "Training in progress. No test evaluation yet. This is an ImageNet-pretrained base-model experiment, not an isolated improvement from dynamics distillation.",
        )
        eta = (
            (cfg.epochs - epoch - 1)
            * np.mean([r["seconds"] for r in history[-3:]])
            / 3600
        )
        print(
            f"Epoch {epoch + 1}/{cfg.epochs}: noisy-val CE={val['loss']:.4f}, noisy-val acc={100 * val['accuracy']:.2f}%, best epoch={best_epoch + 1}, ETA≈{eta:.1f}h",
            flush=True,
        )
        if time.monotonic() - started > cfg.max_hours * 3600 and epoch + 1 < cfg.epochs:
            metrics["status"] = "paused_time_budget"
            save_metrics(out, metrics)
            generate_report(
                out,
                cfg.run_id,
                config,
                metrics,
                info,
                "Time budget reached after an epoch; best.pt and last.pt saved. Resume this same output directory to finish. Test was not evaluated.",
            )
            return metrics
    if not np.isfinite(targets).all():
        raise ValueError("Cannot export incomplete trajectories")
    splits = reserve_meta_splits(train_ids, cfg.split_seed)
    np.savez_compressed(
        artifacts / "meta_targets.npz",
        sample_ids=train_ids,
        targets=targets,
        noisy_labels=noisy[train_ids],
        epochs=np.arange(cfg.epochs),
        base_train_ids=train_ids,
        base_val_ids=val_ids,
        excluded_ids=np.array([], dtype=np.int64),
        dataset_name=np.array(cfg.dataset.name),
        noisy_split=np.array(cfg.dataset.noisy_split),
        split_seed=np.array(cfg.split_seed),
        **splits,
    )
    best = torch.load(best_path, map_location=device, weights_only=False)
    base.load_state_dict(best["state_dict"])
    val, arrays = evaluate(model, val_loader, device, amp)
    np.savez_compressed(
        artifacts / "val_predictions.npz",
        probabilities=arrays[0],
        labels=arrays[1],
        ids=arrays[2],
    )
    # Clean test is first instantiated after the training/selection loop.
    test_raw = cls(root, train=False, download=True)
    test = ObservedImages(
        test_raw.data,
        test_raw.targets,
        np.arange(len(test_raw.data)),
        imagenet_transform(cfg.image_size),
    )
    test_metrics, arrays = evaluate(
        model, DataLoader(test, shuffle=False, **loader_kwargs), device, amp
    )
    np.savez_compressed(
        artifacts / "test_predictions.npz",
        probabilities=arrays[0],
        labels=arrays[1],
        ids=arrays[2],
    )
    metrics = dict(
        status="complete",
        completed_epochs=cfg.epochs,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        val_noisy_acc=val["accuracy"],
        test_acc=test_metrics["accuracy"],
        test_loss=test_metrics["loss"],
        trajectory_samples=len(train_ids),
        trajectory_epochs=cfg.epochs,
    )
    if cfg.dataset.name != "animal10n":
        metrics["target_70_reached"] = test_metrics["accuracy"] >= 0.70
    save_metrics(out, metrics)
    save_run_metadata(
        out, cfg.run_id, config, **info, best_epoch=best_epoch, best_val_loss=best_loss
    )
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        dict(info, best_epoch=best_epoch, best_val_loss=best_loss),
        "One base seed. ImageNet-pretrained ResNet50 with CE warmup then GCE; minimum noisy-validation CE selects checkpoint. Clean training labels unused. Train-time margins include augmentation and changing minibatch weights; use the saved epoch axis. No accuracy threshold defines completion. New checkpoint/preprocessing are not compatible with the historical ResNet34 evaluation loader. Meta heads must be retrained for this base.",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


@hydra.main(
    config_path=str(config_directory()), config_name="kaggle_base", version_base=None
)
def main(cfg):
    train(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
