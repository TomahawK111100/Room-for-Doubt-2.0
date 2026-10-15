"""Cache frozen features once; train heads with held-out noisy trajectory targets."""

from pathlib import Path
import csv
import time
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from room_for_doubt.data.meta_splits import validate_meta_splits
from room_for_doubt.data.preprocessing import meta_transform
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor
from room_for_doubt.models.flow_matching import CFMTrajectoryModel
from room_for_doubt.inference.flow_sampling import sample_flow_trajectories
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.paths import config_directory
from room_for_doubt.utils.reproducibility import set_seed


def choose_device(request):
    if request == "auto":
        request = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return device


def read_targets(cfg):
    if not cfg.target_path:
        raise ValueError("Set target_path to verified meta_targets.npz")
    with np.load(to_absolute_path(cfg.target_path), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    validate_meta_splits(data, data["sample_ids"])
    if (
        data["dataset_name"].item() != cfg.dataset.name
        or data["noisy_split"].item() != cfg.dataset.noisy_split
    ):
        raise ValueError("Target dataset or annotation mismatch")
    if int(data["split_seed"]) != cfg.split_seed:
        raise ValueError("Split seed differs from prepared target")
    if (
        data["targets"].shape != (len(data["sample_ids"]), 50)
        or not np.isfinite(data["targets"]).all()
    ):
        raise ValueError("Expected finite aligned 50-epoch targets")
    return data


def save_config(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(
    config_path=str(config_directory()),
    config_name="meta_retraining",
    version_base=None,
)
def cache_main(cfg):
    data = read_targets(cfg)
    device = choose_device(cfg.device)
    out = Path(HydraConfig.get().runtime.output_dir)
    config = save_config(cfg, out)
    set_seed(cfg.seed)
    model = ConditionedPredictor(cfg.dataset.num_classes).to(device).eval()
    cls = {"cifar10n": CIFAR10, "cifar100n": CIFAR100}[cfg.dataset.name]
    base = cls(
        to_absolute_path(cfg.dataset.data_dir),
        train=True,
        download=False,
        transform=meta_transform(),
    )
    loader = DataLoader(
        Subset(base, data["sample_ids"].tolist()),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    features = []
    start = time.perf_counter()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            features.append(model.extract_features(images.to(device)).cpu().numpy())
            if i % 25 == 0:
                print(f"Feature cache {i + 1}/{len(loader)}", flush=True)
    (out / "artifacts").mkdir(exist_ok=True)
    np.savez_compressed(
        out / "artifacts/features.npz",
        features=np.concatenate(features),
        sample_ids=data["sample_ids"],
        dataset_name=data["dataset_name"],
        preprocessing=np.array(cfg.preprocessing),
    )
    torch.save(model.features.cpu().state_dict(), out / "artifacts/backbone.pt")
    metrics = dict(
        samples=len(data["sample_ids"]), elapsed_seconds=time.perf_counter() - start
    )
    meta = dict(
        seed=cfg.seed,
        device=str(device),
        checkpoint_path="artifacts/backbone.pt",
        backbone="ResNet18_Weights.IMAGENET1K_V1",
        preprocessing=cfg.preprocessing,
    )
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "Frozen deterministic features; no model optimization or test data used.",
    )


def head_loss(model, features, labels, targets, flow, generator=None):
    if not flow:
        return torch.nn.functional.mse_loss(
            model.predict_from_features(features, labels), targets
        )
    # CPU random draws allow a fixed validation generator on CUDA and MPS alike.
    t = torch.rand(len(targets), generator=generator).to(targets.device)
    x0 = torch.randn(targets.shape, generator=generator).to(targets.device)
    xt = (1 - t[:, None]) * x0 + t[:, None] * targets
    return torch.nn.functional.mse_loss(
        model.vector_field(xt, t, features, labels), targets - x0
    )


def fit_head(model, data, features, cfg, out, device):
    validate_meta_splits(data, data["sample_ids"])
    row = {int(s): i for i, s in enumerate(data["sample_ids"])}
    loaders = {}
    for part in ["meta_train", "meta_val"]:
        indices = [row[int(s)] for s in data[part + "_ids"]]
        ds = TensorDataset(
            torch.tensor(features[indices]),
            torch.tensor(data["noisy_labels"][indices], dtype=torch.long),
            torch.tensor(data["targets"][indices], dtype=torch.float32),
        )
        loaders[part] = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=part == "meta_train", num_workers=0
        )
    model.to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate
    )
    flow = cfg.meta_model == "cfm"
    start_epoch, best_loss, best_epoch = 0, float("inf"), None
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    history = []
    if cfg.resume_path:
        saved = torch.load(
            to_absolute_path(cfg.resume_path), map_location="cpu", weights_only=False
        )
        for key, default in [
            ("validation_metric", "native"),
            ("validation_flow_samples", 5),
            ("validation_flow_steps", 10),
        ]:
            if saved["config"].get(key, default) != cfg.get(key, default):
                raise ValueError(f"Resume validation config mismatch: {key}")
        for key in [
            "meta_model",
            "seed",
            "split_seed",
            "target_path",
            "cache_path",
            "epochs",
            "batch_size",
            "learning_rate",
        ]:
            if saved["config"][key] != OmegaConf.to_container(cfg, resolve=True)[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best_loss, best_epoch = (
            saved["epoch"] + 1,
            saved["best_val_loss"],
            saved["best_epoch"],
        )
        history = saved["history"]
        model.load_state_dict(saved["best_state_dict"])
        torch.save(
            {
                "state_dict": saved["best_state_dict"],
                "epoch": best_epoch,
                "val_loss": best_loss,
                "config": saved["config"],
            },
            out / "checkpoints/best.pt",
        )
        model.load_state_dict(saved["state_dict"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(start_epoch, cfg.epochs):
        losses = {}
        for part, loader in loaders.items():
            training = part == "meta_train"
            model.train(training)
            generator = (
                None if training else torch.Generator().manual_seed(cfg.split_seed)
            )
            total, count = 0.0, 0
            with torch.set_grad_enabled(training):
                for features_batch, labels, targets in loader:
                    f, y, x = (
                        features_batch.to(device),
                        labels.to(device),
                        targets.to(device),
                    )
                    if (
                        not training
                        and flow
                        and cfg.get("validation_metric", "native") == "trajectory_mse"
                    ):
                        reconstructed = sample_flow_trajectories(
                            model,
                            f,
                            y,
                            cfg.validation_flow_samples,
                            cfg.validation_flow_steps,
                            generator,
                        ).mean(1)
                        loss = torch.nn.functional.mse_loss(reconstructed, x)
                    else:
                        loss = head_loss(model, f, y, x, flow, generator)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training/validation loss")
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total += loss.item() * len(y)
                    count += len(y)
            losses[part] = total / count
        if losses["meta_val"] < best_loss:
            best_loss, best_epoch = losses["meta_val"], epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                out / "checkpoints/best.pt",
            )
        history.append(
            dict(
                epoch=epoch,
                train_loss=losses["meta_train"],
                val_loss=losses["meta_val"],
            )
        )
        best = torch.load(
            out / "checkpoints/best.pt", map_location="cpu", weights_only=False
        )
        torch.save(
            dict(
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                best_val_loss=best_loss,
                best_epoch=best_epoch,
                best_state_dict=best["state_dict"],
                config=OmegaConf.to_container(cfg, resolve=True),
                history=history,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None,
            ),
            out / "checkpoints/last.pt",
        )
        print(
            f"epoch {epoch + 1}/{cfg.epochs}: train={losses['meta_train']:.5g} val={losses['meta_val']:.5g}",
            flush=True,
        )
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    return dict(
        best_val_loss=best_loss, best_epoch=best_epoch, epochs_completed=len(history)
    )


@hydra.main(
    config_path=str(config_directory()),
    config_name="meta_retraining",
    version_base=None,
)
def main(cfg):
    if cfg.validation_metric not in {"native", "trajectory_mse"}:
        raise ValueError("Unknown validation metric")
    if (
        cfg.preprocessing != "imagenet_pil_bilinear_224"
        or cfg.supervision != "trajectory_regression_no_clean_labels"
    ):
        raise ValueError("Unsupported preprocessing or supervision")
    if cfg.meta_model not in {"conditioned", "cfm"}:
        raise ValueError(
            "Supported models: conditioned or cfm, independent of class count"
        )
    if cfg.optimizer != "adam":
        raise ValueError("This campaign fixes the optimizer to Adam")
    if not cfg.cache_path or cfg.epochs < 1:
        raise ValueError("Require cache_path and positive epochs")
    set_seed(cfg.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = read_targets(cfg)
    cache_dir = Path(to_absolute_path(cfg.cache_path))
    with np.load(cache_dir / "features.npz", allow_pickle=False) as cached:
        if (
            not np.array_equal(cached["sample_ids"], data["sample_ids"])
            or cached["dataset_name"].item() != cfg.dataset.name
        ):
            raise ValueError("Feature rows/dataset do not match targets")
        if cached["preprocessing"].item() != cfg.preprocessing:
            raise ValueError("Feature preprocessing mismatch")
        features = cached["features"]
    if (
        features.shape != (len(data["sample_ids"]), 512)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid feature cache")
    model = (CFMTrajectoryModel if cfg.meta_model == "cfm" else ConditionedPredictor)(
        cfg.dataset.num_classes, pretrained=False
    )
    model.features.load_state_dict(
        torch.load(cache_dir / "backbone.pt", map_location="cpu", weights_only=True)
    )
    out = Path(HydraConfig.get().runtime.output_dir)
    config = save_config(cfg, out)
    device = choose_device(cfg.device)
    start = time.perf_counter()
    metrics = fit_head(model, data, features, cfg, out, device)
    metrics["elapsed_seconds"] = time.perf_counter() - start
    (out / "artifacts").mkdir(exist_ok=True)
    np.savez_compressed(
        out / "artifacts/split_ids.npz",
        **{k: v for k, v in data.items() if k.endswith("_ids")},
    )
    meta = dict(
        seed=cfg.seed,
        device=str(device),
        checkpoint_path=str(out / "checkpoints/best.pt"),
        best_epoch=metrics["best_epoch"],
        best_val_loss=metrics["best_val_loss"],
        preprocessing=cfg.preprocessing,
        scope="historical_complete_case_salvage",
    )
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        f"Checkpoint criterion: {cfg.validation_metric}. For conditioned models native loss is trajectory MSE; for CFM native loss is velocity MSE, and trajectory_mse is Euler-sampled mean-trajectory MSE. Fixed validation random draws each epoch. No clean-label objective. Detector sets never used for fitting or checkpoint selection. No test evaluation. Base checkpoints are accepted by owner assumption.",
    )


if __name__ == "__main__":
    main()
