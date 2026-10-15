from pathlib import Path

import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from room_for_doubt.data.cifar_noisy import CIFARDataModule
from room_for_doubt.logging.callbacks import (
    TrajectoryLoggerCallback,
    DataCartographyCallback,
)
from room_for_doubt.reporting.artifacts import (
    generate_report,
    save_metrics,
    save_run_metadata,
)
from room_for_doubt.models.classifier import (
    ResNetClassifier,
    DescriptorDistillationModule,
)
from room_for_doubt.utils.paths import config_directory


@hydra.main(
    config_path=str(config_directory()), config_name="config", version_base=None
)
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg.seed)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml", resolve=True)

    num_workers = cfg.dataset.num_workers
    dm = CIFARDataModule(
        data_dir=to_absolute_path(cfg.dataset.data_dir),
        dataset_name=cfg.dataset.name,
        noisy_split=cfg.dataset.noisy_split,
        batch_size=cfg.dataset.batch_size,
        num_workers=num_workers,
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.seed,
        num_classes=cfg.model.num_classes,
    )

    if cfg.model.name == "descriptor_distillation":
        model = DescriptorDistillationModule(
            num_classes=cfg.model.num_classes,
            pretrained_teacher=cfg.model.pretrained_teacher,
            learning_rate=cfg.optimizer.lr,
            distill_alpha=cfg.model.distill_alpha,
            temperature=cfg.model.temperature,
        )
        stage_name = "stage2"
    else:
        model = ResNetClassifier(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            learning_rate=cfg.optimizer.lr,
        )
        stage_name = "stage1"

    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Скомпилируем модель для ускорения
    model = torch.compile(model)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        deterministic=cfg.trainer.deterministic,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            ckpt_cb,
            TrajectoryLoggerCallback(str(output_dir)),
            DataCartographyCallback(str(output_dir), stage_name=stage_name),
            LearningRateMonitor(),
        ],
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, dm)
    checkpoint = torch.load(
        ckpt_cb.best_model_path, map_location="cpu", weights_only=False
    )
    best_epoch = checkpoint.get("epoch")
    del checkpoint
    trainer.test(model, dm, ckpt_path="best")

    metrics = {
        k: float(v) for k, v in trainer.callback_metrics.items() if v is not None
    }
    config = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml", resolve=True)
    save_metrics(output_dir, metrics)
    meta = {
        "best_model_path": ckpt_cb.best_model_path,
        "best_epoch": best_epoch,
        "best_val_loss": float(ckpt_cb.best_model_score)
        if ckpt_cb.best_model_score is not None
        else None,
        "seed": cfg.seed,
        "dataset_split": cfg.dataset.noisy_split,
        "checkpoint_path": ckpt_cb.best_model_path,
        "device": str(trainer.strategy.root_device),
    }
    save_run_metadata(output_dir, cfg.run_id, config, **meta)
    generate_report(output_dir, cfg.run_id, config, metrics, meta)


if __name__ == "__main__":
    main()
