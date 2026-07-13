import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.dataset import CIFARDataModule
from src.logging_utils import TrajectoryLoggerCallback, DataCartographyCallback, generate_report, save_metrics, save_run_metadata
from src.model import ResNetClassifier, DescriptorDistillationModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    output_dir = Path(os.getcwd())

    num_workers = os.cpu_count() or 4
    dm = CIFARDataModule(
        data_dir=cfg.dataset.data_dir,
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
            temperature=cfg.model.temperature
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
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        deterministic=cfg.trainer.deterministic,
        precision=cfg.trainer.get("precision", "32-true"),
        callbacks=[
            ckpt_cb, 
            TrajectoryLoggerCallback(str(output_dir)), 
            DataCartographyCallback(str(output_dir), stage_name=stage_name),
            LearningRateMonitor()
        ],
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path="best")

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if v is not None}
    config = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    save_metrics(output_dir, metrics)
    meta = {
        "best_model_path": ckpt_cb.best_model_path,
        "best_val_loss": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score else None,
        "seed": cfg.seed,
        "dataset_split": cfg.dataset.noisy_split,
        "checkpoint_path": ckpt_cb.best_model_path,
    }
    save_run_metadata(output_dir, cfg.run_id, config, **meta)
    generate_report(output_dir, cfg.run_id, config, metrics, meta)


if __name__ == "__main__":
    main()
