
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
import json

from src.dataset import CIFARDataModule
from src.model import ResNetClassifier
from src.logging_utils import save_run_metadata, save_metrics, save_trajectory_batch, generate_report

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Initialize data module
    data_module = CIFARDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.seed
    )

    # Initialize model
    model = ResNetClassifier(
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        learning_rate=cfg.optimizer.lr
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=os.path.join(cfg.output_dir, "checkpoints"),
        filename="best_model",
        save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = TensorBoardLogger(
        save_dir=cfg.output_dir, 
        name="tensorboard_logs", 
        version=cfg.run_id
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model, data_module)

    # Load the best model after training
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    # You can load the best model for further evaluation if needed
    # best_model = ResNetClassifier.load_from_checkpoint(best_model_path)

    # Evaluate on test set
    # trainer.test(model, data_module)

    # Collect per-sample trajectories from training_step outputs
    # This part requires modifying the training_step to return the necessary data
    # And then aggregating it here. For simplicity, let's assume we collect them directly.
    # In a real scenario, you might have a custom logger or callback to handle this efficiently.
    print("Collecting per-sample trajectories...")
    # This is a simplified approach. For large datasets, consider a custom DDP-compatible logger.
    all_sample_trajectories = []
    for epoch in range(cfg.trainer.max_epochs):
        # Simulate iterating through the training data again to get per-sample data for this epoch
        # In a real PL setup, you'd integrate this with a custom callback that saves per-batch data
        # Or modify training_step_end to accumulate data.
        # For this example, we'll re-run a dummy loop or assume data is passed via a simpler mechanism.
        # A more robust solution involves a custom PyTorch Lightning Logger or Callback
        # that saves batch-level outputs from training_step.
        pass # Actual logging will happen inside the training_step if properly configured
    
    # For demonstration, let's assume we have a way to get the aggregated trajectory data
    # from a custom logging callback that saves data during trainer.fit
    # For now, we'll skip actual aggregation here and assume save_trajectory_batch is called per batch.
    
    # Example of how aggregated metrics might look (replace with actual collected metrics)
    metrics = {
        "train_loss_final": trainer.callback_metrics.get("train_loss_epoch").item() if "train_loss_epoch" in trainer.callback_metrics else None,
        "val_loss_final": trainer.callback_metrics.get("val_loss_epoch").item() if "val_loss_epoch" in trainer.callback_metrics else None,
        "val_acc_final": trainer.callback_metrics.get("val_acc_epoch").item() if "val_acc_epoch" in trainer.callback_metrics else None,
        "test_loss_final": trainer.callback_metrics.get("test_loss").item() if "test_loss" in trainer.callback_metrics else None,
        "test_acc_final": trainer.callback_metrics.get("test_acc").item() if "test_acc" in trainer.callback_metrics else None,
    }
    print(f"Final Metrics: {metrics}")

    # Save outputs
    output_dir = os.path.join(cfg.output_dir, cfg.run_id)
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))
    save_metrics(output_dir, metrics)
    save_run_metadata(output_dir, cfg.run_id, OmegaConf.to_container(cfg, resolve=True), 
                      best_epoch=checkpoint_callback.best_epoch,
                      best_val_loss=checkpoint_callback.best_model_score.item(),
                      seed=cfg.seed,
                      dataset_split=cfg.dataset.name,
                      checkpoint_path=best_model_path)
    generate_report(output_dir, cfg.run_id, OmegaConf.to_container(cfg, resolve=True), metrics)


if __name__ == "__main__":
    main()
