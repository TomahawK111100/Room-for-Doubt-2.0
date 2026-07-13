import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np

TRAJECTORY_COLUMNS = [
    "sample_id", "split", "epoch", "noisy_label", "clean_label",
    "per_sample_loss", "per_sample_probs",
]

# Optional uncertainty columns that might be added by distillation
OPTIONAL_TRAJECTORY_COLUMNS = [
    "predictive_entropy", "prediction_margin", "descriptor_distance"
]

def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def save_run_metadata(output_dir, run_id, config, **kwargs):
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "config": config,
        **kwargs,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_metrics(output_dir, metrics):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def save_trajectory_batch(output_dir: str, epoch: int, split: str, batch_data: dict) -> Path:
    rows = []
    loss = batch_data["per_sample_loss"].cpu().numpy()
    probs = batch_data["per_sample_probs"].cpu().numpy()
    
    # Check for optional uncertainty columns
    has_pe = "predictive_entropy" in batch_data
    has_pm = "prediction_margin" in batch_data
    has_dd = "descriptor_distance" in batch_data
    
    if has_pe: pe = batch_data["predictive_entropy"].cpu().numpy()
    if has_pm: pm = batch_data["prediction_margin"].cpu().numpy()
    if has_dd: dd = batch_data["descriptor_distance"].cpu().numpy()
    
    for i in range(len(loss)):
        row = {
            "sample_id": int(batch_data["sample_ids"][i]),
            "split": split,
            "epoch": epoch,
            "noisy_label": int(batch_data["noisy_labels"][i]),
            "clean_label": int(batch_data["clean_labels"][i]),
            "per_sample_loss": float(loss[i]),
            "per_sample_probs": probs[i].tolist(),
        }
        if has_pe: row["predictive_entropy"] = float(pe[i])
        if has_pm: row["prediction_margin"] = float(pm[i])
        if has_dd: row["descriptor_distance"] = float(dd[i])
        rows.append(row)
        
    traj_dir = Path(output_dir) / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = traj_dir / f"epoch_{epoch:03d}_{split}_batch_{batch_data.get('batch_idx', 0):04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def validate_trajectory_schema(df: pd.DataFrame) -> None:
    missing = set(TRAJECTORY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing trajectory columns: {missing}")


import matplotlib.pyplot as plt
import seaborn as sns

class DataCartographyCallback(pl.Callback):
    def __init__(self, output_dir: str, stage_name: str = "stage1"):
        self.output_dir = output_dir
        self.stage_name = stage_name
        self.history = defaultdict(lambda: {
            "predictions": [],
            "margins": [],
            "epochs": [],
            "true_label": None,
            "noisy_label": None
        })
        self.epoch_metrics = defaultdict(list)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stage_name == "stage1" and not trainer.sanity_checking:
            metrics = trainer.callback_metrics
            if "train_loss" in metrics:
                self.epoch_metrics["train_loss"].append(metrics["train_loss"].item())
            if "train_acc" in metrics:
                self.epoch_metrics["train_acc"].append(metrics["train_acc"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.stage_name == "stage1" and not trainer.sanity_checking:
            metrics = trainer.callback_metrics
            if "val_loss" in metrics:
                self.epoch_metrics["val_loss"].append(metrics["val_loss"].item())
            if "val_acc" in metrics:
                self.epoch_metrics["val_acc"].append(metrics["val_acc"].item())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.stage_name != "stage1":
            return
        if trainer.sanity_checking:
            return
        if not isinstance(outputs, dict) or "trajectory" not in outputs:
            return
            
        t = outputs["trajectory"]
        sample_ids = t["sample_ids"].cpu().numpy()
        probs = t["per_sample_probs"].cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        has_pm = "prediction_margin" in t
        if has_pm: pm = t["prediction_margin"].cpu().numpy()
        
        clean_labels = t["clean_labels"].cpu().numpy()
        noisy_labels = t["noisy_labels"].cpu().numpy()
        
        epoch = trainer.current_epoch
        
        for i, sid in enumerate(sample_ids):
            sid_int = int(sid)
            self.history[sid_int]["predictions"].append(int(preds[i]))
            self.history[sid_int]["epochs"].append(epoch)
            
            if self.history[sid_int]["true_label"] is None:
                self.history[sid_int]["true_label"] = int(clean_labels[i])
                self.history[sid_int]["noisy_label"] = int(noisy_labels[i])
                
            if has_pm:
                self.history[sid_int]["margins"].append(float(pm[i]))

    @torch.no_grad()
    def _run_stage2_inference(self, trainer, pl_module):
        pl_module.eval()
        device = pl_module.device
        dataloader = trainer.datamodule.train_dataloader()
        
        distances = {}
        for batch in dataloader:
            # batch is (imgs, noisy_labels, clean_labels, sample_ids)
            imgs = batch[0].to(device)
            sample_ids = batch[3].cpu().numpy()
            
            _, student_features = pl_module.student(imgs)
            _, teacher_features = pl_module.teacher(imgs)
            
            dd = torch.norm(student_features - teacher_features, p=2, dim=1).cpu().numpy()
            
            for i, sid in enumerate(sample_ids):
                distances[str(int(sid))] = float(dd[i])
                
        return distances

    def _plot_stage1_metrics(self, plots_dir: Path):
        epochs = range(1, len(self.epoch_metrics.get("train_loss", [])) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        if "train_loss" in self.epoch_metrics and "val_loss" in self.epoch_metrics:
            ax1.plot(epochs, self.epoch_metrics["train_loss"], label="Train Loss")
            ax1.plot(epochs, self.epoch_metrics["val_loss"], label="Val Loss")
        ax1.set_title("Loss over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        
        if "train_acc" in self.epoch_metrics and "val_acc" in self.epoch_metrics:
            ax2.plot(epochs, self.epoch_metrics["train_acc"], label="Train Acc")
            ax2.plot(epochs, self.epoch_metrics["val_acc"], label="Val Acc")
        ax2.set_title("Accuracy over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "stage1_loss_acc.png", dpi=300)
        plt.close()

    def _plot_stage1_scatter(self, plots_dir: Path, results: dict):
        flips = []
        aum = []
        for sid, data in results.items():
            flips.append(data.get("flips", 0))
            aum.append(data.get("aum", 0.0))
            
        plt.figure(figsize=(8, 6))
        # Draw demarcation lines conceptually
        # Variability is typically divided into Low and High, Confidence (AUM) is Low/High
        
        ax = sns.scatterplot(x=flips, y=aum, alpha=0.6, edgecolor=None)
        
        # Calculate quantiles for demarcation
        if flips and aum:
            flip_q = np.percentile(flips, 50)
            aum_q = np.percentile(aum, 50)
            
            ax.axvline(flip_q, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(aum_q, color='gray', linestyle='--', alpha=0.5)
            
            # Text annotations
            ax.text(np.min(flips), np.max(aum), "Easy", fontsize=12, color='green', verticalalignment='top')
            ax.text(np.max(flips), np.max(aum), "Ambiguous", fontsize=12, color='orange', horizontalalignment='right', verticalalignment='top')
            ax.text(np.min(flips), np.min(aum), "Hard", fontsize=12, color='red', verticalalignment='bottom')

        plt.title("Data Cartography (Stage 1)")
        plt.xlabel("Variability (Flips)")
        plt.ylabel("Confidence (AUM)")
        plt.tight_layout()
        plt.savefig(plots_dir / "stage1_cartography_scatter.png", dpi=300)
        plt.close()

    def _plot_stage1_aum_hist(self, plots_dir: Path, results: dict):
        aum = [data.get("aum", 0.0) for data in results.values()]
        
        plt.figure(figsize=(8, 6))
        sns.histplot(aum, bins=50, kde=True)
        plt.title("AUM Distribution (Stage 1)")
        plt.xlabel("Area Under the Margin (AUM)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "stage1_aum_distribution.png", dpi=300)
        plt.close()

    def _plot_stage2_histogram(self, plots_dir: Path, distances: dict):
        vals = list(distances.values())
        plt.figure(figsize=(8, 6))
        sns.histplot(vals, bins=50, kde=True)
        plt.title("Descriptor Distance Distribution (Stage 2)")
        plt.xlabel("L2 Distance (Student vs Teacher)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "stage2_distance_distribution.png", dpi=300)
        plt.close()

    def on_fit_end(self, trainer, pl_module):
        plots_dir = Path(self.output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        if self.stage_name == "stage1":
            results = {}
            for sid, data in self.history.items():
                preds = data["predictions"]
                flips = 0
                for i in range(1, len(preds)):
                    if preds[i] != preds[i-1]:
                        flips += 1
                data["flips"] = flips
                
                # Calculate AUM (mean margin over epochs)
                margins = data.get("margins", [])
                aum = float(np.mean(margins)) if margins else 0.0
                data["aum"] = aum
                
                results[str(sid)] = dict(data)
                
            out_path = Path(self.output_dir) / "stage1_trajectories.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
                
            self._plot_stage1_metrics(plots_dir)
            self._plot_stage1_scatter(plots_dir, results)
            self._plot_stage1_aum_hist(plots_dir, results)
            
        elif self.stage_name == "stage2":
            distances = self._run_stage2_inference(trainer, pl_module)
            
            out_path = Path(self.output_dir) / "stage2_final_distances.json"
            with open(out_path, "w") as f:
                json.dump(distances, f, indent=2)
                
            self._plot_stage2_histogram(plots_dir, distances)

class TrajectoryLoggerCallback(pl.Callback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def _save(self, trainer, outputs, split: str):
        if isinstance(outputs, dict) and "trajectory" in outputs:
            t = outputs["trajectory"]
            t["batch_idx"] = outputs.get("batch_idx", 0)
            save_trajectory_batch(self.output_dir, trainer.current_epoch, split, t)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict):
            outputs["batch_idx"] = batch_idx
        self._save(trainer, outputs, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict):
            outputs["batch_idx"] = batch_idx
        self._save(trainer, outputs, "val")


def generate_report(output_dir, run_id, config, metrics, metadata=None, notes=None):
    md = Path(output_dir) / "report.md"
    meta = metadata or {}
    with open(md, "w") as f:
        f.write(f"# Experiment Report: {run_id}\n\n")
        f.write("## Configuration\n```yaml\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n```\n\n## Dataset\n")
        f.write(f"- name: {config.get('dataset', {}).get('name')}\n")
        f.write(f"- noisy_split: {config.get('dataset', {}).get('noisy_split')}\n")
        f.write(f"- val_ratio: {config.get('dataset', {}).get('val_ratio')}\n\n")
        f.write("## Checkpoint\n")
        f.write(f"- path: {meta.get('checkpoint_path')}\n")
        f.write(f"- best_epoch: {meta.get('best_epoch')}\n")
        f.write(f"- best_val_loss: {meta.get('best_val_loss')}\n\n")
        f.write("## Metrics\n```json\n")
        f.write(json.dumps(metrics, indent=2))
        f.write("\n```\n\n## Generated Artifacts\n")
        for name in ["config.yaml", "metrics.json", "run_metadata.json", "trajectories/", "checkpoints/"]:
            f.write(f"- `{name}`\n")
        if notes:
            f.write(f"\n## Notes\n{notes}\n")
