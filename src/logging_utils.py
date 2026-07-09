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


class DataCartographyCallback(pl.Callback):
    def __init__(self, output_dir: str, stage_name: str = "stage1"):
        self.output_dir = output_dir
        self.stage_name = stage_name
        self.history = defaultdict(lambda: {
            "predictions": [],
            "margins": [],
            "descriptor_distances": [],
            "epochs": [],
            "true_label": None,
            "noisy_label": None
        })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict) or "trajectory" not in outputs:
            return
            
        t = outputs["trajectory"]
        sample_ids = t["sample_ids"].cpu().numpy()
        probs = t["per_sample_probs"].cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        has_pm = "prediction_margin" in t
        has_dd = "descriptor_distance" in t
        
        if has_pm: pm = t["prediction_margin"].cpu().numpy()
        if has_dd: dd = t["descriptor_distance"].cpu().numpy()
        
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
            if has_dd:
                self.history[sid_int]["descriptor_distances"].append(float(dd[i]))

    def on_fit_end(self, trainer, pl_module):
        # Calculate flips and save
        results = {}
        for sid, data in self.history.items():
            preds = data["predictions"]
            flips = 0
            for i in range(1, len(preds)):
                if preds[i] != preds[i-1]:
                    flips += 1
            data["flips"] = flips
            results[str(sid)] = dict(data)
            
        out_path = Path(self.output_dir) / f"{self.stage_name}_trajectories.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

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
