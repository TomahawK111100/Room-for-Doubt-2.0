import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


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


def generate_report(output_dir, run_id, config, metrics, metadata=None, notes=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md = output_dir / "report.md"
    meta = metadata or {}
    compact = {
        name: format(value, ".4g")
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    tables = output_dir / "tables"
    tables.mkdir(exist_ok=True)
    with (tables / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(compact.items())
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
        f.write("\n```\n\n## Compact Results\n\n| Metric | Value |\n|---|---:|\n")
        for name, value in compact.items():
            f.write(f"| {name} | {value} |\n")
        f.write("\n## Generated Artifacts\n\n")
        for path in sorted(output_dir.iterdir()):
            if path != md:
                f.write(f"- `{path.name}{'/' if path.is_dir() else ''}`\n")
        f.write(f"\n## Notes\n\n{notes or 'No run notes recorded.'}\n")
