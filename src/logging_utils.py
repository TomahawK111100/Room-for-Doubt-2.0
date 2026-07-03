
import json
import os
import torch
import pandas as pd
from datetime import datetime

def save_run_metadata(output_dir, run_id, config, best_epoch=None, best_val_loss=None, seed=None, dataset_split=None, checkpoint_path=None):
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,  # Save the resolved Hydra config
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "seed": seed,
        "dataset_split": dataset_split,
        "checkpoint_path": checkpoint_path
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

def save_metrics(output_dir, metrics):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def save_trajectory_batch(output_dir, epoch, batch_data):
    # batch_data contains per_sample_loss, per_sample_probs, noisy_labels, clean_labels, sample_ids
    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
    
    # Convert tensors to numpy arrays for saving
    per_sample_loss_np = batch_data["per_sample_loss"].numpy()
    per_sample_probs_np = batch_data["per_sample_probs"].numpy()
    noisy_labels_np = batch_data["noisy_labels"].numpy()
    clean_labels_np = batch_data["clean_labels"].numpy()
    sample_ids_np = batch_data["sample_ids"].numpy()

    # Create a DataFrame for this batch
    batch_df = pd.DataFrame({
        "sample_id": sample_ids_np,
        "epoch": epoch,
        "noisy_label": noisy_labels_np,
        "clean_label": clean_labels_np,
        "per_sample_loss": per_sample_loss_np,
        "per_sample_probs": list(per_sample_probs_np) # Store probabilities as list of arrays
    })
    
    # Save each batch individually for now. Later, these can be aggregated.
    # Using a unique filename for each batch to avoid overwriting.
    batch_filename = os.path.join(output_dir, "trajectories", f"epoch_{epoch}_batch_{datetime.now().strftime("%H%M%S%f")}.csv")
    batch_df.to_csv(batch_filename, index=False)

def generate_report(output_dir, run_id, config, metrics, additional_notes=None):
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(f"# Experiment Report: {run_id}\n\n")
        f.write("## Configuration\n")
        f.write(f"```yaml\n{json.dumps(config, indent=2)}\n```\n\n")
        f.write("## Metrics\n")
        f.write(f"```json\n{json.dumps(metrics, indent=2)}\n```\n\n")
        f.write("## Generated Artifacts\n")
        f.write(f"- `config.yaml`\n")
        f.write(f"- `metrics.json`\n")
        f.write(f"- `run_metadata.json`\n")
        f.write(f"- `report.md`\n")
        f.write(f"- `trajectories/` (per-sample logs)\n\n")

        if additional_notes:
            f.write("## Notes\n")
            f.write(f"{additional_notes}\n\n")
    print(f"Report generated at {report_path}")


# Example usage (for testing purposes)
if __name__ == "__main__":
    dummy_output_dir = "./test_output"
    os.makedirs(dummy_output_dir, exist_ok=True)

    # Dummy config and metrics
    dummy_config = {"dataset": {"name": "cifar10n", "val_ratio": 0.1}, "model": {"name": "resnet18"}}
    dummy_metrics = {"train_loss": 0.5, "val_loss": 0.8, "val_acc": 0.7}
    dummy_run_id = "test_run_123"

    save_run_metadata(dummy_output_dir, dummy_run_id, dummy_config, best_epoch=5, best_val_loss=0.75, seed=42, checkpoint_path="checkpoints/best.ckpt")
    save_metrics(dummy_output_dir, dummy_metrics)

    # Dummy batch data for trajectory logging
    batch_data = {
        "per_sample_loss": torch.tensor([0.1, 0.2, 0.3]),
        "per_sample_probs": torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]]),
        "noisy_labels": torch.tensor([1, 0, 1]),
        "clean_labels": torch.tensor([1, 1, 0]),
        "sample_ids": torch.tensor([100, 101, 102])
    }
    save_trajectory_batch(dummy_output_dir, epoch=1, batch_data=batch_data)

    # Generate a dummy report
    generate_report(dummy_output_dir, dummy_run_id, dummy_config, dummy_metrics, additional_notes="Initial dummy run for testing logging utilities.")

    print(f"Test output saved to {dummy_output_dir}")

