from pathlib import Path
import pandas as pd

TRAJECTORY_COLUMNS = [
    "sample_id",
    "split",
    "epoch",
    "noisy_label",
    "clean_label",
    "per_sample_loss",
    "per_sample_probs",
]

# Optional uncertainty columns that might be added by distillation
OPTIONAL_TRAJECTORY_COLUMNS = [
    "predictive_entropy",
    "prediction_margin",
    "descriptor_distance",
]


def save_trajectory_batch(
    output_dir: str, epoch: int, split: str, batch_data: dict
) -> Path:
    rows = []
    loss = batch_data["per_sample_loss"].cpu().numpy()
    probs = batch_data["per_sample_probs"].cpu().numpy()

    # Check for optional uncertainty columns
    has_pe = "predictive_entropy" in batch_data
    has_pm = "prediction_margin" in batch_data
    has_dd = "descriptor_distance" in batch_data

    if has_pe:
        pe = batch_data["predictive_entropy"].cpu().numpy()
    if has_pm:
        pm = batch_data["prediction_margin"].cpu().numpy()
    if has_dd:
        dd = batch_data["descriptor_distance"].cpu().numpy()

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
        if has_pe:
            row["predictive_entropy"] = float(pe[i])
        if has_pm:
            row["prediction_margin"] = float(pm[i])
        if has_dd:
            row["descriptor_distance"] = float(dd[i])
        rows.append(row)

    traj_dir = Path(output_dir) / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = (
        traj_dir
        / f"epoch_{epoch:03d}_{split}_batch_{batch_data.get('batch_idx', 0):04d}.csv"
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def validate_trajectory_schema(df: pd.DataFrame) -> None:
    missing = set(TRAJECTORY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing trajectory columns: {missing}")
