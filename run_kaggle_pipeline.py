import subprocess
import os
import sys
import shutil
from pathlib import Path

seeds = [42, 43, 44, 45, 46]
setups = [
    {"dataset_name": "cifar10n", "noisy_split": "worse"},
    {"dataset_name": "cifar100n", "noisy_split": "noisy100"},
    {"dataset_name": "cifar10n", "noisy_split": "random1"},
]
artifact_root = Path("kaggle_artifacts")

def run_stage(command, log_path=None):
    print(f"\n>>> RUNNING: {' '.join(command)}")
    log_file = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w")
        result = subprocess.run(
            command,
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT if log_file is not None else None,
        )
    finally:
        if log_file is not None:
            log_file.close()
    if result.returncode != 0:
        print(f"!!! ERROR: Command failed with exit code {result.returncode}")
        sys.exit(1)


def copy_file(source, destination):
    source = Path(source)
    if source.exists():
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True
    return False


def copy_directory(source, destination):
    source = Path(source)
    if source.exists():
        destination = Path(destination)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return True
    return False


def collect_run_artifacts(dataset_name, noisy_split, seed, run_dir):
    """Create a self-contained, downloadable bundle for one dataset/seed run."""
    setup_id = f"{dataset_name}_{noisy_split}"
    run_artifact_dir = artifact_root / setup_id / f"seed{seed}"
    copy_directory(run_dir, run_artifact_dir / "training")

    trajectory = f"results/stage1_trajectories_{setup_id}_seed{seed}.json"
    checkpoint = f"checkpoints/best_{setup_id}_seed{seed}.ckpt"
    conditioned_head = f"checkpoints/cond_head_{setup_id}_seed{seed}.pth"
    correction_csv = f"results/threshold_sweep_{setup_id}_seed{seed}.csv"

    for source, name in [
        (trajectory, "trajectories/stage1.json"),
        (checkpoint, "checkpoints/best.ckpt"),
        (conditioned_head, "checkpoints/conditioned_head.pth"),
        (correction_csv, "metrics/threshold_sweep.csv"),
    ]:
        copy_file(source, run_artifact_dir / name)

    for plot in Path("plots").glob(f"*_{setup_id}_seed{seed}.png"):
        copy_file(plot, run_artifact_dir / "plots" / plot.name)


def collect_shared_artifacts():
    """Copy aggregate outputs into one top-level Kaggle download directory."""
    for directory in ("results", "plots", "checkpoints"):
        copy_directory(directory, artifact_root / "_shared" / directory)

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    for setup in setups:
        dataset_name = setup["dataset_name"]
        noisy_split = setup["noisy_split"]
        dataset = dataset_name.removesuffix("n")
        setup_id = f"{dataset_name}_{noisy_split}"
        for seed in seeds:
            print(f"\n{'='*40}")
            print(f"   STARTING SETUP: {setup_id.upper()} | SEED: {seed}")
            print(f"{'='*40}")

            run_dir = f"outputs/{setup_id}_seed{seed}"

            run_stage([
                "python", "src/train.py",
                f"dataset={dataset_name}",
                f"dataset.noisy_split={noisy_split}",
                f"seed={seed}",
                f"hydra.run.dir={run_dir}"
            ])
            
            traj_source = f"{run_dir}/stage1_trajectories.json"
            if os.path.exists(traj_source):
                target_path = f"results/stage1_trajectories_{setup_id}_seed{seed}.json"
                shutil.copy(traj_source, target_path)
            else:
                print(f"!!! WARNING: Trajectory file not found at {traj_source}")
            
            ckpt_source = f"{run_dir}/checkpoints/best.ckpt"
            if os.path.exists(ckpt_source):
                target_ckpt = f"checkpoints/best_{setup_id}_seed{seed}.ckpt"
                shutil.copy(ckpt_source, target_ckpt)
            else:
                print(f"!!! WARNING: Checkpoint file not found at {ckpt_source}")

            run_stage([
                "python", "src/train_conditioned_head.py",
                "--dataset", dataset, "--split", noisy_split, "--seed", str(seed),
            ])
            run_stage([
                "python", "src/evaluate_top2_correction.py",
                "--dataset", dataset, "--split", noisy_split, "--seed", str(seed),
            ])
            collect_run_artifacts(dataset_name, noisy_split, seed, run_dir)

    # Run aggregate evaluations only after every independent run has completed.
    for setup in setups:
        dataset_name = setup["dataset_name"]
        noisy_split = setup["noisy_split"]
        dataset = dataset_name.removesuffix("n")
        setup_id = f"{dataset_name}_{noisy_split}"
        ensemble_checkpoints = [
            Path(f"checkpoints/best_{setup_id}_seed{seed}.ckpt") for seed in seeds
        ]
        if all(checkpoint.exists() for checkpoint in ensemble_checkpoints):
            run_stage([
                "python", "src/evaluate_ensembles.py",
                "--dataset", dataset, "--split", noisy_split,
            ])
        else:
            print(f"!!! WARNING: Skipping {setup_id} ensemble; fewer than {len(seeds)} checkpoints exist.")

    for setup in setups:
        dataset_name = setup["dataset_name"]
        noisy_split = setup["noisy_split"]
        dataset = dataset_name.removesuffix("n")
        setup_id = f"{dataset_name}_{noisy_split}"
        tta_checkpoint = Path(f"checkpoints/best_{setup_id}_seed{seeds[0]}.ckpt")
        if tta_checkpoint.exists():
            run_stage([
                "python", "src/evaluate_tta.py",
                "--dataset", dataset, "--split", noisy_split,
                "--checkpoint", str(tta_checkpoint),
            ], log_path=artifact_root / setup_id / "evaluations" / "tta.log")
        else:
            print(f"!!! WARNING: Skipping {setup_id} TTA; setup checkpoint is required.")

    collect_shared_artifacts()

if __name__ == "__main__":
    main()
