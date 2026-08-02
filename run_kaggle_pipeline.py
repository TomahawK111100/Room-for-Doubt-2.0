import os
import subprocess


SEEDS = [42, 43, 44]
DATASETS = ["cifar10", "cifar100"]


def run_stage(script_path: str, dataset: str, seed: int) -> None:
    command = ["python", script_path, "--dataset", dataset, "--seed", str(seed)]
    subprocess.run(command, check=True, env=os.environ.copy())


def main() -> None:
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"\n{'=' * 18} RUNNING DATASET {dataset} | SEED {seed} {'=' * 18}\n")
            run_stage("src/train_base_model.py", dataset, seed)
            run_stage("src/train_conditioned_head.py", dataset, seed)
            run_stage("src/evaluate_top2_correction.py", dataset, seed)


if __name__ == "__main__":
    main()
