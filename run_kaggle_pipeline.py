import subprocess
import os
import sys
import glob
import shutil

seeds = [42, 43, 44]
datasets = ['cifar10', 'cifar100']

def run_stage(command):
    print(f"\n>>> RUNNING: {' '.join(command)}")
    result = subprocess.run(command, env=os.environ.copy())
    if result.returncode != 0:
        print(f"!!! ERROR: Command failed with exit code {result.returncode}")
        sys.exit(1)

def main():
    for dataset in datasets:
        for seed in seeds:
            print(f"\n{'='*40}")
            print(f"   STARTING DATASET: {dataset.upper()} | SEED: {seed}")
            print(f"{'='*40}")
            
            hydra_dataset = f"{dataset}n"
            run_stage(["python", "src/train.py", f"dataset={hydra_dataset}", f"seed={seed}"])
            
            found_files = glob.glob("**/stage1_trajectories*.json", recursive=True)
            if found_files:
                latest_file = max(found_files, key=os.path.getctime)
                os.makedirs("results", exist_ok=True)
                target_path = f"results/stage1_trajectories_{dataset}_seed{seed}.json"
                shutil.copy(latest_file, target_path)
            
            if os.path.exists("checkpoints/best.ckpt"):
                target_ckpt = f"checkpoints/best_{dataset}_seed{seed}.ckpt"
                shutil.copy("checkpoints/best.ckpt", target_ckpt)

            run_stage(["python", "src/train_conditioned_head.py", "--dataset", dataset, "--seed", str(seed)])
            run_stage(["python", "src/evaluate_top2_correction.py", "--dataset", dataset, "--seed", str(seed)])

if __name__ == "__main__":
    main()
