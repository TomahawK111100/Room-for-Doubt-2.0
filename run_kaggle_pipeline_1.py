"""Run the CIFAR-10N Worse setup independently."""

from run_kaggle_pipeline import run_setups


if __name__ == "__main__":
    run_setups([
        {"dataset_name": "cifar10n", "noisy_split": "worse"},
    ])
