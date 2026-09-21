"""Run the CIFAR-100N and CIFAR-10N Random1 setups independently."""

from run_kaggle_pipeline import run_setups


if __name__ == "__main__":
    run_setups([
        {"dataset_name": "cifar100n", "noisy_split": "noisy100"},
        {"dataset_name": "cifar10n", "noisy_split": "random1"},
    ])
