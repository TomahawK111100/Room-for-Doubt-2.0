
import pytest
import os
import torch
import numpy as np
import json
from omegaconf import OmegaConf

from src.dataset import CIFARDataModule, CIFAR10N
from src.model import ResNetClassifier
from src.logging_utils import save_run_metadata, save_metrics, save_trajectory_batch, generate_report

# --- Fixtures for DataModule and Model ---

@pytest.fixture(scope="module")
def cifar_data_module():
    data_dir = "./data"
    dm = CIFARDataModule(data_dir=data_dir, batch_size=32, num_workers=0, val_ratio=0.1, seed=42)
    dm.prepare_data()
    dm.setup()
    return dm

@pytest.fixture(scope="module")
def resnet_model():
    model = ResNetClassifier(num_classes=10, pretrained=False, learning_rate=1e-3)
    return model

# --- Tests for Dataset and DataModule ---

def test_dataset_splits(cifar_data_module):
    train_dataset = cifar_data_module.cifar10n_train
    val_dataset = cifar_data_module.cifar10n_val
    test_dataset = cifar_data_module.cifar10_test

    assert len(train_dataset) == 45000
    assert len(val_dataset) == 5000
    assert len(test_dataset) == 10000

    # Check for disjointness of sample IDs
    train_ids = set(train_dataset.sample_ids)
    val_ids = set(val_dataset.sample_ids)
    assert train_ids.isdisjoint(val_ids)

    # Check that original CIFAR-10 test set IDs are 0 to 9999 (for the wrapped dataset)
    expected_test_ids = set(range(len(test_dataset)))
    actual_test_ids = set([test_dataset[i][3] for i in range(len(test_dataset))]) # sample_id is 4th element
    assert expected_test_ids == actual_test_ids

def test_stable_sample_ids():
    data_dir = "./data"
    dm1 = CIFARDataModule(data_dir=data_dir, batch_size=32, num_workers=0, val_ratio=0.1, seed=123)
    dm1.prepare_data()
    dm1.setup()

    train_ids_1 = set(dm1.cifar10n_train.sample_ids)
    val_ids_1 = set(dm1.cifar10n_val.sample_ids)

    # Create another instance with the same seed
    dm2 = CIFARDataModule(data_dir=data_dir, batch_size=32, num_workers=0, val_ratio=0.1, seed=123)
    dm2.prepare_data()
    dm2.setup()

    train_ids_2 = set(dm2.cifar10n_train.sample_ids)
    val_ids_2 = set(dm2.cifar10n_val.sample_ids)

    assert train_ids_1 == train_ids_2
    assert val_ids_1 == val_ids_2

def test_dataloader_output_format(cifar_data_module):
    train_loader = cifar_data_module.train_dataloader()
    val_loader = cifar_data_module.val_dataloader()
    test_loader = cifar_data_module.test_dataloader()

    # Train loader
    imgs, noisy_labels, clean_labels, sample_ids = next(iter(train_loader))
    assert imgs.shape == (32, 3, 32, 32)
    assert noisy_labels.shape == (32,)
    assert clean_labels.shape == (32,)
    assert sample_ids.shape == (32,)

    # Val loader
    imgs, noisy_labels, clean_labels, sample_ids = next(iter(val_loader))
    assert imgs.shape == (32, 3, 32, 32)
    assert noisy_labels.shape == (32,)
    assert clean_labels.shape == (32,)
    assert sample_ids.shape == (32,)

    # Test loader (clean labels as noisy, clean_labels as actual clean labels)
    imgs, labels, clean_labels, sample_ids = next(iter(test_loader))
    assert imgs.shape == (32, 3, 32, 32)
    assert labels.shape == (32,)
    assert clean_labels.shape == (32,)
    assert sample_ids.shape == (32,)

# --- Tests for Model ---

def test_resnet_forward_pass(resnet_model):
    dummy_input = torch.randn(1, 3, 32, 32)
    output = resnet_model(dummy_input)
    assert output.shape == (1, 10)

def test_resnet_adaptations(resnet_model):
    assert resnet_model.model.conv1.kernel_size == (3, 3)
    assert resnet_model.model.conv1.stride == (1, 1)
    assert isinstance(resnet_model.model.maxpool, torch.nn.Identity)

# --- Tests for Logging Utilities ---

@pytest.fixture(scope="function")
def dummy_output_dir():
    dir_path = "./test_output_logs"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    import shutil
    shutil.rmtree(dir_path)

def test_save_run_metadata(dummy_output_dir):
    dummy_config = OmegaConf.create({"seed": 42, "dataset": {"name": "cifar10n"}})
    run_id = "test_run_metadata"
    save_run_metadata(dummy_output_dir, run_id, OmegaConf.to_container(dummy_config, resolve=True), seed=42)
    assert os.path.exists(os.path.join(dummy_output_dir, "run_metadata.json"))
    with open(os.path.join(dummy_output_dir, "run_metadata.json"), "r") as f:
        metadata = json.load(f)
        assert metadata["run_id"] == run_id
        assert metadata["seed"] == 42

def test_save_metrics(dummy_output_dir):
    dummy_metrics = {"train_loss": 0.1, "val_acc": 0.9}
    save_metrics(dummy_output_dir, dummy_metrics)
    assert os.path.exists(os.path.join(dummy_output_dir, "metrics.json"))
    with open(os.path.join(dummy_output_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
        assert metrics["train_loss"] == 0.1

def test_save_trajectory_batch(dummy_output_dir):
    batch_data = {
        "per_sample_loss": torch.tensor([0.1, 0.2]),
        "per_sample_probs": torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        "noisy_labels": torch.tensor([1, 0]),
        "clean_labels": torch.tensor([1, 1]),
        "sample_ids": torch.tensor([10, 11])
    }
    save_trajectory_batch(dummy_output_dir, epoch=1, batch_data=batch_data)
    traj_dir = os.path.join(dummy_output_dir, "trajectories")
    assert os.path.exists(traj_dir)
    assert any(f.startswith("epoch_1_batch_") and f.endswith(".csv") for f in os.listdir(traj_dir))

def test_generate_report(dummy_output_dir):
    dummy_config = OmegaConf.create({"seed": 42, "dataset": {"name": "cifar10n"}})
    dummy_metrics = {"train_loss": 0.1, "val_acc": 0.9}
    run_id = "test_report_gen"
    generate_report(dummy_output_dir, run_id, OmegaConf.to_container(dummy_config, resolve=True), dummy_metrics, "Test notes.")
    report_path = os.path.join(dummy_output_dir, "report.md")
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        content = f.read()
        assert f"# Experiment Report: {run_id}" in content
        assert "## Notes" in content
        assert "Test notes." in content


