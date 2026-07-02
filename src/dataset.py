
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np
import os
from PIL import Image

class CIFAR10N(Dataset):
    def __init__(self, root, train=True, transform=None, download=False, split_type='noisy_train', val_ratio=0.1, seed=42):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.split_type = split_type
        self.val_ratio = val_ratio
        self.seed = seed

        self.cifar10_dataset = datasets.CIFAR10(root=self.root, train=self.train, download=False, transform=None)
        if download:
            self._download_cifar10()
        self.noisy_labels = self._load_noisy_labels()

        if self.train:
            self._create_train_val_split()
        else:
            self.data = self.cifar10_dataset.data
            self.targets = self.cifar10_dataset.targets
            self.sample_ids = np.arange(len(self.data))

    def _download_cifar10(self):
        # Dummy download to ensure the base CIFAR-10 is available
        datasets.CIFAR10(root=self.root, train=True, download=True)
        datasets.CIFAR10(root=self.root, train=False, download=True)

    def _load_noisy_labels(self):
        # Placeholder for loading CIFAR-10N labels.
        # In a real scenario, this would load the .pt or .mat file.
        # For now, we'll use a dummy approach or raise an error if not found.
        noisy_label_path = os.path.join(self.root, 'cifar10_noisy_labels.npy')
        if os.path.exists(noisy_label_path):
            return np.load(noisy_label_path, allow_pickle=True).item()
        else:
            # Simulate downloading and processing CIFAR-10N labels
            print("CIFAR-10N labels not found locally. Attempting to simulate download.")
            # In a real setup, you would download from a URL like:
            # https://github.com/UCSC-VL/CIFAR-10N/raw/main/data/CIFAR-10_human.pt
            # For now, create dummy noisy labels for demonstration
            num_train_samples = 50000 # CIFAR-10 train size
            dummy_noisy_labels = {
                'noisy_labels': np.random.randint(0, 10, num_train_samples),
                'clean_labels': np.array(self.cifar10_dataset.targets)[:num_train_samples]
            }
            np.save(noisy_label_path, dummy_noisy_labels)
            print(f"Dummy CIFAR-10N labels created at {noisy_label_path}")
            return dummy_noisy_labels

    def _create_train_val_split(self):
        np.random.seed(self.seed)
        num_train_samples = len(self.cifar10_dataset)
        indices = np.arange(num_train_samples)
        np.random.shuffle(indices)

        num_val_samples = int(num_train_samples * self.val_ratio)
        if self.split_type == 'noisy_train':
            train_indices = indices[num_val_samples:]
            val_indices = indices[:num_val_samples]
            self.data = self.cifar10_dataset.data[train_indices]
            self.noisy_targets = self.noisy_labels['noisy_labels'][train_indices]
            self.clean_targets = self.noisy_labels['clean_labels'][train_indices]
            self.sample_ids = train_indices
        elif self.split_type == 'noisy_validation':
            train_indices = indices[num_val_samples:]
            val_indices = indices[:num_val_samples]
            self.data = self.cifar10_dataset.data[val_indices]
            self.noisy_targets = self.noisy_labels['noisy_labels'][val_indices]
            self.clean_targets = self.noisy_labels['clean_labels'][val_indices]
            self.sample_ids = val_indices
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, noisy_label, clean_label, sample_id = self.data[idx], self.noisy_targets[idx], self.clean_targets[idx], self.sample_ids[idx]
        img = Image.fromarray(img) # Assuming img is a numpy array
        if self.transform:
            img = self.transform(img)
        return img, noisy_label, clean_label, sample_id


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64, num_workers=4, val_ratio=0.1, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def prepare_data(self):
        # Download CIFAR-10 if not already present
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
        # Simulate CIFAR-10N labels download if not present
        cifar10n_dummy = CIFAR10N(self.data_dir, download=True, train=True) # This will create dummy labels

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.cifar10n_train = CIFAR10N(self.data_dir, train=True, transform=self.transform_train, split_type='noisy_train', val_ratio=self.val_ratio, seed=self.seed)
            self.cifar10n_val = CIFAR10N(self.data_dir, train=True, transform=self.transform_test, split_type='noisy_validation', val_ratio=self.val_ratio, seed=self.seed)

        # Assign test dataset for use in dataloaders
        if stage == 'test' or stage is None:
            # Use original clean CIFAR-10 test set
            self.cifar10_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform_test)
            # Wrap it to match the CIFAR10N __getitem__ signature, providing dummy clean_label and sample_id
            class WrappedCIFAR10Test(Dataset):
                def __init__(self, cifar10_dataset):
                    self.cifar10_dataset = cifar10_dataset
                def __len__(self):
                    return len(self.cifar10_dataset)
                def __getitem__(self, idx):
                    img, label = self.cifar10_dataset[idx]
                    return img, label, label, idx # noisy_label=clean_label, clean_label, sample_id
            self.cifar10_test = WrappedCIFAR10Test(self.cifar10_test)

    def train_dataloader(self):
        return DataLoader(self.cifar10n_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10n_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False, pin_memory=True)


if __name__ == '__main__':
    # Basic testing of the dataset and datamodule
    data_module = CIFARDataModule(data_dir='./data', batch_size=128, num_workers=0)
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Train loader size: {len(train_loader.dataset)} samples")
    print(f"Validation loader size: {len(val_loader.dataset)} samples")
    print(f"Test loader size: {len(test_loader.dataset)} samples")

    # Verify data format
    for batch_idx, (imgs, noisy_labels, clean_labels, sample_ids) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Imgs shape {imgs.shape}, Noisy Labels shape {noisy_labels.shape}, Clean Labels shape {clean_labels.shape}, Sample IDs shape {sample_ids.shape}")
        print(f"Sample IDs: {sample_ids}")
        break
    
    for batch_idx, (imgs, noisy_labels, clean_labels, sample_ids) in enumerate(val_loader):
        print(f"Batch {batch_idx}: Imgs shape {imgs.shape}, Noisy Labels shape {noisy_labels.shape}, Clean Labels shape {clean_labels.shape}, Sample IDs shape {sample_ids.shape}")
        print(f"Sample IDs: {sample_ids}")
        break

    for batch_idx, (imgs, labels, clean_labels, sample_ids) in enumerate(test_loader):
        print(f"Batch {batch_idx}: Imgs shape {imgs.shape}, Labels shape {labels.shape}")
        print(f"Sample IDs: {sample_ids}")
        break

    # Test stability of sample IDs in train/val split
    data_module_2 = CIFARDataModule(data_dir='./data', batch_size=128, num_workers=0, seed=42) # Same seed
    data_module_2.prepare_data()
    data_module_2.setup()
    
    train_loader_2 = data_module_2.train_dataloader()
    val_loader_2 = data_module_2.val_dataloader()

    # Get sample IDs from first batches
    _, _, _, sample_ids_train_1 = next(iter(train_loader))
    _, _, _, sample_ids_train_2 = next(iter(train_loader_2))
    
    _, _, _, sample_ids_val_1 = next(iter(val_loader))
    _, _, _, sample_ids_val_2 = next(iter(val_loader_2))

    print(f"First batch train IDs (run 1): {sample_ids_train_1}")
    print(f"First batch train IDs (run 2): {sample_ids_train_2}")
    print(f"First batch val IDs (run 1): {sample_ids_val_1}")
    print(f"First batch val IDs (run 2): {sample_ids_val_2}")

    # Note: With shuffle=True, direct comparison of first batches is not enough
    # to guarantee overall stable IDs for specific samples. However, the underlying
    # split logic itself is deterministic with fixed seed. For a true test of stable
    # IDs, one would need to iterate through entire datasets and compare sets of IDs.
    # This is a basic check.

    # Ensure unique IDs in train and val sets
    train_ids = set(data_module.cifar10n_train.sample_ids)
    val_ids = set(data_module.cifar10n_val.sample_ids)
    print(f"Are train and validation IDs disjoint? {train_ids.isdisjoint(val_ids)}")
