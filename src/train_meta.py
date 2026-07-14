import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path

# Константы для нормализации CIFAR-10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

class MetaRegressionDataset(Dataset):
    def __init__(self, data_dir: str, trajectories_path: str, transform=None):
        self.cifar = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        self.transform = transform
        
        with open(trajectories_path, 'r') as f:
            self.trajectories = json.load(f)
        
        self.valid_indices = []
        self.targets = []
        
        for idx in range(len(self.cifar)):
            str_idx = str(idx)
            if str_idx in self.trajectories:
                self.valid_indices.append(idx)
                flips = self.trajectories[str_idx].get("flips", 0)
                self.targets.append(float(flips))
                
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        orig_idx = self.valid_indices[i]
        img, _ = self.cifar[orig_idx]
        
        if self.transform:
            img = self.transform(img)
            
        target = torch.tensor(self.targets[i], dtype=torch.float32)
        return img, target

class ValSubsetWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        orig_idx = self.subset.indices[idx]
        dataset = self.subset.dataset
        real_cifar_idx = dataset.valid_indices[orig_idx]
        
        img, _ = dataset.cifar[real_cifar_idx]
        if self.transform:
            img = self.transform(img)
            
        target = torch.tensor(dataset.targets[orig_idx], dtype=torch.float32)
        return img, target

class MetaRegressor(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        loss = F.mse_loss(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self(imgs)
        loss = F.mse_loss(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42)
    
    # ВОТ НАШ МАЯЧОК ОТЛАДКИ
    print("🚀 Старт скрипта: импорты прошли успешно!")
    print("⏳ Начинаю загрузку JSON и скачивание CIFAR-10 (это займет пару минут, ждем)...")
    
    data_dir = "./data"
    trajectories_path = "results/stage1_trajectories.json"
    
    if not os.path.exists(trajectories_path):
        raise FileNotFoundError(f"Файл {trajectories_path} не найден! Убедитесь, что Stage 1 завершен.")
        
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    full_dataset = MetaRegressionDataset(data_dir, trajectories_path, transform=transform_train)
    print(f"✅ Датасет готов! Загружено {len(full_dataset)} примеров.")
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    val_set = ValSubsetWrapper(val_subset, transform_val)
    
    num_workers = os.cpu_count() or 4
    persistent_workers = num_workers > 0
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)

    model = MetaRegressor(lr=1e-4)
    model = torch.compile(model)

    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        deterministic=True,
        enable_checkpointing=False
    )

    print("🤖 Модель инициализирована, запускаю Trainer...")
    trainer.fit(model, train_loader, val_loader)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/meta_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Обучение завершено. Веса мета-модели сохранены в {save_path}")

if __name__ == "__main__":
    main()