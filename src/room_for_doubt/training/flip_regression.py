import argparse
from room_for_doubt.models.flip_regression import MetaRegressor
from room_for_doubt.data.flip_regression import MetaRegressionDataset, ValSubsetWrapper
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Константы для нормализации CIFAR-10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(42)

    # ВОТ НАШ МАЯЧОК ОТЛАДКИ
    print("🚀 Старт скрипта: импорты прошли успешно!")
    print(
        "⏳ Начинаю загрузку JSON и скачивание CIFAR-10 (это займет пару минут, ждем)..."
    )

    data_dir = "./data"
    trajectories_path = "artifacts/trajectories/stage1_trajectories.json"

    if not os.path.exists(trajectories_path):
        raise FileNotFoundError(
            f"Файл {trajectories_path} не найден! Убедитесь, что Stage 1 завершен."
        )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    full_dataset = MetaRegressionDataset(
        data_dir, trajectories_path, transform=transform_train
    )
    print(f"✅ Датасет готов! Загружено {len(full_dataset)} примеров.")

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    val_set = ValSubsetWrapper(val_subset, transform_val)

    num_workers = os.cpu_count() or 4
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_subset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    model = MetaRegressor(lr=1e-4)
    model = torch.compile(model)

    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        deterministic=True,
        enable_checkpointing=False,
    )

    print("🤖 Модель инициализирована, запускаю Trainer...")
    trainer.fit(model, train_loader, val_loader)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/meta_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Обучение завершено. Веса мета-модели сохранены в {save_path}")


if __name__ == "__main__":
    main()
