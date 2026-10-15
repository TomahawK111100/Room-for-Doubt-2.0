import argparse
from room_for_doubt.models.direct_noise_detector import DirectCertaintyEstimator
from room_for_doubt.data.direct_noise_detector import DirectCertaintyDataset
import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    # Setup device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Transforms compatible with ImageNet ResNet-18
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    full_dataset = DirectCertaintyDataset(transform=transform)

    # Split into train and validation
    train_size = 45000
    val_size = 5000

    train_dataset = Subset(full_dataset, range(train_size))
    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Initialize the model
    model = DirectCertaintyEstimator(lr=1e-3)

    # Callbacks
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="direct_estimator_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Trainer
    trainer = Trainer(
        max_epochs=5,
        accelerator="gpu"
        if device == "cuda"
        else ("mps" if device == "mps" else "cpu"),
        devices=1,
        callbacks=[checkpoint_callback],
        logger=True,
        log_every_n_steps=50,
    )

    print("Starting training of Direct Certainty Estimator...")
    trainer.fit(model, train_loader, val_loader)
    print("Training complete!")


if __name__ == "__main__":
    main()
