from room_for_doubt.data.trajectory_records import historical_trajectory_path
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor
from room_for_doubt.data.conditioned_trajectories import CondTrajDataset
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from room_for_doubt.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Train conditioned trajectory predictor."
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"]
    )
    parser.add_argument("--split", type=str, default="worse", choices=["worse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trajectories-path",
        default=None,
        help="Verified trajectory file; default uses audited historical filenames",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epoch-start", type=int, default=0)
    parser.add_argument(
        "--missing-epoch-policy", choices=["drop", "error"], default="drop"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    num_classes = 10 if args.dataset == "cifar10" else 100

    # Setup device
    device = torch.device(
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
    setup_id = f"{args.dataset}n_{args.split}"
    trajectory_path = args.trajectories_path or historical_trajectory_path(
        args.dataset, args.seed
    )
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(
            f"Trajectory file '{trajectory_path}' not found. Run stage 1 for this dataset and seed first."
        )
    full_dataset = CondTrajDataset(
        dataset_name=args.dataset,
        traj_file=trajectory_path,
        transform=transform,
        seed=args.seed,
        val_ratio=args.val_ratio,
        start_epoch=args.epoch_start,
        missing_epoch_policy=args.missing_epoch_policy,
    )

    print(f"Trajectory selection: {full_dataset.trajectory_summary}")

    # The dataset exposes only validated base-training sample IDs.
    train_dataset = full_dataset
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )

    # Initialize model, optimizer, and loss function
    model = ConditionedPredictor(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 5
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (images, labels, margins) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            margins = margins.to(device)

            optimizer.zero_grad()
            outputs = model(images, labels)
            loss = criterion(outputs, margins)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

    # Save the model
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/cond_head_{setup_id}_seed{args.seed}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")


if __name__ == "__main__":
    main()
