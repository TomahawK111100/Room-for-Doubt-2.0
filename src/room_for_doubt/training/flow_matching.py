from room_for_doubt.data.trajectory_records import historical_trajectory_path
from room_for_doubt.models.flow_matching import CFMTrajectoryModel
from room_for_doubt.data.flow_trajectories import CondTrajDataset
import os, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
from room_for_doubt.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100"])
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CondTrajDataset(
        dataset_name=args.dataset,
        traj_file=args.trajectories_path
        or historical_trajectory_path(args.dataset, args.seed),
        seed=args.seed,
        val_ratio=args.val_ratio,
        start_epoch=args.epoch_start,
        missing_epoch_policy=args.missing_epoch_policy,
        transform=transform,
    )
    print(f"Trajectory selection: {dataset.trajectory_summary}")
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    model = CFMTrajectoryModel(num_classes=100).to(device)
    optimizer = optim.Adam(model.vector_field.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    print("Starting Flow Matching training on CIFAR-100N...")
    for epoch in range(25):
        model.train()
        total_loss = 0
        for imgs, labels, x_1 in loader:
            imgs, labels, x_1 = imgs.to(device), labels.to(device), x_1.to(device)
            img_features = model.extract_features(imgs)
            t = torch.rand(imgs.size(0), device=device)
            x_0 = torch.randn_like(x_1)
            x_t = (1 - t.unsqueeze(1)) * x_0 + t.unsqueeze(1) * x_1

            optimizer.zero_grad()
            loss = nn.MSELoss()(
                model.vector_field(x_t, t, img_features, labels), x_1 - x_0
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/25 Loss: {total_loss / len(loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        model.state_dict(), f"checkpoints/cfm_head_{args.dataset}_seed{args.seed}.pth"
    )


if __name__ == "__main__":
    main()
