import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from src.utils import set_seed


class CondTrajDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "cifar10",
        root_dir: str = "./data",
        traj_file: str = "results/stage1_trajectories.json",
        transform=None,
    ):
        """
        Dataset that loads CIFAR images alongside their trajectory margins and noisy labels.
        """
        if dataset_name == "cifar10":
            dataset_cls = torchvision.datasets.CIFAR10
        elif dataset_name == "cifar100":
            dataset_cls = torchvision.datasets.CIFAR100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.cifar = dataset_cls(root=root_dir, train=True, download=True, transform=transform)
        with open(traj_file, 'r') as f:
            self.trajectories = json.load(f)
            
    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, _ = self.cifar[idx]
        
        str_idx = str(idx)
        if str_idx in self.trajectories:
            data = self.trajectories[str_idx]
            noisy_label = data.get('noisy_label', 0)
            margins = data.get('margins', [])
        else:
            noisy_label = 0
            margins = []
            
        # Pad or truncate margins to exactly 50 epochs
        if len(margins) > 50:
            margins = margins[:50]
        else:
            margins = margins + [0.0] * (50 - len(margins))
            
        return image, torch.tensor(noisy_label, dtype=torch.long), torch.tensor(margins, dtype=torch.float32)

class ConditionedPredictor(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Frozen resnet18 without the final fc layer
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze resnet parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Embedding for the class label
        self.class_emb = nn.Embedding(num_classes, 32)
        
        # MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 50)
        )

    def forward(self, img, cls):
        # Extract image features
        f = self.features(img)          # [B, 512, 1, 1]
        f = torch.flatten(f, 1)         # [B, 512]
        
        # Extract class embeddings
        c = self.class_emb(cls)         # [B, 32]
        
        # Concatenate and pass through MLP
        x = torch.cat([f, c], dim=1)    # [B, 544]
        out = self.mlp(x)               # [B, 50]
        return out

def main():
    parser = argparse.ArgumentParser(description="Train conditioned trajectory predictor.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", type=str, default="worse")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    num_classes = 10 if args.dataset == "cifar10" else 100

    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms compatible with ImageNet ResNet-18
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    setup_id = f"{args.dataset}n_{args.split}"
    trajectory_path = f"results/stage1_trajectories_{setup_id}_seed{args.seed}.json"
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(
            f"Trajectory file '{trajectory_path}' not found. Run stage 1 for this dataset and seed first."
        )
    full_dataset = CondTrajDataset(dataset_name=args.dataset, traj_file=trajectory_path, transform=transform)
    
    # Use indices 0-45000 for training
    train_dataset = Subset(full_dataset, range(45000))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    
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
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")
        
    # Save the model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f"checkpoints/cond_head_{setup_id}_seed{args.seed}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == '__main__':
    main()
