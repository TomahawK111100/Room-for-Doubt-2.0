import os, json, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision, torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import argparse
from src.utils import set_seed

class CFMVectorField(nn.Module):
    def __init__(self, num_classes=100, feature_dim=512, traj_dim=50):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 32)
        self.mlp = nn.Sequential(
            nn.Linear(traj_dim + 1 + feature_dim + 32, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, traj_dim)
        )
    def forward(self, x_t, t, img_features, class_labels):
        c_emb = self.class_emb(class_labels)
        return self.mlp(torch.cat([x_t, t.unsqueeze(1), img_features, c_emb], dim=1))

class CFMTrajectoryModel(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
        for p in self.features.parameters(): p.requires_grad = False
        self.vector_field = CFMVectorField(num_classes=num_classes)
    def extract_features(self, img):
        with torch.no_grad(): return torch.flatten(self.features(img), 1)

class CondTrajDataset(Dataset):
    def __init__(self, dataset_name="cifar100", root_dir="./data", traj_file="", transform=None):
        self.cifar = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=False, transform=transform)
        with open(traj_file, 'r') as f: self.trajectories = json.load(f)
    def __len__(self): return len(self.cifar)
    def __getitem__(self, idx):
        img, _ = self.cifar[idx]
        data = self.trajectories.get(str(idx), {})
        margins = data.get('margins', [])[:50]
        margins += [0.0] * max(0, 50 - len(margins))
        return img, torch.tensor(data.get('noisy_label', 0), dtype=torch.long), torch.tensor(margins, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = CondTrajDataset(dataset_name=args.dataset, traj_file=f"results/stage1_trajectories_{args.dataset}_seed{args.seed}.json", transform=transform)
    loader = DataLoader(Subset(dataset, range(45000)), batch_size=128, shuffle=True, num_workers=2)
    
    model = CFMTrajectoryModel(num_classes=100).to(device)
    optimizer = optim.Adam(model.vector_field.parameters(), lr=1e-3)
    
    print("Starting Flow Matching training on CIFAR-100N...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for imgs, labels, x_1 in loader:
            imgs, labels, x_1 = imgs.to(device), labels.to(device), x_1.to(device)
            img_features = model.extract_features(imgs)
            t = torch.rand(imgs.size(0), device=device)
            x_0 = torch.randn_like(x_1)
            x_t = (1 - t.unsqueeze(1)) * x_0 + t.unsqueeze(1) * x_1
            
            optimizer.zero_grad()
            loss = nn.MSELoss()(model.vector_field(x_t, t, img_features, labels), x_1 - x_0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10 Loss: {total_loss/len(loader):.4f}")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/cfm_head_{args.dataset}_seed{args.seed}.pth")
