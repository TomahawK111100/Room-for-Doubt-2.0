import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

class CondTrajDataset(Dataset):
    def __init__(self, root_dir='./data', traj_file='results/stage1_trajectories.json', transform=None):
        """
        Dataset that loads CIFAR-10 images alongside their trajectory margins and noisy labels.
        """
        self.cifar = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
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
    def __init__(self):
        super().__init__()
        # Frozen resnet18 without the final fc layer
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze resnet parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Embedding for the class label
        self.class_emb = nn.Embedding(10, 32)
        
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
    full_dataset = CondTrajDataset(transform=transform)
    
    # Use indices 0-45000 for training
    train_dataset = Subset(full_dataset, range(45000))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    
    # Initialize model, optimizer, and loss function
    model = ConditionedPredictor().to(device)
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
    save_path = 'checkpoints/cond_head.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == '__main__':
    main()
