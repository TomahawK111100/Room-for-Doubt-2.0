import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

class DirectCertaintyDataset(Dataset):
    def __init__(self, root_dir='./data', traj_file='results/stage1_trajectories.json', transform=None):
        """
        Dataset that loads CIFAR-10 images alongside their noisy labels and computes 'Badness'.
        Badness is defined as 1 if the noisy label is incorrect, 0 otherwise.
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
            clean_label = data.get('true_label', data.get('clean_label', 0))
            # Badness: 1.0 if the noisy label doesn't match the clean label (i.e. it's a bad prediction)
            badness = 1.0 if noisy_label != clean_label else 0.0
        else:
            noisy_label = 0
            badness = 0.0
            
        return image, torch.tensor(noisy_label, dtype=torch.long), torch.tensor(badness, dtype=torch.float32)

class DirectCertaintyEstimator(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Frozen resnet18 without the final fc layer
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze resnet parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Embedding for the class label
        self.class_emb = nn.Embedding(10, 32)
        
        # MLP for predicting the scalar 'Badness' score directly
        self.mlp = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, img, cls):
        # Extract image features
        f = self.features(img)          # [B, 512, 1, 1]
        f = torch.flatten(f, 1)         # [B, 512]
        
        # Extract class embeddings
        c = self.class_emb(cls)         # [B, 32]
        
        # Concatenate and pass through MLP
        x = torch.cat([f, c], dim=1)    # [B, 544]
        out = self.mlp(x)               # [B, 1]
        
        return out.squeeze(-1)          # [B]

    def training_step(self, batch, batch_idx):
        images, labels, badness = batch
        logits = self(images, labels)
        
        loss = self.criterion(logits, badness)
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == badness).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, labels, badness = batch
        logits = self(images, labels)
        
        loss = self.criterion(logits, badness)
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == badness).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Transforms compatible with ImageNet ResNet-18
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = DirectCertaintyDataset(transform=transform)
    
    # Split into train and validation
    train_size = 45000
    val_size = 5000
    
    train_dataset = Subset(full_dataset, range(train_size))
    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Initialize the model
    model = DirectCertaintyEstimator(lr=1e-3)
    
    # Callbacks
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='direct_estimator_best',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=5,
        accelerator="gpu" if device == "cuda" else ("mps" if device == "mps" else "cpu"),
        devices=1,
        callbacks=[checkpoint_callback],
        logger=True,
        log_every_n_steps=50
    )
    
    print("Starting training of Direct Certainty Estimator...")
    trainer.fit(model, train_loader, val_loader)
    print("Training complete!")

if __name__ == '__main__':
    main()
