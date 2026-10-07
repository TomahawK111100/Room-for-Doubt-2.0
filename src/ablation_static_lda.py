import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Загружаем статический ResNet-18 (как в нашем мета-моделе)
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    features_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    features_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Берем тренировочную выборку (0-45000), как при обучении нашего LDA
    train_subset = Subset(dataset, range(45000))
    loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=4)

    with open("results/stage1_trajectories.json", 'r') as f:
        trajectories = json.load(f)

    static_features = []
    labels = [] # 1 если шум, 0 если чистый

    print("Extracting static features...")
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            feats = features_extractor(imgs).squeeze().cpu().numpy()
            
            for j in range(len(imgs)):
                idx = str(i * 256 + j)
                if idx in trajectories:
                    noisy = trajectories[idx].get('noisy_label', 0)
                    clean = trajectories[idx].get('true_label', trajectories[idx].get('clean_label', 0))
                    labels.append(1 if noisy != clean else 0)
                    static_features.append(feats[j])

    X = np.array(static_features)
    y = np.array(labels)

    print("Training Static LDA...")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    train_acc = lda.score(X, y)
    preds_proba = lda.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, preds_proba)

    print("\n=== ABLATION RESULTS ===")
    print(f"Static Features LDA Accuracy: {train_acc * 100:.2f}%")
    print(f"Static Features LDA AUROC:    {auroc:.4f}")
    print("Compare this to Trajectory LDA Accuracy: ~80.30%")
    print("Если статика сильно хуже, значит темпоральная динамика критически важна!")

if __name__ == '__main__':
    main()