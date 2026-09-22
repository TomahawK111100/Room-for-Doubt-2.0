import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. Автоматический фикс путей (поднимаемся из src в корень Room for Doubt)
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == 'src':
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
else:
    root_dir = current_dir

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.model import ResNetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
data_dir = os.path.join(root_dir, 'data')

def evaluate_ensemble(dataset_name, num_classes, mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    print(f"\n📂 Используем датасет из папки: {data_dir}")
    if dataset_name == 'cifar10':
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    else:
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)
        
    loader = DataLoader(test_set, batch_size=256, shuffle=False)

    models = []
    for seed in [42, 43, 44, 45, 46]:
        # Ищем чекпоинт сначала в src/checkpoints, затем в корневой checkpoints
        ckpt_name = f"best_{dataset_name}n_worse_seed{seed}.ckpt"
        path_in_src = os.path.join(current_dir, "checkpoints", ckpt_name)
        path_in_root = os.path.join(root_dir, "checkpoints", ckpt_name)
        
        ckpt_path = path_in_src if os.path.exists(path_in_src) else path_in_root

        model = ResNetClassifier.load_from_checkpoint(ckpt_path, map_location=device, num_classes=num_classes)
        model.eval()
        models.append(model)

    print(f"🚀 Считаем предсказания ансамбля для {dataset_name.upper()}...")
    all_labels, all_ensemble_probs = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            all_labels.extend(labels.numpy())
            
            batch_probs = []
            for model in models:
                logits = model(imgs)
                batch_probs.append(F.softmax(logits, dim=1))
                
            ensemble_probs = torch.stack(batch_probs).mean(dim=0)
            all_ensemble_probs.extend(ensemble_probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_ensemble_probs = np.array(all_ensemble_probs)

    preds = all_ensemble_probs.argmax(axis=1)
    acc = (preds == all_labels).mean()

    msp = all_ensemble_probs.max(axis=1)
    errors = (preds != all_labels).astype(int)
    auroc = roc_auc_score(errors, -msp)

    print(f"=== РЕЗУЛЬТАТЫ DEEP ENSEMBLE ({dataset_name.upper()}) ===")
    print(f"Ensemble Accuracy: {acc * 100:.2f}%")
    print(f"Ensemble MSP AUROC: {auroc:.4f}")

# Запускаем для CIFAR-10
evaluate_ensemble('cifar10', 10, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Запускаем для CIFAR-100
evaluate_ensemble('cifar100', 100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))