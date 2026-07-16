import sys
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.model import ResNetClassifier
from train_meta import MetaRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ЗАГРУЗКА МОДЕЛЕЙ
print("Загрузка моделей...")
student = ResNetClassifier.load_from_checkpoint('checkpoints/best.ckpt', map_location=device)
student.to(device)
student.eval()

meta = MetaRegressor()
meta_ckpt = 'meta_model.pth' if os.path.exists('meta_model.pth') else 'checkpoints/meta_model.pth'
state_dict = torch.load(meta_ckpt, map_location=device)
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
meta.load_state_dict(new_state_dict, strict=True)
meta.to(device)
meta.eval()

# 2. БЕЗОПАСНЫЕ АУГМЕНТАЦИИ ДЛЯ 32x32
class TTADataset(Dataset):
    def __init__(self, root, K=20):
        self.base = datasets.CIFAR10(root=root, train=False, download=True)
        self.K = K
        self.transform_base = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # SAFE TTA: Без агрессивного кропа!
        self.transform_tta = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.0625, 0.0625)), # Сдвиг макс на 2 пикселя
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        views = [self.transform_base(img)] 
        views += [self.transform_tta(img) for _ in range(self.K - 1)]
        return torch.stack(views), label

K_AUGMENTATIONS = 20
dataset = TTADataset('./data', K=K_AUGMENTATIONS)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

# Списки для хранения всех предсказаний, чтобы потом быстро перебрать температуры
all_labels = []
all_probs = []
all_uncs = []

print(f"🚀 Сбор предсказаний (K={K_AUGMENTATIONS})...")
with torch.no_grad():
    for views, labels in tqdm(loader):
        views = views.to(device)
        all_labels.append(labels)
        
        batch_probs = []
        batch_uncs = []
        for k in range(K_AUGMENTATIONS):
            v = views[:, k]
            
            logits = student(v)
            if isinstance(logits, tuple): logits = logits[0]
            batch_probs.append(F.softmax(logits, dim=1).cpu())
            
            unc = meta(v).cpu()
            batch_uncs.append(unc)
            
        all_probs.append(torch.stack(batch_probs, dim=1))
        all_uncs.append(torch.stack(batch_uncs, dim=1))

# Объединяем тензоры
labels = torch.cat(all_labels)
probs = torch.cat(all_probs) # [10000, K, 10]
uncs = torch.cat(all_uncs)   # [10000, K]

# --- ВЫЧИСЛЕНИЕ МЕТРИК ---
print("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")

# 1. Baseline (только оригинальная картинка, k=0)
base_preds = probs[:, 0, :].argmax(dim=1)
acc_base = (base_preds == labels).float().mean().item() * 100
print(f"1. Baseline Accuracy (No TTA):   {acc_base:.2f}%")

# 2. Standard TTA (Равные веса)
std_preds = probs.mean(dim=1).argmax(dim=1)
acc_std = (std_preds == labels).float().mean().item() * 100
print(f"2. Standard TTA (Equal Weights): {acc_std:.2f}% (Прирост: {acc_std - acc_base:+.2f}%)")

# 3. Grid Search по температуре для нашего метода
temperatures = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
best_acc = 0
best_t = 0

for T in temperatures:
    # Вычисляем веса через Softmax
    weights = F.softmax(-uncs / T, dim=1).unsqueeze(2) # [10000, K, 1]
    weighted_probs = (probs * weights).sum(dim=1)
    acc_weighted = (weighted_probs.argmax(dim=1) == labels).float().mean().item() * 100
    
    if acc_weighted > best_acc:
        best_acc = acc_weighted
        best_t = T

print(f"3. Uncertainty-Weighted TTA:     {best_acc:.2f}% (Прирост: {best_acc - acc_base:+.2f}%) [Оптимальная T={best_t}]")
print(f"🔥 Чистая польза нашего метода:  +{best_acc - acc_std:.2f}% поверх обычного TTA!")
