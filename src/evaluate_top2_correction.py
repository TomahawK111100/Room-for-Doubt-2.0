import os
import json
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import ResNetClassifier
from src.train_conditioned_head import ConditionedPredictor

class DualTransformCIFAR10(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
        # CIFAR-10 stats for ResNetClassifier
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std = (0.2023, 0.1994, 0.2010)
        
        # ImageNet stats for ConditionedPredictor (ResNet-18)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        
        self.transform_student = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std)
        ])
        self.transform_predictor = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform_student(img), self.transform_predictor(img), label

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Train Badness Detector
    print("Training LDA Badness Detector...")
    with open('results/stage1_trajectories.json', 'r') as f:
        trajectories = json.load(f)

    X = []
    y = []
    for k, v in trajectories.items():
        margins = v.get('margins', [])
        # Pad or truncate to 50
        if len(margins) > 50:
            margins = margins[:50]
        else:
            margins = margins + [0.0] * (50 - len(margins))
        
        X.append(margins)
        
        noisy = v.get('noisy_label', 0)
        clean = v.get('true_label', v.get('clean_label', 0))
        y.append(1 if noisy != clean else 0)

    X = np.array(X)
    y = np.array(y)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    print("LDA trained successfully.")

    # 2. Load Models
    print("Loading models...")
    
    student_ckpt = 'checkpoints/best.ckpt'
    if not os.path.exists(student_ckpt):
        print(f"Error: Student checkpoint '{student_ckpt}' not found.")
        return
        
    student = ResNetClassifier.load_from_checkpoint(student_ckpt)
    student.to(device)
    student.eval()

    cond_ckpt = 'checkpoints/cond_head.pth'
    if not os.path.exists(cond_ckpt):
        print(f"Error: ConditionedPredictor checkpoint '{cond_ckpt}' not found.")
        return

    predictor = ConditionedPredictor()
    predictor.load_state_dict(torch.load(cond_ckpt, map_location=device, weights_only=True))
    predictor.to(device)
    predictor.eval()

    # 3. Inference Loop
    print("Evaluating on CIFAR-10 Test Dataset...")
    
    test_dataset = CIFAR10(root='./data', train=False, download=True)
    dual_dataset = DualTransformCIFAR10(test_dataset)
    test_loader = DataLoader(dual_dataset, batch_size=128, shuffle=False, num_workers=2)

    baseline_correct = 0
    corrected_correct = 0
    total_samples = 0
    total_corrections = 0
    
    corrected_badness_c1 = []
    corrected_badness_c2 = []

    with torch.no_grad():
        for img_student, img_predictor, labels in test_loader:
            img_student = img_student.to(device)
            img_predictor = img_predictor.to(device)
            labels = labels.to(device)
            
            # Get logits from Student
            logits = student(img_student)
            
            # Find Top-1 (c1) and Top-2 (c2) predicted classes
            top2 = torch.topk(logits, k=2, dim=1)
            c1 = top2.indices[:, 0]
            c2 = top2.indices[:, 1]
            
            # Use ConditionedPredictor to predict trajectories for both classes
            traj_c1 = predictor(img_predictor, c1)
            traj_c2 = predictor(img_predictor, c2)
            
            # Pass both trajectories to the trained LDA predict_proba to get badness scores
            traj_c1_np = traj_c1.cpu().numpy()
            traj_c2_np = traj_c2.cpu().numpy()
            
            # badness is the probability of class 1 (i.e. noisy_label != clean_label)
            badness_c1 = lda.predict_proba(traj_c1_np)[:, 1]
            badness_c2 = lda.predict_proba(traj_c2_np)[:, 1]
            
            # 4. Correction Logic
            final_preds = c1.clone()
            
            # If badness_c1 > 0.5 AND badness_c2 < 0.5, change the prediction to c2
            mask = (torch.tensor(badness_c1) > 0.5) & (torch.tensor(badness_c2) < 0.5)
            mask = mask.to(device)
            final_preds[mask] = c2[mask]
            
            mask_np = mask.cpu().numpy()
            corrected_badness_c1.extend(badness_c1[mask_np])
            corrected_badness_c2.extend(badness_c2[mask_np])
            
            total_corrections += mask.sum().item()
            baseline_correct += (c1 == labels).sum().item()
            corrected_correct += (final_preds == labels).sum().item()
            total_samples += labels.size(0)

    # 5. Metrics
    baseline_acc = baseline_correct / total_samples
    corrected_acc = corrected_correct / total_samples
    acc_gain = corrected_acc - baseline_acc
    
    print("\n--- Evaluation Results ---")
    print(f"Baseline Accuracy (Top-1): {baseline_acc * 100:.2f}%")
    print(f"Corrected Accuracy:        {corrected_acc * 100:.2f}%")
    print(f"Absolute Accuracy Gain:    {acc_gain * 100:.2f}%")
    print(f"Total Corrections Made:    {total_corrections}")

    # 6. Visualization
    os.makedirs('plots', exist_ok=True)
    
    # Bar chart for Accuracy Gain
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Baseline', 'Corrected'], [baseline_acc * 100, corrected_acc * 100], color=['#1f77b4', '#2ca02c'])
    plt.ylabel('Accuracy (%)')
    plt.title('Baseline vs Corrected Accuracy')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
    plt.savefig('plots/top2_accuracy_gain.png')
    plt.close()
    
    # KDE plot for Badness distributions of corrected samples
    if len(corrected_badness_c1) > 0:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(corrected_badness_c1, fill=True, label='Badness c1 (Original Top-1)')
        sns.kdeplot(corrected_badness_c2, fill=True, label='Badness c2 (New Top-2)')
        plt.xlabel('Badness Score (Probability of being noisy)')
        plt.ylabel('Density')
        plt.title('Badness Distribution for Corrected Samples')
        plt.legend()
        plt.savefig('plots/top2_badness_distribution.png')
        plt.close()
    else:
        print("\nNo corrections made; skipping badness distribution plot.")

if __name__ == '__main__':
    main()
