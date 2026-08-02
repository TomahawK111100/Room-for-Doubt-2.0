import argparse
import os
import json
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import ResNetClassifier
from src.train_conditioned_head import ConditionedPredictor
from src.utils import set_seed

def calculate_aurc(confidences, errors):
    """
    Calculate Area Under the Risk-Coverage (AURC) curve.
    confidences: array-like of confidence scores (higher is more confident)
    errors: array-like of binary errors (1 if error, 0 if correct)
    """
    confidences = np.array(confidences)
    errors = np.array(errors)
    
    # Sort descending by confidence
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_errors = errors[sorted_indices]
    
    # Calculate cumulative error rate (risk)
    coverages = np.arange(1, len(sorted_errors) + 1)
    risks = np.cumsum(sorted_errors) / coverages
    
    # AURC is the mean of risks
    return np.mean(risks)

class DualTransformCIFAR(Dataset):
    def __init__(self, base_dataset, cifar_mean, cifar_std):
        self.base_dataset = base_dataset

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
    parser = argparse.ArgumentParser(description="Evaluate top-2 correction pipeline.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.dataset == "cifar10":
        dataset_cls = torchvision.datasets.CIFAR10
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    else:
        dataset_cls = torchvision.datasets.CIFAR100
        cifar_mean = (0.5071, 0.4867, 0.4408)
        cifar_std = (0.2675, 0.2565, 0.2761)
        num_classes = 100

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Train Badness Detector
    print("Training LDA Badness Detector...")
    trajectory_path = f"results/stage1_trajectories_{args.dataset}_seed{args.seed}.json"
    with open(trajectory_path, 'r') as f:
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
    
    student_ckpt = f"checkpoints/best_{args.dataset}_seed{args.seed}.ckpt"
    if not os.path.exists(student_ckpt):
        raise FileNotFoundError(f"Student checkpoint '{student_ckpt}' not found.")
        
    student = ResNetClassifier.load_from_checkpoint(student_ckpt)
    student.to(device)
    student.eval()

    cond_ckpt = f"checkpoints/cond_head_{args.dataset}_seed{args.seed}.pth"
    if not os.path.exists(cond_ckpt):
        raise FileNotFoundError(f"ConditionedPredictor checkpoint '{cond_ckpt}' not found.")

    predictor = ConditionedPredictor(num_classes=num_classes)
    predictor.load_state_dict(torch.load(cond_ckpt, map_location=device, weights_only=True))
    predictor.to(device)
    predictor.eval()

    # 3. Inference Loop
    print(f"Evaluating on {args.dataset.upper()} test dataset...")

    test_dataset = dataset_cls(root='./data', train=False, download=True)
    dual_dataset = DualTransformCIFAR(test_dataset, cifar_mean=cifar_mean, cifar_std=cifar_std)
    test_loader = DataLoader(dual_dataset, batch_size=128, shuffle=False, num_workers=2)

    baseline_correct = 0
    corrected_correct = 0
    total_samples = 0
    total_corrections = 0
    
    successful_corrections = 0
    harmful_corrections = 0
    neutral_corrections = 0
    
    corrected_badness_c1 = []
    corrected_badness_c2 = []
    
    all_badness_c1 = []
    all_msp = []
    all_entropy = []
    all_margin = []
    all_errors = []

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
            
            # Compute classical uncertainty baselines
            probs = torch.softmax(logits, dim=1)
            msp, _ = torch.max(probs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            top2_probs = torch.topk(probs, k=2, dim=1).values
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            
            # Use ConditionedPredictor to predict trajectories for both classes
            traj_c1 = predictor(img_predictor, c1)
            traj_c2 = predictor(img_predictor, c2)
            
            # Pass both trajectories to the trained LDA predict_proba to get badness scores
            traj_c1_np = traj_c1.cpu().numpy()
            traj_c2_np = traj_c2.cpu().numpy()
            
            # badness is the probability of class 1 (i.e. noisy_label != clean_label)
            badness_c1 = lda.predict_proba(traj_c1_np)[:, 1]
            badness_c2 = lda.predict_proba(traj_c2_np)[:, 1]
            
            all_badness_c1.extend(badness_c1)
            all_msp.extend(msp.cpu().numpy())
            all_entropy.extend(entropy.cpu().numpy())
            all_margin.extend(margin.cpu().numpy())
            all_errors.extend((c1 != labels).cpu().numpy().astype(int))
            
            # 4. Correction Logic
            final_preds = c1.clone()
            
            # If badness_c1 > 0.5 AND badness_c2 < 0.5, change the prediction to c2
            mask = (torch.tensor(badness_c1) > 0.5) & (torch.tensor(badness_c2) < 0.5)
            mask = mask.to(device)
            final_preds[mask] = c2[mask]
            
            mask_correct_c1 = (c1 == labels)
            mask_correct_c2 = (c2 == labels)
            
            successful_corrections += (mask & ~mask_correct_c1 & mask_correct_c2).sum().item()
            harmful_corrections += (mask & mask_correct_c1).sum().item()
            neutral_corrections += (mask & ~mask_correct_c1 & ~mask_correct_c2).sum().item()
            
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
    
    print("\n--- Detailed Correction Tracking ---")
    print(f"Total Corrections Attempted: {total_corrections}")
    print(f"  Successful (Fixed error):  {successful_corrections}")
    print(f"  Harmful (Broke correct):   {harmful_corrections}")
    print(f"  Neutral (Still wrong):     {neutral_corrections}")
    
    auroc_badness = roc_auc_score(all_errors, all_badness_c1)
    auroc_msp = roc_auc_score(all_errors, -np.array(all_msp))
    auroc_entropy = roc_auc_score(all_errors, all_entropy)
    auroc_margin = roc_auc_score(all_errors, -np.array(all_margin))

    print("\n--- Error Detection AUROC (Classical Baselines) ---")
    print(f"Badness c1 (Ours): {auroc_badness:.4f}")
    print(f"MSP:               {auroc_msp:.4f}")
    print(f"Entropy:           {auroc_entropy:.4f}")
    print(f"Margin:            {auroc_margin:.4f}")

    aurc_badness = calculate_aurc(1.0 - np.array(all_badness_c1), all_errors)
    aurc_msp = calculate_aurc(all_msp, all_errors)
    aurc_entropy = calculate_aurc(-np.array(all_entropy), all_errors)
    aurc_margin = calculate_aurc(all_margin, all_errors)

    print("\n--- Error Detection AURC (Lower is better) ---")
    print(f"Badness c1 (Ours): {aurc_badness:.4f}")
    print(f"MSP:               {aurc_msp:.4f}")
    print(f"Entropy:           {aurc_entropy:.4f}")
    print(f"Margin:            {aurc_margin:.4f}")

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
