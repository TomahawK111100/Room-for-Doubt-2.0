import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.model import ResNetClassifier

def calculate_aurc(confidences, errors):
    confidences = np.array(confidences)
    errors = np.array(errors)

    sorted_indices = np.argsort(confidences)[::-1]
    sorted_errors = errors[sorted_indices]

    coverages = np.arange(1, len(sorted_errors) + 1)
    risks = np.cumsum(sorted_errors) / coverages

    return np.mean(risks)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deep Ensemble baseline.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", type=str, default="worse")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    setup_id = f"{args.dataset}n_{args.split}"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    test_dataset = dataset_cls(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    seeds = [42, 43, 44, 45, 46]
    models = []
    
    print("Loading models...")
    for seed in seeds:
        ckpt_path = f"checkpoints/best_{setup_id}_seed{seed}.ckpt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        model = ResNetClassifier.load_from_checkpoint(ckpt_path, num_classes=num_classes)
        model.to(device)
        model.eval()
        models.append(model)
        
    print(f"Loaded {len(models)} models for {args.dataset.upper()}.")

    all_labels = []
    all_individual_probs = {i: [] for i in range(len(models))}
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            all_labels.append(labels.cpu())
            
            for i, model in enumerate(models):
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                all_individual_probs[i].append(probs.cpu())

    all_labels = torch.cat(all_labels)
    individual_probs = [torch.cat(all_individual_probs[i]) for i in range(len(models))]
    
    individual_accs = []
    for i in range(len(models)):
        preds = individual_probs[i].argmax(dim=1)
        acc = (preds == all_labels).float().mean().item()
        individual_accs.append(acc)
        
    avg_individual_acc = np.mean(individual_accs)
    
    stacked_probs = torch.stack(individual_probs, dim=0)
    ensemble_probs = stacked_probs.mean(dim=0)
    ensemble_preds = ensemble_probs.argmax(dim=1)
    
    ensemble_acc = (ensemble_preds == all_labels).float().mean().item()
    
    ensemble_msp, _ = ensemble_probs.max(dim=1)
    ensemble_entropy = -torch.sum(ensemble_probs * torch.log(ensemble_probs + 1e-8), dim=1)
    
    errors = (ensemble_preds != all_labels).numpy().astype(int)
    msp_np = ensemble_msp.numpy()
    entropy_np = ensemble_entropy.numpy()
    
    auroc_msp = roc_auc_score(errors, -msp_np)
    auroc_entropy = roc_auc_score(errors, entropy_np)
    
    aurc_msp = calculate_aurc(msp_np, errors)
    aurc_entropy = calculate_aurc(-entropy_np, errors)
    
    print("\n--- Ensemble Evaluation Results ---")
    print(f"Individual Accuracies: {['{:.2f}%'.format(a*100) for a in individual_accs]}")
    print(f"Average Individual Acc: {avg_individual_acc*100:.2f}%")
    print(f"Ensemble Accuracy:      {ensemble_acc*100:.2f}%")
    print("\n--- Uncertainty Metrics (Error Detection) ---")
    print(f"MSP AUROC:     {auroc_msp:.4f}")
    print(f"Entropy AUROC: {auroc_entropy:.4f}")
    print(f"MSP AURC:      {aurc_msp:.4f}")
    print(f"Entropy AURC:  {aurc_entropy:.4f}")
    
    os.makedirs("results", exist_ok=True)
    results_path = f"results/ensemble_metrics_{setup_id}.json"
    
    results = {
        "dataset": args.dataset,
        "noisy_split": args.split,
        "num_models": len(models),
        "individual_accs": individual_accs,
        "avg_individual_acc": avg_individual_acc,
        "ensemble_acc": ensemble_acc,
        "error_detection": {
            "msp": {
                "auroc": float(auroc_msp),
                "aurc": float(aurc_msp)
            },
            "entropy": {
                "auroc": float(auroc_entropy),
                "aurc": float(aurc_entropy)
            }
        }
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
