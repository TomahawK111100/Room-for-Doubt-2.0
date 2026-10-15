from room_for_doubt.data.trajectory_records import (
    historical_trajectory_path,
    load_verified_trajectories,
)
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.ensemble import RandomForestClassifier

from room_for_doubt.models.classifier import ResNetClassifier
from room_for_doubt.models.flow_matching import CFMTrajectoryModel
from room_for_doubt.utils.reproducibility import set_seed


class FastTTADataset(Dataset):
    def __init__(self, base_dataset, K=20):
        self.base = base_dataset
        self.K = K

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img, label = item[0], item[1] if len(item) == 2 else item[1]

        # Base view (view 0)
        views = [transforms.functional.to_tensor(img)]

        # 19 augmented views (Random Crop + Horizontal Flip)
        for _ in range(self.K - 1):
            pil_view = img
            if torch.rand(1).item() > 0.5:
                pil_view = transforms.functional.hflip(pil_view)
            pil_view = transforms.functional.pad(pil_view, 4, padding_mode="reflect")
            i, j, h, w = transforms.RandomCrop.get_params(
                pil_view, output_size=(32, 32)
            )
            pil_view = transforms.functional.crop(pil_view, i, j, h, w)
            views.append(transforms.functional.to_tensor(pil_view))

        return torch.stack(views), label  # [K, 3, 32, 32] in [0, 1]


from room_for_doubt.inference.flow_sampling import generate_trajectories


@torch.no_grad()
def collect_tta_predictions(loader, student, cfm, detector, device, num_samples=5):
    all_probs = []
    all_badness = []
    all_labels = []

    cifar_mean = torch.tensor([0.5071, 0.4867, 0.4408], device=device).view(1, 3, 1, 1)
    cifar_std = torch.tensor([0.2675, 0.2565, 0.2761], device=device).view(1, 3, 1, 1)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for batch_idx, (views, labels) in enumerate(loader):
        B, K = views.size(0), views.size(1)
        views = views.to(device)  # [B, K, 3, 32, 32]
        views_flat = views.view(B * K, 3, 32, 32)

        # 1. Student inference
        v_s = (views_flat - cifar_mean) / cifar_std
        logits = student(v_s)
        probs = F.softmax(logits, dim=1)  # [B*K, 100]
        pred_classes = probs.argmax(dim=1)  # [B*K]

        # 2. Fast GPU interpolation & normalization for CFM
        v_p = F.interpolate(
            views_flat, size=(224, 224), mode="bilinear", align_corners=False
        )
        v_p = (v_p - imagenet_mean) / imagenet_std
        feats = cfm.extract_features(v_p)  # [B*K, 512]

        # 3. Trajectory generation
        trajs = (
            generate_trajectories(cfm, feats, pred_classes, num_samples=num_samples)
            .cpu()
            .numpy()
        )

        # 4. Badness detection
        trajs_flat = trajs.reshape(-1, 50)
        badness_flat = detector.predict_proba(trajs_flat)[:, 1]
        badness = badness_flat.reshape(B * K, num_samples).mean(axis=1)

        all_probs.append(probs.view(B, K, 100).cpu())
        all_badness.append(torch.tensor(badness, dtype=torch.float32).view(B, K))
        all_labels.append(labels)

        del views, views_flat, v_s, v_p, feats, trajs, trajs_flat, badness_flat, badness
        if device.type == "mps":
            torch.mps.empty_cache()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")

    all_probs = torch.cat(all_probs, dim=0)
    all_badness = torch.cat(all_badness, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_probs, all_badness, all_labels


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Uncertainty-Weighted TTA on CIFAR-100N."
    )
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_augmentations", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--trajectories-path",
        default=None,
        help="Verified trajectory file; default uses audited historical filenames",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epoch-start", type=int, default=0)
    parser.add_argument(
        "--missing-epoch-policy", choices=["drop", "error"], default="drop"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 1. Train Random Forest Badness Detector on Stage 1 verified base-training trajectories
    print("Loading Stage 1 trajectories for detector training...")
    trajectory_path = args.trajectories_path or historical_trajectory_path(
        args.dataset, args.seed
    )
    identity_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=False
    )
    verified = load_verified_trajectories(
        trajectory_path,
        args.dataset,
        identity_dataset.targets,
        args.seed,
        args.val_ratio,
    )
    print("Detector supervision: oracle clean TRAIN labels")
    included, _, excluded = verified.aligned_targets(
        start_epoch=args.epoch_start, missing_policy=args.missing_epoch_policy
    )
    print(
        f"Trajectory selection: {len(included)} complete train samples, {len(excluded)} excluded; epochs {args.epoch_start}–{args.epoch_start + 49}"
    )
    X, y = verified.oracle_detector_arrays(
        start_epoch=args.epoch_start, missing_policy=args.missing_epoch_policy
    )

    print("Fitting Random Forest Badness Detector (n_estimators=100, max_depth=10)...")
    detector = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ).fit(np.array(X), np.array(y))

    # 2. Load models
    print("Loading ResNet-34 student and CFM models...")
    student = (
        ResNetClassifier.load_from_checkpoint(
            f"checkpoints/best_{args.dataset}n_worse_seed{args.seed}.ckpt",
            num_classes=100,
        )
        .to(device)
        .eval()
    )

    cfm = CFMTrajectoryModel(num_classes=100).to(device)
    cfm.load_state_dict(
        torch.load(
            f"checkpoints/cfm_head_{args.dataset}_seed{args.seed}.pth",
            map_location=device,
            weights_only=True,
        )
    )
    cfm.eval()

    # 3. Validation Phase (T tuning)
    print(
        f"\n--- Phase 1: Validation Set Sweep (Verified base-validation IDs, K={args.num_augmentations}) ---"
    )
    val_dataset = Subset(
        torchvision.datasets.CIFAR100(root="./data", train=True, download=False),
        verified.val_ids.tolist(),
    )
    val_tta_dataset = FastTTADataset(val_dataset, K=args.num_augmentations)
    val_loader = DataLoader(
        val_tta_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    val_probs, val_badness, val_labels = collect_tta_predictions(
        val_loader, student, cfm, detector, device, num_samples=5
    )

    val_base_preds = val_probs[:, 0, :].argmax(dim=1)
    val_acc_base = (val_base_preds == val_labels).float().mean().item() * 100
    print(f"\nValidation Baseline (No TTA):   {val_acc_base:.2f}%")

    val_std_preds = val_probs.mean(dim=1).argmax(dim=1)
    val_acc_std = (val_std_preds == val_labels).float().mean().item() * 100
    print(
        f"Validation Standard TTA:        {val_acc_std:.2f}% (Gain vs Base: {val_acc_std - val_acc_base:+.2f}%)"
    )

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_T = 1.0
    best_val_acc_uw = 0.0

    print("\n--- Sweeping Temperature T for UW-TTA ---")
    for T in temperatures:
        weights = F.softmax(-val_badness / T, dim=1).unsqueeze(-1)
        weighted_probs = (val_probs * weights).sum(dim=1)
        acc_uw = (
            weighted_probs.argmax(dim=1) == val_labels
        ).float().mean().item() * 100
        print(
            f"  T = {T:4.1f} | Val UW-TTA Acc = {acc_uw:.2f}% (vs Base: {acc_uw - val_acc_base:+.2f}%, vs Std: {acc_uw - val_acc_std:+.2f}%)"
        )
        if acc_uw > best_val_acc_uw:
            best_val_acc_uw = acc_uw
            best_T = T

    print(
        f"\n>>> Selected Optimal Temperature T = {best_T:.1f} (Val UW-TTA: {best_val_acc_uw:.2f}%)"
    )

    # 4. Test Set Evaluation
    print(f"\n--- Phase 2: Test Set Evaluation (K={args.num_augmentations}) ---")
    test_base_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=False
    )
    test_tta_dataset = FastTTADataset(test_base_dataset, K=args.num_augmentations)
    test_loader = DataLoader(
        test_tta_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    test_probs, test_badness, test_labels = collect_tta_predictions(
        test_loader, student, cfm, detector, device, num_samples=5
    )

    test_base_preds = test_probs[:, 0, :].argmax(dim=1)
    test_acc_base = (test_base_preds == test_labels).float().mean().item() * 100

    test_std_preds = test_probs.mean(dim=1).argmax(dim=1)
    test_acc_std = (test_std_preds == test_labels).float().mean().item() * 100

    test_weights = F.softmax(-test_badness / best_T, dim=1).unsqueeze(-1)
    test_weighted_probs = (test_probs * test_weights).sum(dim=1)
    test_acc_uw = (
        test_weighted_probs.argmax(dim=1) == test_labels
    ).float().mean().item() * 100

    print("\n================ FINAL TTA RESULTS (CIFAR-100N) ================")
    print(f"1. Baseline Accuracy (No TTA):   {test_acc_base:.2f}%")
    print(
        f"2. Standard TTA (Equal Weights): {test_acc_std:.2f}% (Gain vs Base: {test_acc_std - test_acc_base:+.2f}%)"
    )
    print(
        f"3. UW-TTA (Our Method, T={best_T:.1f}):  {test_acc_uw:.2f}% (Gain vs Base: {test_acc_uw - test_acc_base:+.2f}%, Gain vs Std: {test_acc_uw - test_acc_std:+.2f}%)"
    )
    print("=================================================================")

    os.makedirs("outputs/evaluations", exist_ok=True)
    output_path = f"outputs/evaluations/tta_metrics_cifar100n_seed{args.seed}.json"
    os.makedirs("outputs/evaluations", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "num_augmentations": args.num_augmentations,
                "best_temperature": best_T,
                "val_baseline_acc": val_acc_base,
                "val_std_tta_acc": val_acc_std,
                "val_uw_tta_acc": best_val_acc_uw,
                "test_baseline_acc": test_acc_base,
                "test_std_tta_acc": test_acc_std,
                "test_uw_tta_acc": test_acc_uw,
                "gain_vs_base": test_acc_uw - test_acc_base,
                "gain_vs_std": test_acc_uw - test_acc_std,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
