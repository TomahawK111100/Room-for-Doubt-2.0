from room_for_doubt.data.trajectory_records import (
    historical_trajectory_path,
    load_verified_trajectories,
)
import torch
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.ensemble import RandomForestClassifier
from room_for_doubt.models.classifier import ResNetClassifier
from room_for_doubt.models.flow_matching import CFMTrajectoryModel
from room_for_doubt.utils.reproducibility import set_seed


class DualTransformCIFAR(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.t_s = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self.t_p = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img, label = item[0], item[1] if len(item) == 2 else item[1]
        return self.t_s(img), self.t_p(img), label


from room_for_doubt.inference.flow_sampling import generate_trajectories


@torch.no_grad()
def evaluate_calibrated_on_loader(
    loader,
    student,
    cfm,
    detector,
    device,
    p_threshold=0.90,
    max_candidates=10,
    num_samples=20,
):
    all_orig_logits = []
    all_labels = []
    all_candidate_masks = []
    all_candidate_badness = []

    for img_s, img_p, labels in loader:
        img_s = img_s.to(device)
        img_p = img_p.to(device)
        B = img_s.size(0)

        logits = student(img_s)  # [B, 100]
        probs = torch.softmax(logits, dim=1)  # [B, 100]

        # Dynamic Candidate Selection (Top-p / Nucleus)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)
        shifted_cumsum = torch.roll(cumsum_probs, shifts=1, dims=1)
        shifted_cumsum[:, 0] = 0.0

        mask = shifted_cumsum < p_threshold
        if max_candidates is not None:
            mask[:, max_candidates:] = False

        feats = cfm.extract_features(img_p)  # [B, 512]

        sample_indices = []
        candidate_classes = []
        for i in range(B):
            cands = sorted_indices[i][mask[i]]
            sample_indices.extend([i] * len(cands))
            candidate_classes.extend(cands.tolist())

        batch_cand_mask = np.zeros((B, 100), dtype=bool)
        batch_cand_badness = np.zeros((B, 100), dtype=np.float32)

        if len(sample_indices) > 0:
            sample_idx_tensor = torch.tensor(sample_indices, device=device)
            cand_cls_tensor = torch.tensor(
                candidate_classes, device=device, dtype=torch.long
            )
            cand_feats = feats[sample_idx_tensor]

            t_c = (
                generate_trajectories(
                    cfm, cand_feats, cand_cls_tensor, num_samples=num_samples
                )
                .cpu()
                .numpy()
            )
            t_c_flat = t_c.reshape(-1, 50)
            badness_flat = detector.predict_proba(t_c_flat)[:, 1]
            badness_cand = badness_flat.reshape(len(sample_indices), num_samples).mean(
                axis=1
            )

            for idx_in_list, (s_idx, c_cls) in enumerate(
                zip(sample_indices, candidate_classes)
            ):
                batch_cand_mask[s_idx, c_cls] = True
                batch_cand_badness[s_idx, c_cls] = badness_cand[idx_in_list]

        all_orig_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        all_candidate_masks.append(batch_cand_mask)
        all_candidate_badness.append(batch_cand_badness)

    all_orig_logits = np.concatenate(all_orig_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_candidate_masks = np.concatenate(all_candidate_masks, axis=0)
    all_candidate_badness = np.concatenate(all_candidate_badness, axis=0)

    return all_orig_logits, all_labels, all_candidate_masks, all_candidate_badness


def compute_adjusted_accuracy(orig_logits, labels, cand_masks, cand_badness, lam):
    adj_logits = orig_logits.copy()
    adj_logits[cand_masks] -= lam * cand_badness[cand_masks]
    preds = adj_logits.argmax(axis=1)
    acc = (preds == labels).mean()
    return preds, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p_threshold", type=float, default=0.90)
    parser.add_argument("--max_candidates", type=int, default=10)
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

    print(f"Loading Stage 1 trajectories for {args.dataset} seed {args.seed}...")
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

    print("Loading student and CFM models...")
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

    print(
        f"\n--- Phase 1: Validation Set Nucleus Evaluation (p={args.p_threshold}, max_cand={args.max_candidates}) ---"
    )
    val_dataset = Subset(
        torchvision.datasets.CIFAR100(root="./data", train=True, download=False),
        verified.val_ids.tolist(),
    )
    val_loader = DataLoader(
        DualTransformCIFAR(val_dataset), batch_size=64, shuffle=False, num_workers=2
    )

    val_logits, val_labels, val_masks, val_badness = evaluate_calibrated_on_loader(
        val_loader,
        student,
        cfm,
        detector,
        device,
        p_threshold=args.p_threshold,
        max_candidates=args.max_candidates,
        num_samples=20,
    )

    _, val_base_acc = compute_adjusted_accuracy(
        val_logits, val_labels, val_masks, val_badness, lam=0.0
    )
    print(f"Validation Baseline Accuracy: {val_base_acc * 100:.2f}%")

    print("\n--- Sweeping Lambda in [1.0, 10.0] ---")
    lambdas = [float(x) for x in range(1, 11)]
    best_lam = 0.0
    best_val_acc = val_base_acc

    for lam in lambdas:
        _, acc = compute_adjusted_accuracy(
            val_logits, val_labels, val_masks, val_badness, lam=lam
        )
        print(
            f"  Lambda = {lam:4.1f} | Val Accuracy = {acc * 100:.2f}% (Diff: {(acc - val_base_acc) * 100:+.2f}%)"
        )
        if acc > best_val_acc:
            best_val_acc = acc
            best_lam = lam

    if best_lam > 0.0:
        print(
            f"\n>>> Optimal Lambda Selected: {best_lam:.1f} (Val Acc: {best_val_acc * 100:.2f}%)"
        )
    else:
        print(
            f"\n>>> No Lambda improved validation baseline ({val_base_acc * 100:.2f}%). Triggering conservative fallback (lambda=0.0)."
        )

    print(f"\n--- Phase 2: Test Set Nucleus Evaluation (p={args.p_threshold}) ---")
    test_loader = DataLoader(
        DualTransformCIFAR(
            torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
        ),
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )

    test_logits, test_labels, test_masks, test_badness = evaluate_calibrated_on_loader(
        test_loader,
        student,
        cfm,
        detector,
        device,
        p_threshold=args.p_threshold,
        max_candidates=args.max_candidates,
        num_samples=20,
    )

    test_base_preds, test_base_acc = compute_adjusted_accuracy(
        test_logits, test_labels, test_masks, test_badness, lam=0.0
    )

    # Evaluate with selected optimal lambda
    eval_lam = best_lam
    test_corr_preds, test_corr_acc = compute_adjusted_accuracy(
        test_logits, test_labels, test_masks, test_badness, lam=eval_lam
    )

    mask_override = test_corr_preds != test_base_preds
    total_overrides = int(mask_override.sum())
    successful_overrides = int(
        (
            mask_override
            & (test_corr_preds == test_labels)
            & (test_base_preds != test_labels)
        ).sum()
    )
    harmful_overrides = int(
        (
            mask_override
            & (test_corr_preds != test_labels)
            & (test_base_preds == test_labels)
        ).sum()
    )

    print("\n================ FINAL EVALUATION RESULTS ================")
    print(f"Selected Lambda:      {eval_lam:.1f}")
    print(f"Baseline Accuracy:    {test_base_acc * 100:.2f}%")
    print(
        f"Corrected Accuracy:   {test_corr_acc * 100:.2f}% (Gain: {(test_corr_acc - test_base_acc) * 100:+.2f}%)"
    )
    print(f"Total Overrides:      {total_overrides}")
    print(f"Successful Overrides: {successful_overrides}")
    print(f"Harmful Overrides:    {harmful_overrides}")

    # Also log diagnostics for unconstrained lambda sweep on test for analysis
    print("\n--- Diagnostic Test Curve Across Lambda (for analysis) ---")
    for lam in [1.0, 2.0, 3.0, 5.0, 10.0]:
        t_preds, t_acc = compute_adjusted_accuracy(
            test_logits, test_labels, test_masks, test_badness, lam=lam
        )
        t_diff = (t_acc - test_base_acc) * 100
        t_ov = int((t_preds != test_base_preds).sum())
        print(
            f"  Diagnostic Test Lambda = {lam:4.1f} | Test Acc = {t_acc * 100:.2f}% ({t_diff:+.2f}%) | Overrides: {t_ov}"
        )
    print("==========================================================")


if __name__ == "__main__":
    main()
