"""S4 inference graphs shared by timing and external evaluation."""

from pathlib import Path

import torch
from room_for_doubt.data.kaggle_artifacts import restore_base
from room_for_doubt.evaluation.posthoc_runtime import ForwardPlan
from room_for_doubt.training.posthoc_controls import load_head


def make_confirmation_plan(root, bundle, method, device):
    root = Path(root)
    if method == "early5_teacher":
        models = {}
        for epoch in range(1, 6):
            path = root / f"base/checkpoints/temporal/epoch_{epoch:03}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model = restore_base(bundle, "cpu")
            model.load_state_dict(saved["state_dict"], strict=True)
            models[epoch] = model.to(device)
        return ForwardPlan(models, None, "teacher", tuple(models)), len(models)
    model = restore_base(bundle, device)
    if method == "base":
        return ForwardPlan({"base": model}, "base", "base"), 1
    kind = {"early5": "student", "direct_soft5": "direct"}[method]
    head, policy = load_head(root / method, device)
    contract = policy["contract"]
    if (
        contract["base_digest"] != bundle["checkpoint_digest"]
        or contract["branch"] != method
    ):
        raise ValueError("Head does not belong to the frozen base and method")
    return (
        ForwardPlan(
            {"base": model}, "base", kind, alpha=policy["policy"]["alpha"], head=head
        ),
        1,
    )
