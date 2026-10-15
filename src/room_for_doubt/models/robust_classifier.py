"""ImageNet-initialized backbone and generalized cross entropy (Zhang 2018)."""

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


def make_pretrained_classifier(num_classes, weights="IMAGENET1K_V2"):
    model = resnet50(weights=None if weights is None else ResNet50_Weights[weights])
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def robust_loss(logits, labels, q, use_ce=False):
    if not 0 < q <= 1:
        raise ValueError("GCE q must be in (0, 1]")
    logp = F.log_softmax(logits.float(), dim=1).gather(1, labels[:, None]).squeeze(1)
    return -logp.mean() if use_ce else (-torch.expm1(q * logp) / q).mean()


def probability_margin(logits, labels):
    probs = logits.float().softmax(1)
    observed = probs.gather(1, labels[:, None]).squeeze(1)
    other = probs.clone().scatter_(1, labels[:, None], -1).max(1).values
    return observed - other
