"""Class-complete residual posterior on frozen features (experimental)."""

import torch
from torch import nn
from torch.nn import functional as F


class TemporalResidualHead(nn.Module):
    def __init__(self, mean, scale, classes, hidden=256):
        super().__init__()
        mean, scale = torch.as_tensor(mean).float(), torch.as_tensor(scale).float()
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError(
                "Finite feature normalization with positive scale required"
            )
        if classes < 2 or hidden < 1:
            raise ValueError("Invalid residual head dimensions")
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.net = nn.Sequential(
            nn.Linear(len(mean), hidden), nn.ReLU(), nn.Linear(hidden, classes)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        value = self.net((features - self.mean) / self.scale)
        return value - value.mean(-1, keepdim=True)


def residual_posterior(base, residual, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Residual alpha must be in [0,1]")
    if base.shape != residual.shape or not torch.isfinite(residual).all():
        raise ValueError("Aligned finite class residuals required")
    if alpha == 0:
        return base.clone()  # Exact identity policy, including numerical precision.
    return F.softmax(base.clamp_min(1e-12).log() + alpha * residual, dim=-1)


def distillation_kl(base, residual, teacher):
    if teacher.shape != base.shape:
        raise ValueError("Teacher and student class coordinates differ")
    logp = F.log_softmax(base.clamp_min(1e-12).log() + residual, dim=-1)
    return F.kl_div(logp, teacher, reduction="batchmean")
