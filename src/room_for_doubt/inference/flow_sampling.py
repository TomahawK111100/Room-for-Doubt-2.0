"""Euler integration of class-conditioned flow trajectories."""

import torch


@torch.no_grad()
def sample_flow_trajectories(
    model, features, classes, num_samples, num_steps, generator
):
    """Fixed CPU random stream, usable for common reconstruction validation."""
    features = features.repeat_interleave(num_samples, 0)
    classes = classes.repeat_interleave(num_samples, 0)
    x = torch.randn((len(features), 50), generator=generator).to(features.device)
    for step in range(num_steps):
        t = torch.full((len(features),), step / num_steps, device=features.device)
        x = x + model.vector_field(x, t, features, classes) / num_steps
    return x.reshape(-1, num_samples, 50)


@torch.no_grad()
def generate_trajectories(model, img_features, classes, num_samples):
    B, device = img_features.size(0), img_features.device
    img_features = img_features.repeat_interleave(num_samples, dim=0)
    classes = classes.repeat_interleave(num_samples, dim=0)
    x = torch.randn(B * num_samples, 50, device=device)
    dt = 1.0 / 10
    for step in range(10):
        t_tensor = torch.full((B * num_samples,), step * dt, device=device)
        x = x + model.vector_field(x, t_tensor, img_features, classes) * dt
    return x.view(B, num_samples, 50)
