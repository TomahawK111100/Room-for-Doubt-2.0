"""FP32 deployment reference: share a classifier's features across view scoring."""

import numpy as np
import torch


class FrozenViewPredictor(torch.nn.Module):
    def __init__(self, base, head=None, projection=None):
        super().__init__()
        self.base, self.head = base, head
        if projection is None:
            self.register_buffer("center", None)
            self.register_buffer("components", None)
        else:
            self.register_buffer(
                "center",
                torch.as_tensor(np.array(projection["mean"]), dtype=torch.float32),
            )
            self.register_buffer(
                "components",
                torch.as_tensor(
                    np.array(projection["components"]), dtype=torch.float32
                ),
            )

    @torch.inference_mode()
    def forward(self, images, method, temperature="uniform"):
        if images.ndim != 5 or method not in {"base", "uniform", "early"}:
            raise ValueError(
                "Require [images,views,channels,height,width] and a known method"
            )
        n, views = images.shape[:2]
        if not n or not views or (method == "base" and views != 1):
            raise ValueError("Base uses one view; batches must be nonempty")
        use_head = method == "early" and views > 1 and temperature != "uniform"
        if use_head and (
            self.head is None
            or self.components is None
            or not np.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError(
                "Weighted inference requires the fitted head/PCA and positive temperature"
            )
        captured = []
        hook = (
            self.base.fc.register_forward_pre_hook(
                lambda module, args: captured.append(args[0])
            )
            if use_head
            else None
        )
        try:
            p = self.base(images.flatten(0, 1)).float().softmax(-1)
        finally:
            if hook is not None:
                hook.remove()
        probabilities = p.reshape(n, views, -1)
        if not use_head:
            return probabilities.mean(1)
        if len(captured) != 1:
            raise ValueError("Expected one base feature tensor")
        features = (captured[0] - self.center) @ self.components.T
        paths = self.head.predict_from_features(features, p.argmax(-1))
        badness = ((1 - paths.mean(-1)) / 2).clamp(0, 1).reshape(n, views)
        weights = (-badness / temperature).softmax(1)
        return (probabilities * weights[..., None]).sum(1)
