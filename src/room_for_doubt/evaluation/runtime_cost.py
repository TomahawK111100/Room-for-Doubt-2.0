"""Synchronized wall-clock timings; CPU/MPS peak memory is intentionally absent."""

import time
import numpy as np
import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def measure(function, device, warmup, repetitions, images):
    if warmup < 1 or repetitions < 2 or images < 1:
        raise ValueError(
            "Require warmup >= 1, repetitions >= 2 and a positive image count"
        )
    for _ in range(warmup):
        function()
        synchronize(device)
    resident = peak = None
    if device.type == "cuda":
        resident = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repetitions):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
    median = float(np.median(timings))
    return dict(
        batch_ms_median=median,
        batch_ms_p95=float(np.quantile(timings, 0.95)),
        amortized_ms_per_image=median / images,
        images_per_second=1000 * images / median,
        resident_cuda_allocated_mb=resident / 1e6 if resident is not None else None,
        peak_cuda_allocated_mb=peak / 1e6 if peak is not None else None,
        extra_peak_cuda_allocated_mb=(peak - resident) / 1e6
        if peak is not None
        else None,
    ), np.array(timings)
