# Kaggle: stronger CIFAR-100N base

Use **GPU T4 x2** and enable **Internet** in Notebook settings. The launcher uses
both visible GPUs through PyTorch DataParallel. Batch size 128 is global (normally
64 per GPU). Actual GPU memory use and runtime must be measured on Kaggle; local
checks do not substitute for a two-T4 run.

Run in one Python cell:

```python
import os, subprocess, sys
from pathlib import Path

REPO = "https://github.com/TomahawK111100/Room-for-Doubt-2.0.git"
REVISION = "main"  # Replace with a published commit SHA for a pinned run.
ROOT = Path("/kaggle/working/room-for-doubt")
if not ROOT.exists():
    subprocess.run(["git", "clone", REPO, str(ROOT)], check=True)
subprocess.run(["git", "fetch", "origin", REVISION], cwd=ROOT, check=True)
subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=ROOT, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(ROOT)], check=True)
env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
           MPLCONFIGDIR="/kaggle/working/matplotlib")
subprocess.run([sys.executable, str(ROOT / "scripts/kaggle_run.py")],
               cwd=ROOT, env=env, check=True)
```

First run requires no uploaded data, private model or token. The repository is
public. The runner downloads official CIFAR images and noisy annotations, and
public torchvision ImageNet-1K weights. Run ID and settings are in
`configs/kaggle_base.yaml`.

## What runs

ResNet-50 keeps the ImageNet stem and uses 224x224 normalized images. Training
augmentation is CIFAR random crop and horizontal flip before resize. The loss is
cross entropy for epochs 0–4, then generalized cross entropy with q=0.7. SGD,
cosine learning rate, one base seed, 50 epochs. This fixed recipe is a pilot, not
a guarantee of reaching 70%.

The checkpoint selection criterion is always cross entropy on the **observed
noisy validation labels**, even when the training objective changes to GCE.
Clean training labels are never exposed by the training dataset wrapper.
The clean test is evaluated only after completing training and selecting the
checkpoint. Training logs show noisy-validation accuracy, not clean test accuracy.

This strengthens the base with external ImageNet supervision. Do not attribute
its gain to Room for Doubt. To measure the latter, retrain meta heads on the new
trajectories and compare policies on this same base. The historical 32x32
ResNet-34 loader cannot load this ResNet-50 checkpoint.

## Saved output and continuation

All outputs are in `/kaggle/working/rfd_outputs/cifar100n_resnet50_gce_seed42/`:

- `checkpoints/best.pt`: minimum noisy-validation CE model;
- `checkpoints/last.pt`: atomic epoch state, optimizer, scaler, RNG, best model,
  history and all collected margins;
- `artifacts/meta_targets.npz`: completed trajectories and reserved meta splits;
- `artifacts/split_ids.npz`, `val_predictions.npz`, `test_predictions.npz`;
- resolved config, metrics, metadata, environment, history, report and tables.

Use **Save Version with outputs** to preserve artifacts. If the process stops,
rerunning in the same session resumes automatically from the last complete epoch.
The runner pauses after an epoch when its 10.5-hour time budget has elapsed; it
never pretends a partial run is complete. In a new session, attach the previous
notebook output as input and run the same cell: the launcher finds that run's
`checkpoints/last.pt` and copies its directory to working storage. Retain the
original output directory structure. Only attach one matching prior run.

A restart may repeat the interrupted epoch. `last.pt` also contains the selected
best state, allowing recovery if writing `best.pt` and `last.pt` was interrupted.
Keep scientific configuration and code revision fixed across resumes. There is
no need to upload checkpoints to GitHub.

The exported margin is p(observed class) minus the largest competing probability,
recorded on each sample's augmented training forward before its update. Epochs
are explicit. Every training sample appears once per epoch, including the final
partial batch. These are training-time dynamics, not an epoch-end evaluation pass.

## References

- [Torchvision ResNet-50 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- [Generalized cross entropy, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html)
- [Kaggle P100 retirement and T4x2 availability](https://www.kaggle.com/product-announcements/735239)
