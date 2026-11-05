"""Opt-in auxiliary teachers; never change best-validation checkpoint selection."""

import hashlib
import json
from pathlib import Path
import torch


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def snapshot_epochs(config):
    epochs = list(config.get("temporal_snapshot_epochs", []))
    if any(
        type(e) is not int or not 1 <= e <= config["epochs"] for e in epochs
    ) or epochs != sorted(set(epochs)):
        raise ValueError(
            "Temporal snapshot epochs must be sorted unique one-based epochs"
        )
    return epochs


def save_snapshot(model, config, epoch, directory, atomic_save):
    """Idempotent after an interrupted epoch; reject incompatible existing weights."""
    from room_for_doubt.training.kaggle_base import resume_contract

    if epoch + 1 not in snapshot_epochs(config):
        return
    root = Path(directory) / "temporal"
    root.mkdir(exist_ok=True)
    path = root / f"epoch_{epoch + 1:03d}.pt"
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=False)
        if (
            resume_contract(old["config"]) != resume_contract(config)
            or old["epoch"] != epoch
            or set(old["state_dict"]) != set(state)
            or any(not torch.equal(old["state_dict"][k], v) for k, v in state.items())
        ):
            raise ValueError("Existing temporal snapshot differs; use a new run")
    else:
        atomic_save(
            dict(
                state_dict=state,
                epoch=epoch,
                config=config,
                role="auxiliary_teacher_only",
            ),
            path,
        )
    entries = []
    for e in snapshot_epochs(config):
        item = root / f"epoch_{e:03d}.pt"
        if item.exists():
            entries.append(dict(epoch=e, file=item.name, sha256=file_digest(item)))
    target = root / "manifest.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            dict(schema=1, epochs=snapshot_epochs(config), checkpoints=entries),
            indent=2,
        )
    )
    temporary.replace(target)


def verify_snapshots(source, config, required_epochs):
    """No scalar-history reconstruction or silent fallback to best/last weights."""
    from room_for_doubt.training.kaggle_base import resume_contract

    required = list(required_epochs)
    if not required or required != sorted(set(required)):
        raise ValueError("Require explicit ordered unique teacher epochs")
    root = Path(source) / "checkpoints/temporal"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest["epochs"] != snapshot_epochs(config):
        raise ValueError("Temporal snapshot manifest/config mismatch")
    entries = {row["epoch"]: row for row in manifest["checkpoints"]}
    if len(entries) != len(manifest["checkpoints"]):
        raise ValueError("Duplicate temporal snapshot epoch")
    result = []
    for e in required:
        if type(e) is not int or e not in snapshot_epochs(config) or e not in entries:
            raise ValueError(f"Missing real temporal snapshot for epoch {e}")
        item = entries[e]
        if item["file"] != f"epoch_{e:03d}.pt":
            raise ValueError("Unexpected temporal snapshot filename")
        path = root / item["file"]
        if file_digest(path) != item["sha256"]:
            raise ValueError("Temporal snapshot SHA256 mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint["epoch"] != e - 1
            or checkpoint.get("role") != "auxiliary_teacher_only"
            or resume_contract(checkpoint["config"]) != resume_contract(config)
        ):
            raise ValueError("Temporal snapshot identity/config mismatch")
        result.append(dict(epoch=e, path=str(path), sha256=item["sha256"]))
    return result
