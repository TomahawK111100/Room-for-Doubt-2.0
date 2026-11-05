"""First 50 Google classes of WebVision 1.0, in original file-list order.

The official validation set is evaluation-only here. No image downloading,
redistribution, clean-training targets, or guessed folder/class remapping.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

CLASSES = 50
TRAIN_ROWS = 65944
EVALUATION_PER_CLASS = 50
LISTS = {True: "train_filelist_google.txt", False: "val_filelist.txt"}


def records(root, train, check_files=True):
    root = Path(root).resolve()
    rows, seen = [], set()
    for line_number, line in enumerate(
        (root / "info" / LISTS[train]).read_text().splitlines(), 1
    ):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid WebVision file-list row {line_number}")
        name, label = fields
        label = int(label)
        if not 0 <= label < 1000:
            raise ValueError("WebVision label outside original 0..999 classes")
        if label >= CLASSES:
            continue
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            raise ValueError("Invalid or duplicate WebVision image path")
        if train and (not relative.parts or relative.parts[0] != "google"):
            raise ValueError(
                "Require original Google image paths, not a renamed subset"
            )
        seen.add(name)
        path = root / relative if train else root / "val_images_256" / relative
        if check_files and not path.is_file():
            raise FileNotFoundError(f"Missing listed Mini-WebVision image: {path}")
        rows.append((name, label, path))
    labels = np.array([r[1] for r in rows], dtype=np.int64)
    if set(labels) != set(range(CLASSES)):
        raise ValueError("Mini-WebVision must contain every original class 0..49")
    if train and len(rows) != TRAIN_ROWS:
        raise ValueError("Require all65944 original first50 Google training rows")
    if not train and not np.all(
        np.bincount(labels, minlength=CLASSES) == EVALUATION_PER_CLASS
    ):
        raise ValueError("Require all 50 official validation images per class")
    return rows


class FileImages:
    """Lazy RGB decoding; fixed256px square realization for this transfer protocol."""

    def __init__(self, paths):
        self.paths = tuple(Path(p) for p in paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        with Image.open(self.paths[int(index)]) as image:
            return np.asarray(
                image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            ).copy()


class MiniWebVision:
    def __init__(self, root, train=True, download=False):
        # download is compatibility only; dataset terms must be handled by the owner.
        self.root = Path(root).resolve()
        rows = records(self.root, train)
        self.data = FileImages([r[2] for r in rows])
        self.targets = np.array([r[1] for r in rows], dtype=np.int64)
        self.source_paths = tuple(r[0] for r in rows)
        self.source_ids = np.arange(len(rows), dtype=np.int64)
        self.clean_training_targets = None

    def __len__(self):
        return len(self.targets)


def file_manifest(root):
    """Bind original metadata and selected image bytes; test labels remain opaque."""
    root = Path(root).resolve()
    inventory = []
    for train in (True, False):
        metadata = root / "info" / LISTS[train]
        # Training rows need observed labels. Evaluation list is parsed for complete
        # membership, never passed to fitting or checkpoint/policy selection.
        rows = records(root, train)
        digest = hashlib.sha256()
        for name, _, path in rows:
            image_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    image_digest.update(block)
            digest.update(json.dumps([name, image_digest.hexdigest()]).encode() + b"\n")
        inventory.append(
            dict(
                split="train" if train else "evaluation",
                records=len(rows),
                filelist_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                image_bytes_sha256=digest.hexdigest(),
            )
        )
    identity = dict(
        dataset="miniwebvision",
        version="webvision1_google_classes0_49",
        decoded_realization="RGB_PIL_bilinear_square256_v1",
        partitions=inventory,
    )
    return dict(
        **identity,
        manifest_sha256=hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        official_reference_checksums_verified=False,
    )


def verify_identity(raw, config):
    dataset = config["dataset"]
    if (
        not isinstance(raw, MiniWebVision)
        or dataset["num_classes"] != CLASSES
        or dataset.get("native_size") != 256
    ):
        raise ValueError("Mini-WebVision requires the declared 50-class256px adapter")
    if dataset.get("noisy_split") != "google_first50":
        raise ValueError("Mini-WebVision requires original Google first50 labels")
    if (
        not dataset.get("manifest_sha256")
        or file_manifest(config["data_dir"])["manifest_sha256"]
        != dataset["manifest_sha256"]
    ):
        raise ValueError("Mini-WebVision image/metadata identity differs")


def locate_data(root):
    candidates = {
        p.parent.parent
        for p in Path(root).rglob(LISTS[True])
        if p.parent.name == "info" and (p.parent / LISTS[False]).is_file()
    }
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one original-layout WebVision1.0 release, found {len(candidates)}"
        )
    return candidates.pop()
