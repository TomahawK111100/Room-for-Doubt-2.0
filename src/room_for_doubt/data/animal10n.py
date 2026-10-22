"""Read the documented KAIST binary release without inventing clean train labels.

Rows are <big-endian int32 ID><int32 label><3x64x64 uint8 pixels>.
Stable internal IDs are offsets in naturally sorted training batch files; train
and test have separate namespaces. Files are memory mapped, not expanded in RAM.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

OFFICIAL_URL = "https://dm.kaist.ac.kr/datasets/animal-10n/"
# SELFIE/reader/batch_patcher.py accumulates bytes as result * 256 + b.
RECORD = np.dtype([("id", ">i4"), ("label", ">i4"), ("pixels", "u1", (3, 64, 64))])
EXPECTED_COUNTS = {True: 50000, False: 5000}


def batch_paths(root, train):
    root = Path(root)
    pattern = "data_batch_*.bin" if train else "test_batch.bin"
    paths = list(root.glob(pattern))
    if train:
        if any(not re.fullmatch(r"data_batch_\d+\.bin", p.name) for p in paths):
            raise ValueError("Unexpected Animal-10N batch filename")
        paths.sort(key=lambda p: int(re.search(r"\d+", p.name).group()))
    if not paths:
        raise FileNotFoundError(
            f"Missing official Animal-10N binary files at {root}. Obtain the release "
            f"through {OFFICIAL_URL}, extract it, and pass its binary directory. "
            "No dataset or noisy-label fallback is used."
        )
    return paths


def file_manifest(root):
    """Hash both official partitions as opaque files, without using test labels."""
    files = []
    for train in (True, False):
        total = 0
        for path in batch_paths(root, train):
            size = path.stat().st_size
            if size == 0 or size % RECORD.itemsize:
                raise ValueError(f"Truncated or incompatible Animal-10N binary: {path}")
            count = size // RECORD.itemsize
            total += count
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files.append(
                dict(
                    split="train" if train else "test",
                    file=path.name,
                    records=count,
                    bytes=size,
                    sha256=digest.hexdigest(),
                )
            )
        if total != EXPECTED_COUNTS[train]:
            raise ValueError(
                f"Animal-10N partition count {total}; expected {EXPECTED_COUNTS[train]}"
            )
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(
        format="kaist_binary_i32_i32_chw64_be_v1",
        files=files,
        manifest_sha256=identity,
        format_reference=OFFICIAL_URL,
        supplied_file_origin_independently_verified=False,
        official_reference_checksums_available=False,
    )


class BinaryImages:
    def __init__(self, paths):
        self.paths = list(paths)
        self.records = [np.memmap(p, dtype=RECORD, mode="r") for p in paths]
        self.ends = np.cumsum([len(r) for r in self.records])

    def __getstate__(self):
        return self.paths

    def __setstate__(self, paths):
        self.__init__(paths)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        part = int(np.searchsorted(self.ends, index, side="right"))
        offset = int(self.ends[part - 1]) if part else 0
        return self.records[part][int(index) - offset]["pixels"].transpose(1, 2, 0)


class Animal10N:
    """Minimal raw-image interface used by the existing base/study pipeline."""

    def __init__(self, root, train=True, download=False):
        # download is API compatibility only: official distribution uses a form.
        paths = batch_paths(root, train)
        if any(p.stat().st_size % RECORD.itemsize for p in paths):
            raise ValueError("Truncated Animal-10N binary record")
        self.data = BinaryImages(paths)
        if len(self.data) != EXPECTED_COUNTS[bool(train)]:
            raise ValueError("Unexpected Animal-10N partition size")
        self.targets = np.concatenate([r["label"] for r in self.data.records]).astype(
            np.int64
        )
        self.source_ids = np.concatenate([r["id"] for r in self.data.records]).astype(
            np.int64
        )
        if np.any(self.targets < 0) or np.any(self.targets >= 10):
            raise ValueError("Animal-10N labels must be integers in 0..9")
        if len(np.unique(self.source_ids)) != len(self.source_ids):
            raise ValueError("Duplicate source IDs within Animal-10N partition")
        self.clean_training_targets = None


def verify_identity(raw, config):
    """Animal inputs must match the manifest frozen before model fitting."""
    if config["dataset"]["name"] != "animal10n":
        return
    expected = config["dataset"].get("manifest_sha256")
    if not expected or file_manifest(config["data_dir"])["manifest_sha256"] != expected:
        raise ValueError(
            "Animal-10N file identity differs from the frozen configuration"
        )
    if config["dataset"].get("native_size") != 64 or not isinstance(raw, Animal10N):
        raise ValueError("Animal-10N requires its 64px binary adapter")
