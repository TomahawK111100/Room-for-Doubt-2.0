"""Exact CIFAR-10.1 v6 release. Imported only by the final evaluation."""

import hashlib
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

ASSETS = {
    "cifar10.1_v6_data.npy": ("24349cc5f3fb59648d75cd8015b5a184b1f5c250", 6144128),
    "cifar10.1_v6_labels.npy": ("e713b09cecf2ecd933970001909fcdec7c907ab0", 8128),
}


def load_v6(root, original_images):
    arrays, digests = [], {}
    for name, (blob_id, size) in ASSETS.items():
        path = Path(root) / name
        if path.exists():
            value = path.read_bytes()
        else:
            request = Request(
                f"https://api.github.com/repos/modestyachts/CIFAR-10.1/git/blobs/{blob_id}",
                headers={"Accept": "application/vnd.github+json"},
            )
            with urlopen(request, timeout=60) as response:
                blob = json.loads(response.read(size * 2 + 4096))
            if blob["sha"] != blob_id or blob["encoding"] != "base64":
                raise ValueError(f"Unexpected Git blob response for {name}")
            value = base64.b64decode("".join(blob["content"].split()), validate=True)
        header = f"blob {len(value)}\0".encode()
        if len(value) != size or hashlib.sha1(header + value).hexdigest() != blob_id:
            raise ValueError(f"Unexpected CIFAR-10.1 release file: {name}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(value)
            temporary.replace(path)
        digests[name] = hashlib.sha256(value).hexdigest()
        arrays.append(np.load(path, allow_pickle=False))
    images, labels = arrays
    if (
        images.shape != (2000, 32, 32, 3)
        or images.dtype != np.uint8
        or labels.shape != (2000,)
        or labels.dtype.kind not in "iu"
        or np.any((labels < 0) | (labels >= 10))
        or not np.array_equal(np.bincount(labels.astype(np.int64)), np.full(10, 200))
    ):
        raise ValueError("Unexpected CIFAR-10.1 v6 shape or class balance")
    original = {hashlib.sha256(image.tobytes()).digest() for image in original_images}
    external = {hashlib.sha256(image.tobytes()).digest() for image in images}
    if len(external) != len(images) or original.intersection(external):
        raise ValueError("Duplicate external images or overlap with CIFAR-10 train/val")
    return images, labels, digests
