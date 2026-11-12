"""Pinned CIFAR-10.2 test archive, used only after frozen input verification."""

import hashlib
from io import BytesIO
from urllib.request import urlopen

import numpy as np

from room_for_doubt.data.confirmation import CLASSES

BLOB = "7a3829681be56af70ea80ec99cf3822e2f5edff3"
SIZE = 6393664
URL = "https://raw.githubusercontent.com/modestyachts/cifar-10.2/master/cifar102_test.npz"


def validate_test(images, labels, class_names, source_ids, reference_images):
    if (
        images.shape != (2000, 32, 32, 3) or images.dtype != np.uint8
        or labels.shape != (2000,) or labels.dtype.kind not in "iu"
        or np.any(labels >= 10) or np.any(labels < 0)
        or not np.array_equal(np.bincount(labels.astype(np.int64)), np.full(10, 200))
        or tuple(class_names) != CLASSES
        or source_ids.shape != (2000,) or source_ids.dtype.kind not in "iu"
        or np.any(source_ids < 0) or len(np.unique(source_ids)) != 2000
    ):
        raise ValueError("Unexpected CIFAR-10.2 test layout, class order, or sample IDs")
    external = {image.tobytes() for image in images}
    if len(external) != len(images):
        raise ValueError("CIFAR-10.2 contains duplicate pixels")
    for name, reference in reference_images.items():
        if any(image.tobytes() in external for image in reference):
            raise ValueError(f"CIFAR-10.2 pixel overlap with {name}")


def load_test(reference_images):
    with urlopen(URL, timeout=90) as response:
        data = response.read(SIZE + 1)
    git_header = f"blob {len(data)}\0".encode()
    if len(data) != SIZE or hashlib.sha1(git_header + data).hexdigest() != BLOB:
        raise ValueError("CIFAR-10.2 download differs from the frozen Git blob")
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        images, labels = archive["images"], archive["labels"]
        class_names = np.char.decode(archive["label_names"], "utf-8") if archive["label_names"].dtype.kind == "S" else archive["label_names"]
        source_ids = archive["ti_indices"]
    validate_test(images, labels, class_names, source_ids, reference_images)
    return images, labels, source_ids, hashlib.sha256(data).hexdigest()
