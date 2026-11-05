"""Read official WebVision1 archives without republishing images in Outputs.

Metadata-only inspection does not request image archives. Image download requires
explicit owner acceptance of the official terms; no automatic terms acceptance.
"""

import contextlib
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

from room_for_doubt.data.mini_webvision import LISTS, records

BASE_URL = "https://data.vision.ee.ethz.ch/cvl/webvision/"
TERMS_URL = BASE_URL + "download.html"
ARCHIVES = {True: "google_resized_256.tar", False: "val_images_256.tar"}


@contextlib.contextmanager
def archive(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        with tarfile.open(fileobj=response, mode="r|*") as stream:
            yield stream


def fetch_lists(root):
    """Preserve the two original lists byte-for-byte; never fetch image data."""
    root = Path(root)
    info = root / "info"
    info.mkdir(parents=True, exist_ok=True)
    wanted = {"info/" + name for name in LISTS.values()}
    missing = {name for name in wanted if not (root / name).is_file()}
    if missing:
        with archive(BASE_URL + "info.tar") as stream:
            for member in stream:
                name = member.name.removeprefix("./")
                if name not in missing or not member.isfile():
                    continue
                target = root / name
                temporary = target.with_suffix(target.suffix + ".partial")
                with stream.extractfile(member) as source, temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                temporary.replace(target)
                missing.remove(name)
                if not missing:
                    break
        if missing:
            raise ValueError("Official metadata archive lacks " + str(sorted(missing)))
    summary = {
        "source_url": BASE_URL + "info.tar",
        "image_data_downloaded": False,
        "training_rows": len(records(root, True, check_files=False)),
        "evaluation_rows": len(records(root, False, check_files=False)),
        "evaluation_partition": "official_validation_first50_held_out",
        "official_archive_checksums_verified": False,
    }
    (info / "source_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def extract_selected(stream, wanted, max_seconds=None):
    """Extract exact listed names only; no generic tar extraction or silent drops."""
    pending = {
        name: Path(path) for name, path in wanted.items() if not Path(path).is_file()
    }
    started = time.monotonic()
    if not pending:
        return 0
    extracted = 0
    for member in stream:
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            raise TimeoutError(
                "Image archive time budget reached; downloaded files can resume"
            )
        name = member.name.removeprefix("./")
        if name not in pending or not member.isfile():
            continue
        target = pending[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        with stream.extractfile(member) as source, temporary.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        with Image.open(temporary) as image:
            image.verify()
        temporary.replace(target)
        del pending[name]
        extracted += 1
        if extracted % 5000 == 0:
            print(
                "Downloaded selected images:",
                extracted,
                "; remaining:",
                len(pending),
                flush=True,
            )
        if not pending:
            break
    if pending:
        raise ValueError(
            f"Archive did not contain {len(pending)} listed images; first: {next(iter(pending))}"
        )
    return extracted


def fetch_images(root, accept_terms=False, max_minutes=60):
    if not accept_terms:
        raise PermissionError(
            "Owner must explicitly accept WebVision terms before image download: "
            + TERMS_URL
        )
    if max_minutes <= 0:
        raise ValueError("Positive download time budget required")
    root = Path(root).resolve()
    # Working Outputs must never capture images when Kaggle commits a notebook.
    if root.is_relative_to(Path("/kaggle/working")):
        raise ValueError(
            "Use /kaggle/temp for images; never store them in notebook Outputs"
        )
    metadata = fetch_lists(root)
    deadline = time.monotonic() + max_minutes * 60
    for train in (True, False):
        wanted = {}
        for name, _, path in records(root, train, check_files=False):
            archive_name = name if train else "val_images_256/" + name
            wanted[archive_name] = path
        if all(path.is_file() for path in wanted.values()):
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Download budget exhausted before next archive")
        print(
            "Streaming",
            ARCHIVES[train],
            "; retaining only selected first50 classes",
            flush=True,
        )
        with archive(BASE_URL + ARCHIVES[train]) as stream:
            extract_selected(stream, wanted, remaining)
    return dict(
        metadata,
        image_data_downloaded=True,
        terms_url=TERMS_URL,
        distribution="local ephemeral research use; no image redistribution",
    )
