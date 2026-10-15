#!/usr/bin/env python3
"""Reject oversized public working-tree files or reachable Git blobs."""

import argparse
from pathlib import Path
import subprocess

MAX_BYTES = 100_000_000


def oversized_files(root: Path, limit: int):
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    for name in sorted(set(names) - {""}):
        path = root / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size >= limit:
            yield name, path.stat().st_size


def oversized_blobs(root: Path, limit: int):
    objects = subprocess.check_output(
        ["git", "rev-list", "--objects", "--all"], cwd=root
    )
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype) %(objectsize) %(rest)"],
        cwd=root,
        input=objects,
        capture_output=True,
        check=True,
    )
    for line in result.stdout.decode().splitlines():
        kind, size, *name = line.split(" ", 2)
        if kind == "blob" and int(size) >= limit:
            yield name[0] if name else "<unnamed blob>", int(size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="Check all locally reachable Git blobs too",
    )
    args = parser.parse_args()
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    failures = list(oversized_files(root, MAX_BYTES))
    if args.history:
        failures.extend(oversized_blobs(root, MAX_BYTES))
    for name, size in sorted(set(failures)):
        print(f"Too large: {name} ({size:,} bytes; limit {MAX_BYTES:,})")
    if failures:
        raise SystemExit(1)
    print("File size check passed (limit: 100 MB).")


if __name__ == "__main__":
    main()
