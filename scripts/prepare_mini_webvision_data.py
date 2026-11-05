"""Prepare official Mini-WebVision metadata, optionally owner-approved images."""

import argparse
import json
from pathlib import Path

from room_for_doubt.data.webvision_download import fetch_images, fetch_lists

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/kaggle/temp/rfd_miniwebvision_data")
    )
    parser.add_argument("--accept-webvision-terms", action="store_true")
    parser.add_argument("--max-download-minutes", type=float, default=60)
    args = parser.parse_args()
    if args.accept_webvision_terms:
        result = fetch_images(args.data_dir, True, args.max_download_minutes)
    else:
        result = fetch_lists(args.data_dir)
    print(json.dumps(result, indent=2))
