"""Replay recorded Animal-10N outputs on CPU, without training or downloads."""

import argparse
from room_for_doubt.analysis.animal10n_audit import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/rfd_animal10n_audit")
    parser.add_argument("--max-minutes", type=float, default=45)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    result = run(args.input_root, args.output, args.max_minutes, args.inspect_only)
    raise SystemExit(
        0
        if result["status"] in {"complete_replay_checks", "inputs_ready_not_replayed"}
        else 1
    )
