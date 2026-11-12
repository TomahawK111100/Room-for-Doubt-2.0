"""Run fitting or the separate external evaluation of the teacher controls."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from room_for_doubt.analysis.temporal_teacher_fit import fit_controls
from room_for_doubt.data.confirmation import SEEDS
from room_for_doubt.training.temporal_snapshots import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    planning = ROOT / "docs/planning"
    s4_hash = file_digest(planning / "confirmation_protocol_v1.md")
    protocol_hash = file_digest(planning / "temporal_teacher_protocol_v1.md")
    sources = {}
    for seed in SEEDS:
        matches = list(args.inputs.rglob(f"rfd_s4_seed{seed}/frozen.json"))
        if len(matches) != 1:
            raise ValueError(f"Attach exactly one frozen S4 output for seed {seed}")
        sources[seed] = matches[0].parent

    if args.evaluate:
        from room_for_doubt.evaluation.temporal_teacher import evaluate

        controls = {}
        for seed in SEEDS:
            matches = list(args.inputs.rglob(f"rfd_s4_teacher_control/seed{seed}/frozen.json"))
            if len(matches) != 1:
                raise ValueError(f"Attach exactly one frozen control output for seed {seed}")
            controls[seed] = matches[0].parent
        metrics = evaluate(sources, controls, args.data, args.output, args.scratch,
                           s4_hash, protocol_hash, args.device)
        print(json.dumps(metrics, indent=2), flush=True)
        return

    for seed, source in sources.items():
        print(f"Fitting controls for seed {seed}", flush=True)
        metrics = fit_controls(source, args.output / f"seed{seed}", args.scratch,
                               args.data, seed, s4_hash, protocol_hash, args.device)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
