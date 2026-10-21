"""Run verified remaining CIFAR jobs without downloading notebook Outputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from omegaconf import OmegaConf

import kaggle_followups as followups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/rfd_cifar_controls_recovery"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recovered = args.output / "recovered_caches"
    report = followups.inspect_inputs(args.input_root, followups.REMAINING, recovered)
    attempts = []
    for row in report["studies"]:
        if row["key"] not in followups.REMAINING or row["valid"]:
            continue
        # Only a missing shared cache can be repaired this way. No missing study,
        # partial training, wrong weights or absent heads are replaced.
        if (row["rejection_reasons"] != ["Required files missing"]
                or not row["missing_files"]
                or any(not Path(p).is_absolute() or Path(p).parent.name != "cache"
                       for p in row["missing_files"])):
            continue
        try:
            from torchvision.datasets import CIFAR10, CIFAR100
            from room_for_doubt.data.cifar_noisy import load_observed_labels
            from room_for_doubt.data.kaggle_artifacts import load_completed_run
            from room_for_doubt.data.native_cache_recovery import recover_native_cache
            cfg = OmegaConf.load(Path(row["study"]) / "config.yaml")
            data = followups.prepare_data(args.input_root, args.output / "data", [dict(dataset=cfg.dataset.name)])
            noisy = load_observed_labels(data, cfg.dataset.name, cfg.dataset.noisy_split)
            bundle = load_completed_run(Path(row["base"]), noisy)
            if bundle["checkpoint_digest"] != row["expected_digest"]:
                raise ValueError("Actual base checkpoint identity differs")
            cls = CIFAR10 if cfg.dataset.name == "cifar10n" else CIFAR100
            test = cls(str(data), train=False, download=False)
            native = Path(row["study"]) / "native_features"
            val_ids = np.asarray(json.loads((native / "val_complete.json").read_text())["ids"])
            test_ids = np.asarray(json.loads((native / "test_complete.json").read_text())["ids"])
            if (not np.array_equal(val_ids, bundle["data"]["base_val_ids"])
                    or not np.array_equal(test_ids, np.arange(len(test)))):
                raise ValueError("Native IDs differ from official source splits")
            recover_native_cache(row["study"], recovered / row["key"], row["expected_digest"],
                                 dict(val=noisy[val_ids], test=np.asarray(test.targets)[test_ids]))
            attempts.append(dict(job=row["key"], status="recovered_no_download_no_inference"))
            print("Восстановлен локальный кэш:", row["key"], flush=True)
        except (OSError, ValueError, KeyError) as exc:
            attempts.append(dict(job=row["key"], status="blocked", reason=str(exc)))
            print("Восстановление не прошло проверку:", row["key"], str(exc), flush=True)
    report = followups.inspect_inputs(args.input_root, followups.REMAINING, recovered)
    (args.output / "input_inspection.json").write_text(json.dumps(report, indent=2))
    valid = {row["key"] for row in report["studies"] if row["valid"]}
    results = []
    root = Path(__file__).resolve().parents[1]
    for key in followups.REMAINING:
        if key not in valid:
            results.append(dict(job=key, status="blocked_missing_verified_inputs"))
            continue
        print("Запускаю готовую базу:", key, flush=True)
        result = subprocess.run([
            sys.executable, str(root / "scripts/kaggle_followups.py"),
            "--profile", "remaining", "--jobs", key,
            "--input-root", str(args.input_root), "--cache-root", str(recovered),
            "--output", str(args.output / key), "--max-hours", "2",
        ], cwd=root)
        results.append(dict(job=key, status="complete" if result.returncode == 0 else "failed",
                            returncode=result.returncode))
    status = "complete" if all(r["status"] == "complete" for r in results) else "partial"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    metadata = dict(run_id="remaining_cifar_available_controls", git_commit=commit,
                    input_root=str(args.input_root), output=str(args.output),
                    expected_jobs=list(followups.REMAINING), notebook_downloads=False,
                    base_retraining=False, recoveries=attempts)
    OmegaConf.save(OmegaConf.create(metadata), args.output / "config.yaml", resolve=True)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.output / "metrics.json").write_text(json.dumps(dict(status=status, jobs=results), indent=2))
    lines = ["Status: " + status] + [f"{r['job']}: {r['status']}" for r in results]
    (args.output / "morning_summary.txt").write_text("\n".join(lines) + "\n")
    (args.output / "report.md").write_text(
        "# Available CIFAR controls\n\n" + "\n".join(lines)
        + "\n\nNative cache substitution retains explicit provenance and verifies saved frozen-policy accuracies. "
          "No missing base, head, study or training epoch is synthesized. Blocked jobs remain incomplete.\n")
    with zipfile.ZipFile(args.output / "followup_summary.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob("*")):
            if (path.is_file() and path.suffix in {".yaml", ".json", ".csv", ".md", ".txt"}
                    and ".hydra" not in path.parts):
                archive.write(path, str(path.relative_to(args.output)))
    print("\n".join(lines), flush=True)
    print("Сводка:", args.output / "followup_summary.zip", flush=True)


if __name__ == "__main__":
    main()
