"""Build private Kaggle notebooks with immutable S4 source snapshots."""

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/planning/confirmation_protocol_v1.md"
TEACHER_PROTOCOL = ROOT / "docs/planning/temporal_teacher_protocol_v1.md"
PROTOCOL_SHA256 = "6e9e35a59fa0d356891f6cd36b62f54ae7905a58501408bf87d8221db7199d2c"
TEACHER_PROTOCOL_SHA256 = "c18a65abfbcd2cd87de52f65e9103f92f519b8ba9714693e2695d1a8398e3bae"
SEEDS = (42, 43, 44)


def build(directory, seed, *, teacher_control=False, teacher_evaluation=False):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    stage = "evaluate" if seed is None else "fit"
    if teacher_control:
        stage = "teacher_control"
    if teacher_evaluation:
        stage = "teacher_evaluation"
    excluded = set()
    if stage not in ("evaluate", "teacher_evaluation"):
        excluded.update(("data/cifar10_1.py", "evaluation/confirmation.py"))
    if not teacher_evaluation:
        excluded.update(("data/cifar10_2.py", "evaluation/temporal_teacher.py"))
    tag = f"seed{seed}" if stage == "fit" else stage.replace("evaluate", "evaluation")
    source = ROOT / "src/room_for_doubt"
    files = sorted(
        [
            p
            for p in source.rglob("*.py")
            if p.relative_to(source).as_posix() not in excluded
        ]
        + list((ROOT / "configs").rglob("*.yaml"))
        + [ROOT / "scripts/run.py", ROOT / "scripts/confirmation_s4.py", PROTOCOL]
    )
    if teacher_control or teacher_evaluation:
        files += [ROOT / "scripts/temporal_teacher_fit.py", TEACHER_PROTOCOL]
    buf = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError("Source snapshot contains a symlink")
            name, data = path.relative_to(ROOT).as_posix(), path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, (2026, 11, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    protocol_hash = hashes[PROTOCOL.relative_to(ROOT).as_posix()]
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("Scientific protocol differs from the frozen digest")
    if teacher_control or teacher_evaluation:
        protocol_hash = hashes[TEACHER_PROTOCOL.relative_to(ROOT).as_posix()]
        if protocol_hash != TEACHER_PROTOCOL_SHA256:
            raise ValueError("Teacher protocol differs from the frozen digest")
    payload = buf.getvalue()
    manifest = dict(
        protocol_sha256=protocol_hash,
        stage=stage,
        seed=seed,
        files=hashes,
        source_archive_sha256=hashlib.sha256(payload).hexdigest(),
    )
    code = """import base64, hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import torch

DEADLINE = time.monotonic() + 6 * 3600
MANIFEST = __MANIFEST__
PAYLOAD = __PAYLOAD__
STAGE, SEED = MANIFEST["stage"], MANIFEST["seed"]
TAG = __TAG__
OUT = Path("/kaggle/working") / f"rfd_s4_{TAG}"
ROOT = Path("/kaggle/temp/rfd_s4_source")
DATA = Path("/kaggle/temp/rfd_s4_cifar10n")
assert Path("/kaggle/input").is_dir(), "Run in Kaggle"
assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0), "Select GPU T4 x2"
assert shutil.disk_usage("/kaggle/working").free >= 6 * 1024**3, "Need 6 GiB free"
payload = base64.b64decode(PAYLOAD, validate=True)
assert hashlib.sha256(payload).hexdigest() == MANIFEST["source_archive_sha256"]
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert sorted(archive.namelist()) == sorted(MANIFEST["files"])
    for name, sha in MANIFEST["files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == sha
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "source_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "hydra-core>=1.3,<1.4", "pytorch-lightning>=2,<3", "scikit-learn",
    "pandas", "matplotlib", "seaborn", "tqdm"], check=True, timeout=900)
(OUT / "environment.json").write_text(json.dumps({
    "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, indent=2))
env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1",
           MPLBACKEND="Agg", MPLCONFIGDIR="/kaggle/temp/rfd_s4_mpl",
           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
cmd = [sys.executable, str(ROOT / "scripts/confirmation_s4.py"), STAGE,
       "--data", str(DATA), "--output", str(OUT)]
__STAGE_COMMAND__
subprocess.run(cmd, env=env, cwd=ROOT, check=True)
"""
    if teacher_control or teacher_evaluation:
        stage_command = """cmd = [sys.executable, str(ROOT / "scripts/temporal_teacher_fit.py"),
       "--inputs", "/kaggle/input", "--scratch", "/kaggle/temp/rfd_teacher_scratch",
       "--data", str(DATA), "--output", str(OUT)]"""
        if teacher_evaluation:
            stage_command += '\ncmd.append("--evaluate")'
    elif stage == "fit":
        stage_command = 'cmd += ["--seed", str(SEED), "--deadline", str(DEADLINE)]'
    else:
        stage_command = """roots = []
for seed in (42, 43, 44):
    candidates = list(Path("/kaggle/input").rglob(f"rfd_s4_seed{seed}/frozen.json"))
    if len(candidates) != 1:
        raise ValueError(f"Attach exactly one frozen output for seed {seed}")
    roots.append(str(candidates[0].parent))
cmd += ["--frozen", *roots, "--external", "/kaggle/temp/cifar10_1_v6"]"""
    code = (
        code.replace("__MANIFEST__", repr(manifest))
        .replace("__TAG__", repr(tag))
        .replace("__PAYLOAD__", repr(base64.b64encode(payload).decode()))
        .replace("__STAGE_COMMAND__", stage_command)
    )
    ast.parse(code)
    path = out / f"S4_{tag}.ipynb"
    notebook = dict(
        nbformat=4,
        nbformat_minor=5,
        metadata=dict(
            kernelspec=dict(name="python3", display_name="Python 3", language="python")
        ),
        cells=[
            dict(
                cell_type="code",
                id=f"s4-{tag}",
                execution_count=None,
                metadata={},
                outputs=[],
                source=code.splitlines(keepends=True),
            )
        ],
    )
    path.write_text(json.dumps(notebook, indent=1))
    (out / f"{tag}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    teacher = parser.add_mutually_exclusive_group()
    teacher.add_argument("--teacher-control", action="store_true")
    teacher.add_argument("--teacher-evaluation", action="store_true")
    args = parser.parse_args()
    if args.teacher_control:
        print(build(args.output, None, teacher_control=True))
        return
    if args.teacher_evaluation:
        print(build(args.output, None, teacher_evaluation=True))
        return
    for seed in (*SEEDS, None):
        print(build(args.output, seed))


if __name__ == "__main__":
    main()
