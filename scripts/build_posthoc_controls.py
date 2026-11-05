"""Build two deterministic Kaggle S2 notebooks without Git pushes or data bundles."""

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(directory, package, benchmark=True, stage="all"):
    if stage not in ("all", "benchmark"):
        raise ValueError("Build stage must be all or benchmark")
    if stage == "benchmark" and package != "A":
        raise ValueError("Runtime is a single package A job")
    if package not in ("A", "B"):
        raise ValueError("Package must be A or B")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(
        list((ROOT / "src/room_for_doubt").rglob("*.py"))
        + list((ROOT / "configs").rglob("*.yaml"))
        + [
            ROOT / "scripts/posthoc_controls.py",
            ROOT / "docs/planning/decisive_controls_protocol_v1.md",
        ]
    )
    payload = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.is_symlink():
                raise ValueError("Symlink in source snapshot")
            name = str(p.relative_to(ROOT))
            data = p.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, (2026, 10, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    data = payload.getvalue()
    manifest = dict(
        schema=1,
        package=package,
        source_archive_sha256=hashlib.sha256(data).hexdigest(),
        files=hashes,
        no_base_training=True,
        protocol="s2_v1",
        default_stage=stage,
        benchmark=package == "A" and (benchmark or stage == "benchmark"),
        new_head_fits=0 if stage == "benchmark" else (10 if package == "A" else 3),
    )
    code = """# S2: fixed protocol, no base training. Attach ALL FOUR completed Outputs listed below.
# A = capacity/target controls + runtime; B = independent-teacher controls.
# For a CPU-only diagnostic set MODE = "preflight". No training or test evaluation in that mode.
import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
import torch
PACKAGE = __PACKAGE__
MODE = "all"
RUN_BENCHMARK = __BENCHMARK__
OUT = Path("/kaggle/working/rfd_posthoc_controls_" + PACKAGE)
ROOT = Path("/kaggle/working/rfd_posthoc_source_" + PACKAGE)
DATA = Path("/kaggle/temp/rfd_posthoc_cifar")
MANIFEST = __MANIFEST__
PAYLOAD = __PAYLOAD__
INPUTS = [
 "https://www.kaggle.com/code/tomahawkkz/cifar100n-temporal-residual-pilot-seed42",
 "https://www.kaggle.com/code/tomahawkkz/cifar100n-temporal-frozen-test-seed42",
 "https://www.kaggle.com/code/tomahawkkz/cifar100n-temporal-replication-seed43",
 "https://www.kaggle.com/code/tomahawkkz/cifar100n-temporal-replication-seed44",
]
assert Path("/kaggle/input").is_dir(), "Run in Kaggle"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT).mkdir(parents=True, exist_ok=True)
blob = base64.b64decode(PAYLOAD)
assert hashlib.sha256(blob).hexdigest() == MANIFEST["source_archive_sha256"]
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    assert set(z.namelist()) == set(MANIFEST["files"])
    for name, expected in MANIFEST["files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        value = z.read(name)
        assert hashlib.sha256(value).hexdigest() == expected
        path = ROOT / name
        if path.exists(): assert path.read_bytes() == value, "Preserve previous source; use fresh session for a changed build"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
(OUT / "artifacts").mkdir(exist_ok=True)
(OUT / "artifacts/source_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
(OUT / "artifacts/source_snapshot.zip").write_bytes(blob)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "hydra-core>=1.3,<1.4", "pytorch-lightning>=2,<3", "scikit-learn", "pandas", "matplotlib", "seaborn", "tqdm"], check=True)
env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(OUT / "matplotlib"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
base = [sys.executable, str(ROOT / "scripts/posthoc_controls.py"), "output_dir="+str(OUT), "input_root=/kaggle/input", "data_dir="+str(DATA), "package="+PACKAGE, "run_id=posthoc_controls_"+PACKAGE+"_v1", "benchmark="+str(RUN_BENCHMARK).lower()]
subprocess.run(base+["stage=preflight"], cwd=ROOT, env=env, check=True)
report = json.loads((OUT / "preflight.json").read_text())
print(json.dumps(report, indent=2), flush=True)
if report["status"] != "verified_files":
    print("STOPPED BEFORE TRAINING: attach full original Outputs, not summaries. Required:", *INPUTS, sep="\\n", flush=True)
elif MODE == "preflight":
    print("CPU preflight complete. No fit/test access. For the real run select GPU T4x2, Internet On, MODE=all.", flush=True)
else:
    assert MODE in ("all", "benchmark"), "MODE must be all, benchmark or preflight"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2, "Select GPU T4x2 before Save & Run All"
    assert shutil.disk_usage("/kaggle/working").free > 4 * 1024**3, "Need 4GiB output storage"
    torch.set_num_threads(4);torch.manual_seed(42)
    x = torch.randn(16,16); reference = x @ x.T; witness=[]
    for i in range(2):
        actual=x.to(f"cuda:{i}") @ x.T.to(f"cuda:{i}")
        torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
        witness.append(dict(index=i, name=torch.cuda.get_device_name(i)))
    (OUT / "artifacts/kernel_witness.json").write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,gpus=witness),indent=2))
    print("CUDA_WITNESS_PASS", witness, flush=True)
    (OUT / "artifacts/pip_freeze.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    subprocess.run(base+["stage="+MODE], cwd=ROOT, env=env, check=True)
"""
    code = (
        code.replace("__PACKAGE__", repr(package))
        .replace("__BENCHMARK__", repr(manifest["benchmark"]))
        .replace("__MANIFEST__", repr(manifest))
        .replace("__PAYLOAD__", repr(base64.b64encode(data).decode()))
    )
    if stage == "benchmark":
        code = code.replace('MODE = "all"', 'MODE = "benchmark"')
        code = code.replace("RUN_BENCHMARK = False", "RUN_BENCHMARK = True")
        code = code.replace(
            'OUT = Path("/kaggle/working/rfd_posthoc_controls_" + PACKAGE)',
            'OUT = Path("/kaggle/working/rfd_posthoc_runtime_v1")',
        )
        code = code.replace(
            'ROOT = Path("/kaggle/working/rfd_posthoc_source_" + PACKAGE)',
            'ROOT = Path("/kaggle/working/rfd_posthoc_runtime_source_v1")',
        )
        code = code.replace(
            "# A = capacity/target controls + runtime; B = independent-teacher controls.",
            "# Runtime only: existing frozen models, no fitting or clean-test evaluation.",
        )
    ast.parse(code)
    notebook = dict(
        nbformat=4,
        nbformat_minor=5,
        metadata=dict(
            kernelspec=dict(name="python3", display_name="Python 3", language="python")
        ),
        cells=[
            dict(
                cell_type="code",
                id="s2-posthoc-" + package.lower(),
                execution_count=None,
                metadata={},
                outputs=[],
                source=code.splitlines(keepends=True),
            )
        ],
    )
    (out / f"posthoc_controls_{package}.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1)
    )
    (out / f"posthoc_controls_{package}_cell.py").write_text(code)
    diagnostic = json.loads(json.dumps(notebook))
    diagnostic["cells"][0]["source"] = code.replace(
        f'MODE = "{stage}"', 'MODE = "preflight"'
    ).splitlines(keepends=True)
    (out / f"posthoc_controls_{package}_preflight.ipynb").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=1)
    )
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "source_snapshot.zip").write_bytes(data)
    print(
        json.dumps(
            dict(
                package=package,
                notebook=str(out / f"posthoc_controls_{package}.ipynb"),
                source_sha256=manifest["source_archive_sha256"],
                source_files=len(files),
            )
        )
    )
    return notebook, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=str(ROOT / "outputs/posthoc_controls_launch_20261027")
    )
    parser.add_argument("--defer-benchmark", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()
    for package in ("A",) if args.runtime_only else ("A", "B"):
        build(
            Path(args.output) / package,
            package,
            benchmark=not args.defer_benchmark,
            stage="benchmark" if args.runtime_only else "all",
        )
