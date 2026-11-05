"""Freeze source/config files into a notebook; no git push, data or secrets."""

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(directory, replication_seed=None, selected_control=False):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p for p in (ROOT / "src/room_for_doubt").rglob("*.py")]
        + [p for p in (ROOT / "configs").rglob("*.yaml")]
        + [ROOT / "scripts/run.py", ROOT / "scripts/temporal_residual_pilot.py"]
    )
    if replication_seed is not None:
        if replication_seed not in (43, 44):
            raise ValueError("Replication seed must be43 or44")
        files.append(ROOT / "scripts/temporal_replication.py")
    if selected_control:
        files.append(ROOT / "scripts/temporal_selected_control.py")
    buffer = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError("Source snapshot must not contain symlinks")
            name, data = str(path.relative_to(ROOT)), path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, (2026, 10, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = dict(
        schema=1,
        source_archive_sha256=digest,
        files=hashes,
        test_evaluation=False,
        seed=42,
        dataset="cifar100n",
    )
    code = """# One CIFAR100N seed42 pilot, ending on noisy validation. GPU T4 x2, Internet On.
import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
import torch

OUT = Path("/kaggle/working/rfd_temporal_pilot_seed42")
ROOT = Path("/kaggle/working/room_for_doubt_temporal_source")
DATA = Path("/kaggle/temp/rfd_temporal_cifar_data")
MANIFEST = __MANIFEST__
PAYLOAD = __PAYLOAD__
assert Path("/kaggle/input").is_dir(), "Run in Kaggle"
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, "Select GPU T4 x2 before Save & Run All"
assert not OUT.exists() and not ROOT.exists(), "Fresh pilot only: preserve previous output before restarting"
assert shutil.disk_usage("/kaggle/working").free > 6 * 1024**3, "Need at least 6GiB free storage"
payload = base64.b64decode(PAYLOAD, validate=True)
assert hashlib.sha256(payload).hexdigest() == MANIFEST["source_archive_sha256"]
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert sorted(archive.namelist()) == sorted(MANIFEST["files"])
    for name, expected in MANIFEST["files"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        value = archive.read(name)
        assert hashlib.sha256(value).hexdigest() == expected
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
OUT.mkdir(parents=True)
(OUT / "artifacts").mkdir()
(OUT / "artifacts/source_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
(OUT / "artifacts/source_snapshot.zip").write_bytes(payload)
# Preserve Kaggle's compatible torch/torchvision wheels; install only missing analysis dependencies.
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "hydra-core>=1.3,<1.4", "pytorch-lightning>=2,<3", "scikit-learn", "pandas", "matplotlib", "seaborn", "tqdm"], check=True)
(OUT / "artifacts/pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
torch.set_num_threads(4)
torch.manual_seed(42)
x = torch.randn(16, 16)
reference = x @ x.T
witness = []
for index in range(2):
    actual = x.to(f"cuda:{index}") @ x.T.to(f"cuda:{index}")
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
    witness.append(dict(index=index, name=torch.cuda.get_device_name(index), finite=bool(torch.isfinite(actual).all())))
environment = dict(torch=torch.__version__, cuda=torch.version.cuda, gpus=witness,
                   python=sys.version, source_sha256=MANIFEST["source_archive_sha256"])
(OUT / "artifacts/kernel_witness.json").write_text(json.dumps(environment, indent=2))
print("CUDA_WITNESS_PASS", json.dumps(witness), flush=True)
env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(OUT / "matplotlib"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
subprocess.run([sys.executable, str(ROOT / "scripts/temporal_residual_pilot.py"), "--output", str(OUT), "--data-dir", str(DATA)], cwd=ROOT, env=env, check=True)
"""
    if replication_seed is not None:
        manifest.update(
            seed=replication_seed,
            test_evaluation=True,
            protocol="two branches freeze before any test; fixed epoch1 control",
        )
        code = code.replace(
            "One CIFAR100N seed42 pilot, ending on noisy validation",
            "CIFAR100N replication and fixed single-epoch control",
        )
        code = code.replace(
            "rfd_temporal_pilot_seed42",
            f"rfd_temporal_replication_seed{replication_seed}",
        )
        code = code.replace(
            '"scripts/temporal_residual_pilot.py"), "--output"',
            f'"scripts/temporal_replication.py"), "--seed", "{replication_seed}", "--output"',
        )
    if selected_control:
        manifest.update(
            test_evaluation=True,
            seeds=[43, 44],
            protocol="post-pilot noisy-validation selected single snapshot control",
        )
        code = code.replace(
            "One CIFAR100N seed42 pilot, ending on noisy validation",
            "Validation-selected single-snapshot controls43/44; attach both replication Outputs",
        )
        code = code.replace(
            "rfd_temporal_pilot_seed42", "rfd_temporal_selected_control"
        )
        code = code.replace(
            '"scripts/temporal_residual_pilot.py"',
            '"scripts/temporal_selected_control.py"',
        )
    code = code.replace("__MANIFEST__", repr(manifest)).replace(
        "__PAYLOAD__", repr(base64.b64encode(payload).decode())
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
                id="temporal-pilot",
                execution_count=None,
                metadata={},
                outputs=[],
                source=code.splitlines(keepends=True),
            )
        ],
    )
    (out / "temporal_pilot.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1)
    )
    (out / "temporal_pilot_cell.py").write_text(code)
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "source_snapshot.zip").write_bytes(payload)
    print(
        json.dumps(
            dict(
                notebook=str(out / "temporal_pilot.ipynb"),
                source_sha256=digest,
                source_files=len(files),
                bytes=len(payload),
            )
        )
    )
    return notebook, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=str(ROOT / "outputs/temporal_pilot_launch_20261026")
    )
    build(parser.parse_args().output)
