"""Build a small evaluation-only notebook. Full pilot Output must be attached."""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    script = (ROOT / "scripts/temporal_residual_final_test.py").read_bytes()
    sha = hashlib.sha256(script).hexdigest()
    code = """# Frozen CIFAR100N clean-test evaluation. Attach completed temporal pilot Output.
import base64, hashlib, os, subprocess, sys
from pathlib import Path
import torch
assert Path('/kaggle/input').is_dir(), 'Run in Kaggle'
assert torch.cuda.is_available(), 'Select GPU T4 x2'
SCRIPT_SHA256 = __SHA__
SCRIPT = __SCRIPT__
value = base64.b64decode(SCRIPT, validate=True)
assert hashlib.sha256(value).hexdigest() == SCRIPT_SHA256
path = Path('/kaggle/working/frozen_temporal_test.py')
path.write_bytes(value)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'hydra-core>=1.3,<1.4', 'pytorch-lightning>=2,<3', 'scikit-learn', 'pandas', 'matplotlib', 'seaborn', 'tqdm'], check=True)
torch.manual_seed(42)
x = torch.randn(16,16)
torch.testing.assert_close((x.cuda() @ x.T.cuda()).cpu(), x @ x.T, rtol=1e-4, atol=1e-4)
print('EVALUATION_GPU_WITNESS_PASS', torch.cuda.get_device_name(), flush=True)
env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg', MPLCONFIGDIR='/kaggle/working/matplotlib', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
subprocess.run([sys.executable, str(path)], env=env, check=True)
print('Download /kaggle/working/rfd_temporal_test_seed42/temporal_test_results.zip')
"""
    code = code.replace("__SHA__", repr(sha)).replace(
        "__SCRIPT__", repr(base64.b64encode(script).decode())
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
                id="frozen-final-test",
                execution_count=None,
                metadata={},
                outputs=[],
                source=code.splitlines(keepends=True),
            )
        ],
    )
    (out / "temporal_test.ipynb").write_text(json.dumps(notebook, indent=1))
    (out / "cell.py").write_text(code)
    (out / "launch_manifest.json").write_text(
        json.dumps(
            dict(
                script_sha256=sha,
                pilot_version=362574753,
                pilot_url="https://www.kaggle.com/code/tomahawkkz/cifar100n-temporal-residual-pilot-seed42",
                stage="evaluate_only",
                fixed_alpha=1,
                training_performed=False,
            ),
            indent=2,
        )
    )
    print(json.dumps(dict(notebook=str(out / "temporal_test.ipynb"), sha256=sha)))
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=str(ROOT / "outputs/temporal_test_launch_20261027")
    )
    build(parser.parse_args().output)
