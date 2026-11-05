"""Verify independent replica wiring without running the generated notebook."""

import ast
import base64
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mini_notebook_builder", ROOT / "scripts/build_mini_webvision_notebook.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.mark.parametrize("seed", [43, 44])
def test_replica_changes_only_planned_seed_and_runtime(tmp_path, seed):
    builder.build(tmp_path, accepted=True, seed=seed)
    code = (tmp_path / "mini_webvision_cell.py").read_text()
    tree = ast.parse(code)
    constants = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant)
    }
    payload = base64.b64decode(constants["PAYLOAD"])
    assert hashlib.sha256(payload).hexdigest() == constants["SOURCE_SHA256"]
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert sorted(z.namelist()) == sorted(builder.FILES)
        cfg = yaml.safe_load(z.read("configs/mini_webvision_pilot.yaml"))
        original = yaml.safe_load((ROOT / "configs/mini_webvision_pilot.yaml").read_text())
        assert cfg["base"]["seed"] == seed
        assert cfg["base"]["num_workers"] == 0
        assert cfg["base"]["max_hours"] == 6.5
        assert cfg["max_hours"] == 8.5
        expected = yaml.safe_load(yaml.safe_dump(original))
        expected["pilot_seed"] = expected["base"]["seed"] = seed
        expected["base"]["run_id"] = f"miniwebvision_seed{seed}"
        expected["base"]["num_workers"] = 0
        expected["base"]["max_hours"] = 6.5
        expected["max_hours"] = 8.5
        assert cfg == expected
        for name in builder.FILES:
            if name != "configs/mini_webvision_pilot.yaml":
                assert z.read(name) == (ROOT / name).read_bytes()
    assert f'rfd_miniwebvision_seed{seed}' in code
    assert f'"--seed", "{seed}"' in code
    assert '"--seed", "42"' not in code
    assert "RESUME_SOURCE" not in code
    nb = json.loads((tmp_path / "mini_webvision_pilot.ipynb").read_text())
    assert "".join(nb["cells"][1]["source"]) == code


def test_replica_consent_gate_before_network(tmp_path):
    builder.build(tmp_path, accepted=False, seed=43)
    code = (tmp_path / "mini_webvision_cell.py").read_text()
    assert "OWNER_ACCEPTED_WEBVISION_TERMS = False" in code
    assert code.index("assert OWNER_ACCEPTED_WEBVISION_TERMS") < code.index('subprocess.run(["git"')


@pytest.mark.parametrize("seed", [43, 44])
@pytest.mark.parametrize("case", ["valid", "missing", "wrong_seed", "wrong_epoch"])
def test_resume_guard_refuses_missing_or_wrong_checkpoint(tmp_path, seed, case):
    builder.build(tmp_path, accepted=True, seed=seed, resume_epochs=43)
    code = (tmp_path / "mini_webvision_cell.py").read_text()
    start = code.index("# Refuse a fresh base")
    end = code.index('subprocess.run(["git", "clone"')
    assert start < end < code.index("Data ready.")
    guard = code[start:end]
    guard = guard.replace("Path('/kaggle/input')", "input_root")
    checkpoint = {
        "epoch": 41 if case == "wrong_epoch" else 42,
        "history": [None] * 43,
        "config": {
            "seed": seed + 1 if case == "wrong_seed" else seed,
            "base_split_seed": 42,
            "epochs": 50,
            "dataset": {"manifest_sha256": "1d08d37359e3e8a5ecb38664b02b38d9e08a99cea50e7e91ca624eb4cc004f90"},
        },
        "targets": SimpleNamespace(shape=(59350, 50)),
    }
    for key in ["state_dict", "optimizer", "scaler", "python_rng", "numpy_rng", "torch_rng", "best_state_dict", "train_ids", "val_ids"]:
        checkpoint[key] = None
    env = {
        "input_root": SimpleNamespace(rglob=lambda pattern: [] if case == "missing" else [tmp_path / "last.pt"]),
        "torch": SimpleNamespace(load=lambda *args, **kwargs: checkpoint),
    }
    if case == "valid":
        exec(compile(guard, "generated_resume_guard", "exec"), env)
    else:
        with pytest.raises(AssertionError):
            exec(compile(guard, "generated_resume_guard", "exec"), env)
    assert "shutil.copy2(RESUME_SOURCE, checkpoint_dir / 'last.pt')" in code
    assert "range(43, 50)" not in code  # Original trainer owns the epoch/scheduler state.
