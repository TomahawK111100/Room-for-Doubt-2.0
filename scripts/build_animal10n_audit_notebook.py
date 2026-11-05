"""Build a self-contained source-overlay launcher; never submit a Kaggle job."""

import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/animal10n_audit_preparation_20261022"
COMMIT = "e85077892dded27f4120feb4694d16eac3e39e8f"
FILES = [
    "src/room_for_doubt/analysis/animal10n_audit.py",
    "scripts/animal10n_audit.py",
]
INPUTS = [
    "tomahawkkz/animal-10n",
    "tomahawkkz/seed43-animal",
    "tomahawkkz/seed-44-animal",
]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            info = zipfile.ZipInfo(name, (2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (ROOT / name).read_bytes())
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    code = """# CPU audit only. Attach all three completed Notebook Outputs and Animal-10N data.
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working/room-for-doubt-animal10n-audit")
OUT = Path("/kaggle/working/rfd_animal10n_audit")
COMMIT = __COMMIT__
PAYLOAD_SHA256 = __DIGEST__
PAYLOAD = __PAYLOAD__
ALLOWED = __FILES__
assert Path("/kaggle/input").is_dir(), "Run this cell in Kaggle."
assert not ROOT.exists(), "Use a fresh session, so the source checkout cannot contain old edits."

# Fast, standard-library check BEFORE installing anything or cloning.
mount = Path("/kaggle/input")
roots = {seed: [p for p in mount.rglob(f"rfd_animal10n_seed{seed}")
                if p.is_dir() and (p / "metrics.json").is_file()]
         for seed in (42, 43, 44)}
data = [p.parent for p in mount.rglob("source_provenance.json")
        if (p.parent / "data_batch_1.bin").is_file()
        and (p.parent / "test_batch.bin").is_file()]
print("Attached runs:", {seed: [str(p) for p in paths] for seed, paths in roots.items()})
assert all(len(paths) == 1 for paths in roots.values()), "Attach one full completed Output per seed42/43/44. Summaries are insufficient."
assert len(data) == 1, "Attach the existing Animal-10N binary dataset, once."
assert all((paths[0] / "base/checkpoints/best.pt").is_file() for paths in roots.values()), "Input is missing full checkpoints."

subprocess.run(["git", "clone", "--no-checkout", "https://github.com/TomahawK111100/Room-for-Doubt-2.0.git", str(ROOT)], check=True)
subprocess.run(["git", "fetch", "origin", COMMIT], cwd=ROOT, check=True)
subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=ROOT, check=True)
payload = base64.b64decode(PAYLOAD, validate=True)
assert hashlib.sha256(payload).hexdigest() == PAYLOAD_SHA256, "Source package checksum differs."
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert sorted(archive.namelist()) == sorted(ALLOWED), "Unexpected source overlay paths."
    for name in ALLOWED:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))

# Keep Kaggle's installed torch/torchvision. No pretrained-weight downloads.
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "hydra-core>=1.3,<1.4", "pytorch-lightning>=2,<3",
                "scikit-learn", "pandas", "matplotlib", "seaborn", "tqdm"], check=True)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "artifacts").mkdir(exist_ok=True)
(OUT / "artifacts/source_overlay.zip").write_bytes(payload)
(OUT / "artifacts/audit_cli.py").write_bytes((ROOT / "scripts/animal10n_audit.py").read_bytes())
(OUT / "artifacts/source_overlay_manifest.json").write_text(json.dumps({
    "base_commit": COMMIT, "overlay_sha256": PAYLOAD_SHA256,
    "files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ALLOWED},
    "notebook_sources": __INPUTS__, "dataset_source": "tomahawkkz/animal-10n-data",
    "training_performed": False
}, indent=2) + "\\n")
env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="")
subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=(OUT / "artifacts/requirements.txt").open("w"), check=True)
command = [sys.executable, str(ROOT / "scripts/animal10n_audit.py"), "--output", str(OUT), "--max-minutes", "45"]
result = subprocess.run(command, cwd=ROOT, env=env)
print("Download:", OUT / "animal10n_audit_summary.zip")
assert result.returncode == 0, "Audit did not pass. Send the summary ZIP; do not retrain."
"""
    for token, value in {
        "__COMMIT__": repr(COMMIT),
        "__DIGEST__": repr(digest),
        "__PAYLOAD__": repr(base64.b64encode(payload).decode()),
        "__FILES__": repr(FILES),
        "__INPUTS__": repr(INPUTS),
    }.items():
        code = code.replace(token, value)
    ast.parse(code)
    instructions = """# Animal-10N: проверка трёх завершённых запусков

Обучение не выполняется. Accelerator: **None (CPU)**. Internet: **On**.

В **Add Input → Notebooks → Your Work** подключите полные Outputs:
- https://www.kaggle.com/code/tomahawkkz/animal-10n
- https://www.kaggle.com/code/tomahawkkz/seed43-animal
- https://www.kaggle.com/code/tomahawkkz/seed-44-animal

В **Add Input → Datasets → Your Datasets** подключите существующий
`tomahawkkz/animal-10n-data` (оба `.bin` и `source_provenance.json`).
Не подключайте summary ZIP вместо полного Output. Ничего не скачивайте на Mac.

Нажмите **Save Version → Save & Run All**. После завершения скачайте только
`rfd_animal10n_audit/animal10n_audit_summary.zip` и передайте его в чат, в том числе
при ошибке. Проверка воспроизводит сохранённые решения на noisy validation,
все головы по сохранённым PCA-признакам, метрики и парные интервалы. Восемь
validation-изображений на базу используются для smoke-проверки ResNet.
Полная повторная extraction/PCA и новый подбор метода не выполняются.
45 минут — проверяемый бюджет вычислительного этапа, не измеренная оценка времени
и не жёсткий таймаут всего ноутбука. Реальное исполнение этого аудита ещё не проверено.
"""
    notebook = dict(
        cells=[
            dict(
                cell_type="markdown", metadata={}, source=instructions.splitlines(True)
            ),
            dict(
                cell_type="code",
                execution_count=None,
                metadata={},
                outputs=[],
                source=code.splitlines(True),
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        nbformat=4,
        nbformat_minor=5,
    )
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"animal-audit-{index}"
    (OUT / "animal10n_audit.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    )
    (OUT / "animal10n_audit_cell.py").write_text(code)
    (OUT / "README.md").write_text(instructions)
    (OUT / "source_overlay.zip").write_bytes(payload)
    (OUT / "config.yaml").write_text(
        f"base_commit: {COMMIT}\nsource_overlay_sha256: {digest}\naccelerator: cpu\ntraining: false\nmax_minutes: 45\n"
    )
    (OUT / "run_metadata.json").write_text(
        json.dumps(
            dict(
                run_id=OUT.name,
                status="prepared_not_executed",
                base_commit=COMMIT,
                source_overlay_sha256=digest,
                notebook_sources=INPUTS,
            ),
            indent=2,
        )
        + "\n"
    )
    (OUT / "metrics.json").write_text(
        json.dumps(
            dict(
                status="prepared_not_executed",
                target_kaggle_replay_executed=False,
                training_performed=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(OUT / "animal10n_audit.ipynb")


if __name__ == "__main__":
    build()
