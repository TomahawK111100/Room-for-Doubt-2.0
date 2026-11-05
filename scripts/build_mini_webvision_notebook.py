"""Build one pilot notebook from an explicit source overlay; never submit it."""

import argparse
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e85077892dded27f4120feb4694d16eac3e39e8f"
FILES = [
    "configs/mini_webvision_pilot.yaml",
    "scripts/animal10n_pilot.py",
    "scripts/mini_webvision_pilot.py",
    "scripts/prepare_mini_webvision_data.py",
    "src/room_for_doubt/data/mini_webvision.py",
    "src/room_for_doubt/data/webvision_download.py",
    "src/room_for_doubt/data/animal10n.py",
    "src/room_for_doubt/data/cifar_noisy.py",
    "src/room_for_doubt/data/kaggle_artifacts.py",
    "src/room_for_doubt/data/observed_images.py",
    "src/room_for_doubt/training/kaggle_base.py",
    "src/room_for_doubt/analysis/feature_study.py",
    "src/room_for_doubt/analysis/augmentation_control.py",
    "src/room_for_doubt/evaluation/campaign_cache.py",
]


def build(out, accepted=False, seed=42, resume_epochs=None):
    if seed not in (42, 43, 44):
        raise ValueError("Only the planned base seeds42/43/44 are supported")
    if resume_epochs is not None and not 1 <= resume_epochs < 50:
        raise ValueError("Resume requires a completed epoch count between1and49")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            info = zipfile.ZipInfo(name, (2026, 10, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            data = (ROOT / name).read_bytes()
            if seed != 42 and name == "configs/mini_webvision_pilot.yaml":
                protocol = yaml.safe_load(data)
                protocol["pilot_seed"] = seed
                protocol["base"]["seed"] = seed
                protocol["base"]["run_id"] = f"miniwebvision_seed{seed}"
                # Runtime amendment from seed42: workers0 reduces worker memory
                # pressure; the two replications still required checkpoint recovery.
                protocol["base"]["num_workers"] = 0
                protocol["base"]["max_hours"] = 6.5
                protocol["max_hours"] = 8.5
                data = yaml.safe_dump(protocol, sort_keys=False).encode()
            archive.writestr(info, data)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    code = """# One frozen Mini-WebVision pilot. GPU T4 x2, Internet On, no Inputs required.
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

OWNER_ACCEPTED_WEBVISION_TERMS = __ACCEPTED__
assert OWNER_ACCEPTED_WEBVISION_TERMS, "The owner must accept the official WebVision terms before image download. https://data.vision.ee.ethz.ch/cvl/webvision/download.html"
ROOT = Path("/kaggle/working/room-for-doubt-miniwebvision")
OUT = Path("/kaggle/working/rfd_miniwebvision_seed42")
DATA = Path("/kaggle/temp/rfd_miniwebvision_data")
COMMIT = __COMMIT__
SOURCE_SHA256 = __DIGEST__
PAYLOAD = __PAYLOAD__
ALLOWED = __FILES__
assert Path("/kaggle/input").is_dir(), "Run this cell in Kaggle."
assert not ROOT.exists(), "Use a fresh session; preserve any previous Output for explicit resume."
assert not OUT.exists(), "Use a fresh run directory; do not overwrite earlier results."
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, "Select GPU T4 x2 before Save & Run All."
subprocess.run(["git", "clone", "--no-checkout", "https://github.com/TomahawK111100/Room-for-Doubt-2.0.git", str(ROOT)], check=True)
subprocess.run(["git", "fetch", "origin", COMMIT], cwd=ROOT, check=True)
subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=ROOT, check=True)
payload = base64.b64decode(PAYLOAD, validate=True)
assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    assert sorted(archive.namelist()) == sorted(ALLOWED)
    for name in ALLOWED:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "hydra-core>=1.3,<1.4", "pytorch-lightning>=2,<3", "scikit-learn", "pandas", "matplotlib", "seaborn", "tqdm"], check=True)
OUT.mkdir(parents=True)
artifacts = OUT / "artifacts"
(artifacts / "code").mkdir(parents=True)
(artifacts / "source_overlay.zip").write_bytes(payload)
for name in ALLOWED:
    (artifacts / "code" / name.replace("/", "__")).write_bytes((ROOT / name).read_bytes())
(artifacts / "source_overlay_manifest.json").write_text(json.dumps({
    "base_commit": COMMIT, "overlay_sha256": SOURCE_SHA256,
    "owner_accepted_webvision_terms": OWNER_ACCEPTED_WEBVISION_TERMS,
    "terms_url": "https://data.vision.ee.ethz.ch/cvl/webvision/download.html",
    "files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ALLOWED},
    "image_storage": str(DATA), "image_redistribution": False
}, indent=2) + "\\n")
env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1", MPLBACKEND="Agg")

# Save an inspectable failure even when network/data acquisition stops before training.
def export_setup_failure(error):
    from datetime import datetime, timezone
    (OUT / "config.yaml").write_bytes((ROOT / "configs/mini_webvision_pilot.yaml").read_bytes())
    (OUT / "metrics.json").write_text(json.dumps({"status": "blocked_data_preparation", "training_performed": False, "test_evaluated": False, "error": str(error)}, indent=2) + "\\n")
    (OUT / "run_metadata.json").write_text(json.dumps({"run_id": OUT.name, "created_at": datetime.now(timezone.utc).isoformat(), "base_commit": COMMIT, "source_overlay_sha256": SOURCE_SHA256}, indent=2) + "\\n")
    (OUT / "report.md").write_text("# Experiment Report: Mini-WebVision setup\\n\\n## Configuration\\nSee config.yaml and source overlay manifest.\\n\\n## Dataset\\nOfficial WebVision1.0 first50 Google classes; acquisition incomplete.\\n\\n## Checkpoint\\nNone.\\n\\n## Metrics\\nNo model training or test evaluation.\\n\\n## Compact Results\\nBlocked before training.\\n\\n## Generated Artifacts\\nSource snapshot and setup log.\\n\\n## Notes\\n" + str(error) + "\\n")
    with zipfile.ZipFile(OUT / "miniwebvision_summary.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix in {".yaml", ".json", ".md", ".txt", ".csv", ".py", ".log"}:
                archive.write(path, str(path.relative_to(OUT)))

try:
    # Images stay under /kaggle/temp and will not be published in notebook Outputs.
    command = [sys.executable, str(ROOT / "scripts/prepare_mini_webvision_data.py"), "--data-dir", str(DATA), "--accept-webvision-terms", "--max-download-minutes", "60"]
    print("Downloading only the original first50-class membership. Images remain in ephemeral storage.", flush=True)
    with (OUT / "data_preparation.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait() != 0:
            raise RuntimeError("Data preparation failed; see data_preparation.log in the summary ZIP")
    print("Data ready. Starting frozen input preflight and the single base42 pilot.", flush=True)
except Exception as error:
    export_setup_failure(error)
    print("Download", OUT / "miniwebvision_summary.zip", flush=True)
    raise
subprocess.run([sys.executable, str(ROOT / "scripts/mini_webvision_pilot.py"), "--input-root", str(DATA), "--output", str(OUT), "--seed", "42"], cwd=ROOT, env=env, check=True)
print("Download", OUT / "miniwebvision_summary.zip", flush=True)
"""
    for key, value in {
        "__ACCEPTED__": repr(accepted),
        "__COMMIT__": repr(COMMIT),
        "__DIGEST__": repr(digest),
        "__PAYLOAD__": repr(base64.b64encode(payload).decode()),
        "__FILES__": repr(FILES),
    }.items():
        code = code.replace(key, value)
    if seed != 42:
        code = code.replace("rfd_miniwebvision_seed42", f"rfd_miniwebvision_seed{seed}")
        code = code.replace('"--seed", "42"', f'"--seed", "{seed}"')
        code = code.replace("single base42 pilot", f"single base{seed} replication")
    if resume_epochs is not None:
        code = code.replace("no Inputs required.", "full previous Notebook Output required.")
        anchor = 'subprocess.run(["git", "clone", "--no-checkout"'
        start = code.index(anchor)
        guard = f'''
# Refuse a fresh base when a continuation was requested.
import gc
RESUME_CANDIDATES = list(Path('/kaggle/input').rglob('rfd_miniwebvision_seed{seed}/base/checkpoints/last.pt'))
assert len(RESUME_CANDIDATES) == 1, 'Attach exactly one full Output for base{seed}; last.pt required. No training started.'
RESUME_SOURCE = RESUME_CANDIDATES[0]
checkpoint = torch.load(RESUME_SOURCE, map_location='cpu', weights_only=False)
assert checkpoint['epoch'] == {resume_epochs - 1} and len(checkpoint['history']) == {resume_epochs}, 'Unexpected saved epoch; refusing to start'
assert checkpoint['config']['seed'] == {seed}, 'Wrong base seed'
assert checkpoint['config']['base_split_seed'] == 42, 'Wrong data split'
assert checkpoint['config']['dataset']['manifest_sha256'] == '1d08d37359e3e8a5ecb38664b02b38d9e08a99cea50e7e91ca624eb4cc004f90', 'Wrong data manifest'
assert checkpoint['config']['epochs'] == 50
assert checkpoint['targets'].shape == (59350, 50)
assert all(k in checkpoint for k in ['state_dict','optimizer','scaler','python_rng','numpy_rng','torch_rng','best_state_dict','train_ids','val_ids'])
del checkpoint
gc.collect()
print('RESUME VERIFIED: base{seed}, {resume_epochs}/50 complete; continuing at {resume_epochs + 1}/50.', flush=True)
'''
        code = code[:start] + guard + code[start:]
        anchor = '\n# Save an inspectable failure'
        start = code.index(anchor)
        recovery = f'''
import shutil
checkpoint_dir = OUT / 'base/checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(RESUME_SOURCE, checkpoint_dir / 'last.pt')
resume_hash = hashlib.sha256()
with RESUME_SOURCE.open('rb') as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        resume_hash.update(chunk)
(artifacts / 'resume_provenance.json').write_text(json.dumps({{
    'source_checkpoint': str(RESUME_SOURCE), 'source_checkpoint_sha256': resume_hash.hexdigest(),
    'base_seed': {seed}, 'completed_epochs': {resume_epochs}, 'next_epoch_display': {resume_epochs + 1},
    'scientific_settings': 'unchanged; original50epoch schedule; optimizer/RNG/targets restored',
    'runtime_workers': 0
}}, indent=2) + '\\n')
'''
        code = code[:start] + recovery + code[start:]
    ast.parse(code)
    instructions = """# Mini-WebVision: один фиксированный пилот seed42

**GPU T4 x2, Internet On. Inputs не нужны.** База → головы → парные сравнения →
статический контроль выполняются одной ячейкой. Seeds43/44 сами не запускаются.
Перед скачиванием изображений владелец должен принять официальные условия:
https://data.vision.ee.ethz.ch/cvl/webvision/download.html
Без подтверждения ячейка останавливается до сети и вычислений.

Официальные архивы Google/validation имеют объём около17.0GB/0.87GB;
потоковая загрузка сохраняет только перечисленные изображения классов0–49 и
может закончиться раньше конца архива. Состав:65944/2500 изображений.
Файлы изображений находятся в `/kaggle/temp`, не попадают в Outputs и не
публикуются как отдельный датасет. Время загрузки/обучения пока не измерено;
лимиты — мягкие остановки, не гарантия завершения за4,5ч.

После завершения нужен только `rfd_miniwebvision_seed42/miniwebvision_summary.zip`.
При ошибке сохраните полный Output и этот архив. Не перезапускайте обучение
автоматически и не меняйте настройки на основании тестовых результатов.
"""
    if seed != 42:
        instructions = instructions.replace("seed42", f"seed{seed}")
        instructions = instructions.replace(
            "Seeds43/44 сами не запускаются.",
            f"В этом ноутбуке запускается только база{seed}; три головы42/43/44 вложены в неё.",
        )
    if resume_epochs is not None:
        instructions = (
            f"# Продолжение базы{seed}: эпохи{resume_epochs + 1}–50\n\n"
            f"Подключите полный Notebook Output `mini-webvision-seed{seed}` с last.pt. "
            "При отсутствующем или неверном checkpoint ячейка остановится до сети и обучения. "
            "Сохраняется50эпох и прежняя научная конфигурация. После базы автоматически идут все сравнения.\n\n"
            + instructions.replace("Inputs не нужны.", "Полный предыдущий Notebook Output обязателен.")
        )
        instructions = instructions.replace(
            "Время загрузки/обучения пока не измерено;\nлимиты — мягкие остановки, не гарантия завершения за4,5ч.",
            "По завершённому seed42 ориентир6–8ч на полный запуск, плюс возможные задержки.\n"
            "workers0; лимиты6,5ч на базу/8,5ч на pipeline мягкие, не гарантия срока.\n"
            "Все научные настройки, сплит seed42,50эпох и три головы сохранены.",
        )
    notebook = dict(
        cells=[
            dict(
                id="mini-protocol",
                cell_type="markdown",
                metadata={},
                source=instructions.splitlines(True),
            ),
            dict(
                id="mini-pilot",
                cell_type="code",
                metadata={},
                execution_count=None,
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
            "language_info": {"name": "python"},
        },
        nbformat=4,
        nbformat_minor=5,
    )
    (out / "mini_webvision_pilot.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    )
    (out / "mini_webvision_cell.py").write_text(code)
    (out / "source_overlay.zip").write_bytes(payload)
    (out / "README.md").write_text(instructions)
    (out / "source_overlay_manifest.json").write_text(
        json.dumps(
            dict(
                base_commit=COMMIT,
                overlay_sha256=digest,
                files=FILES,
                owner_accepted_terms=accepted,
                base_seed=seed,
                resume_epochs=resume_epochs,
                runtime_profile="replication_workers0_8.5h" if seed != 42 else "pilot_4.5h",
            ),
            indent=2,
        )
        + "\n"
    )
    print(out / "mini_webvision_pilot.ipynb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/miniwebvision_preparation_20261023",
    )
    parser.add_argument("--owner-accepted-webvision-terms", action="store_true")
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    parser.add_argument("--resume-epochs", type=int)
    args = parser.parse_args()
    build(args.output, args.owner_accepted_webvision_terms, args.seed, args.resume_epochs)
