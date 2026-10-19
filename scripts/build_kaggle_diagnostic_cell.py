"""Package unpublished research code into one cell atop the public base commit."""

import base64
import io
from pathlib import Path
import subprocess
import zipfile
import json

PUBLIC_COMMIT = "ba613a76a817cd5dfa5589cd214de94d06fd5c87"
FILES = [
    "configs/kaggle_base.yaml",
    "src/room_for_doubt/training/kaggle_base.py",
    "configs/post_kaggle.yaml",
    "configs/diagnose_kaggle.yaml",
    "scripts/kaggle_diagnose.py",
    "scripts/kaggle_postprocess.py",
    "src/room_for_doubt/analysis/post_kaggle.py",
    "src/room_for_doubt/analysis/kaggle_diagnostics.py",
    "src/room_for_doubt/data/kaggle_artifacts.py",
    "src/room_for_doubt/inference/post_kaggle_cache.py",
    "src/room_for_doubt/inference/diagnostic_cache.py",
    "src/room_for_doubt/training/post_kaggle_meta.py",
    "src/room_for_doubt/training/diagnostic_controls.py",
    "src/room_for_doubt/evaluation/post_kaggle.py",
    "src/room_for_doubt/evaluation/diagnostic_metrics.py",
    "src/room_for_doubt/evaluation/diagnostic_study.py",
    "src/room_for_doubt/models/scalar_margin.py",
    "src/room_for_doubt/reporting/artifacts.py",
]


def build(root, destination, study="diagnostic", campaign_profile=None):
    root, destination = Path(root), Path(destination)
    files = list(FILES)
    if study in {"features", "extended", "campaign"}:
        files += [
            "configs/feature_study.yaml",
            "src/room_for_doubt/models/feature_margin.py",
            "src/room_for_doubt/inference/classifier_features.py",
            "src/room_for_doubt/training/feature_heads.py",
            "src/room_for_doubt/evaluation/feature_study.py",
            "src/room_for_doubt/analysis/feature_study.py",
        ]
    elif study != "diagnostic":
        raise ValueError("Unknown cell study")
    if study in {"extended", "campaign"}:
        files.append("configs/feature_study_extended.yaml")
    if study == "campaign":
        files += [
            "configs/kaggle_campaign.yaml",
            "scripts/kaggle_campaign.py",
            "src/room_for_doubt/analysis/kaggle_campaign.py",
            "src/room_for_doubt/analysis/campaign_stage.py",
            "src/room_for_doubt/evaluation/campaign_ensemble.py",
        ]
    if campaign_profile:
        if study != "campaign" or campaign_profile not in {"campaign_cifar100", "campaign_cifar10"}:
            raise ValueError("Invalid replication profile")
        files.append(f"configs/{campaign_profile}.yaml")
    command = (
        "study-base-features"
        if study in {"features", "extended"}
        else "diagnose-kaggle"
    )
    output_dir = (
        "/kaggle/working/rfd_feature_study"
        if study in {"features", "extended"}
        else "/kaggle/working/rfd_diagnostics"
    )
    run_id = (
        "base_feature_study_seed42"
        if study in {"features", "extended"}
        else "diagnostic_resnet50_seed42"
    )
    title = (
        "Сравнение признаков ResNet18 и ResNet50: 18 малых голов"
        if study in {"features", "extended"}
        else "Диагностика + 9 контрольных голов"
    )
    if study == "extended":
        run_id, title = (
            "expanded_feature_study_seed42",
            "Notebook A: расширенная проверка 7 целей, 42 головы",
        )
    if study == "campaign":
        output_dir, run_id = "/kaggle/working/rfd_campaign", "independent_runs"
        title = "Notebook B: независимые базы CIFAR100N/CIFAR10N и сравнения"
    if campaign_profile:
        run_id = "replication_cifar100" if campaign_profile == "campaign_cifar100" else "replication_cifar10"
        title = f"Replication: {run_id}, only missing base seeds"
    cli = subprocess.check_output(
        ["git", "show", PUBLIC_COMMIT + ":src/room_for_doubt/cli.py"], cwd=root
    ).decode()
    entries = '\n    "diagnose-kaggle": "analysis.kaggle_diagnostics",\n    "post-kaggle": "analysis.post_kaggle",'
    if study in {"features", "extended", "campaign"}:
        entries += '\n    "study-base-features": "analysis.feature_study",'
    if study == "campaign":
        entries += '\n    "run-kaggle-campaign": "analysis.kaggle_campaign",'
    cli = cli.replace("COMMANDS = {", "COMMANDS = {" + entries)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.writestr(name, (root / name).read_bytes())
        archive.writestr("src/room_for_doubt/cli.py", cli)
    encoded = base64.b64encode(payload.getvalue()).decode()
    launch = f'[sys.executable, str(ROOT / "scripts/kaggle_diagnose.py"), "--command", {command!r}, "--output-dir", {output_dir!r}'
    if study == "extended":
        launch += ', "--profile", "extended"'
    launch += "]"
    if study == "campaign":
        launch = '[sys.executable, str(ROOT / "scripts/kaggle_campaign.py")]'
    if campaign_profile:
        launch = launch[:-1] + f', "--profile", {campaign_profile!r}]'
    behavior = (
        "Обучение новых баз и автоматический downstream"
        if study == "campaign"
        else "База не переобучается"
    )
    final_archive = (
        "campaign_summary.zip" if study == "campaign" else "summary_bundle.zip"
    )
    cell = f"""# Одна ячейка: {title}. {behavior}.
# GPU T4 x2, Internet On. Подключи Output базы и предыдущего post-kaggle.
import base64
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working/room-for-doubt-{study}")
REPOSITORY = "https://github.com/TomahawK111100/Room-for-Doubt-2.0.git"
COMMIT = {PUBLIC_COMMIT!r}
assert Path("/kaggle/input").exists(), "Эта ячейка предназначена для Kaggle."
subprocess.run(["nvidia-smi"], check=True)
if not ROOT.exists():
    subprocess.run(["git", "clone", REPOSITORY, str(ROOT)], check=True)
subprocess.run(["git", "fetch", "origin", COMMIT], cwd=ROOT, check=True)
subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=ROOT, check=True)

# В пакете только код и конфигурации; результаты и веса берутся из inputs.
UPDATE = {encoded!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(UPDATE))) as archive:
    for name in archive.namelist():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(name))
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(ROOT)], check=True)
env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", MPLCONFIGDIR="/kaggle/working/matplotlib")
subprocess.run({launch}, cwd=ROOT, env=env, check=True)
print("ГОТОВО. Скачай {final_archive} из {output_dir}/{run_id}.")
"""
    compile(cell, str(destination), "exec")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cell)
    notebook = dict(
        cells=[
            dict(
                cell_type="code",
                execution_count=None,
                metadata={},
                outputs=[],
                source=cell.splitlines(keepends=True),
            )
        ],
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python"),
        ),
        nbformat=4,
        nbformat_minor=5,
    )
    destination.with_suffix(".ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1)
    )
    print(
        f"Cell: {destination}, {destination.stat().st_size} bytes; {len(files) + 1} code/config files"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=["diagnostic", "features", "extended", "campaign"],
        default="diagnostic",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    filename = {
        "features": "kaggle_feature_study_cell.py",
        "diagnostic": "kaggle_diagnostic_cell.py",
        "extended": "kaggle_experiments_A_cell.py",
        "campaign": "kaggle_experiments_B_cell.py",
    }[args.study]
    build(root, root / "outputs" / filename, args.study)
