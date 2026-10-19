"""Dispatch research commands without importing training code during discovery."""

import argparse
import runpy
import sys

COMMANDS = {
    "contrast-feature-targets": "analysis.feature_contrasts",
    "study-augmentation-control": "analysis.augmentation_control",
    "run-kaggle-campaign": "analysis.kaggle_campaign",
    "study-base-features": "analysis.feature_study",
    "diagnose-kaggle": "analysis.kaggle_diagnostics",
    "post-kaggle": "analysis.post_kaggle",
    "ablate-targets": "analysis.target_ablation",
    "train-kaggle-base": "training.kaggle_base",
    "study-architectures": "analysis.architecture_study",
    "diagnose-confidence": "analysis.confidence_diagnostic",
    "summarize-evaluation": "analysis.summarize_evaluation",
    "evaluate-corrected": "evaluation.corrected_campaign",
    "prepare-meta-inputs": "analysis.meta_inputs",
    "cache-meta-features": "training.meta_feature_cache",
    "train-meta-corrected": "training.meta_retraining",
    "prepare-trajectories": "analysis.prepare_trajectories",
    "train-classifier": "training.classifier",
    "train-conditioned-trajectory": "training.conditioned_trajectory",
    "train-flow": "training.flow_matching",
    "train-trajectory": "training.trajectory_regression",
    "train-flip-regressor": "training.flip_regression",
    "train-direct-noise-detector": "training.direct_noise_detector",
    "evaluate-top2": "evaluation.top2_correction",
    "evaluate-flow-calibration": "evaluation.flow_calibration",
    "evaluate-flow-tta": "evaluation.flow_tta",
    "evaluate-badness": "evaluation.trajectory_badness",
    "evaluate-selective": "evaluation.selective_prediction",
    "evaluate-ensemble": "evaluation.deep_ensemble",
    "ablate-static-lda": "analysis.static_lda_ablation",
    "plot-trajectories": "plotting.trajectory_tsne",
    "plot-cartography": "plotting.cartography",
}


def main():
    parser = argparse.ArgumentParser(
        description="Room for Doubt research commands. Run from the repository root.",
        epilog="Pass --help after a command for its options. See docs/reproduction.md for reproduction requirements.",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = "room_for_doubt." + COMMANDS[args.command]
    sys.argv = [module, *args.arguments]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
