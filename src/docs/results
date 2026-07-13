# ICOMP-2026: Experimental Results Template

**Project:** Robust Feature Distillation under Extreme Asymmetric Noise (Data Cartography Approach)
**Dataset:** CIFAR-10N / CIFAR-100N

## Main Classification Results

Evaluation of the model's robustness against severe label corruption (`cifar10n=worse` configuration).

| Scenario (Noise Type) | Method / Architecture | Test Accuracy (%) | Epochs to Converge | Inference Time (ms) |
| :--- | :--- | :---: | :---: | :---: |
| **CIFAR-10N (Worse)** | *Baseline (Standard ResNet-34)* | - | 100 | - |
| | *DivideMix (Li et al.)* | - | - | - |
| | *SOP (Liu et al.)* | - | - | - |
| | **Our Method (Stage 1: Cartography / Teacher)** | [Wait for log] | 100 | - |
| | **Our Method (Stage 2: Distillation / Student)** | **[Target: Best]** | **[Target: Fast]** | - |
| **CIFAR-100N (Noisy)** | *Baseline (Standard ResNet-34)* | - | 100 | - |
| | **Our Method (Full Pipeline)** | - | - | - |

## Ablation Study (Data Cartography Metrics)

Analysis of the sample distribution mapped during Stage 1.

| Cartography Zone | Sample Count | Mean Confidence (AUM) | Mean Variability (Flips) | Action in Stage 2 |
| :--- | :---: | :---: | :---: | :--- |
| **Easy-to-Learn** | - | High | Low | Distilled |
| **Hard-to-Learn** | - | Low | High | Distilled |
| **Ambiguous / Noise** | - | Low | Low | **Filtered / Ignored** |

---
*Note: The table will be updated with empirical values upon the completion of the Kaggle GPU training sessions.*