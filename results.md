# Experimental Results: Room for Doubt (ICOMP-2026)

**Project:** Distilling training dynamics for inference-time error correction and uncertainty estimation  
**Dataset:** CIFAR-10N (Worse labels - extreme asymmetric noise)  
**Status:** Final Results (Averaged over 3 seeds)

## 1. Main Result: Error Correction via Top-2 Trajectory Fusion
| Metric | Value (Mean ± Std) |
|:-------|:------|
| Baseline Accuracy (Top-1) | 76.49% |
| **Corrected Accuracy** | **80.35% ± 0.45%** |
| **Absolute Accuracy Gain** | **+3.86% ± 0.45%** |

## 2. Uncertainty Estimation Quality (Error Detection)
| Metric | AUROC ↑ | AURC ↓ |
|:-------|:------|:-------|
| Maximum Softmax Probability (MSP) | 0.8036 | 0.0985 |
| **Trajectory Badness (Ours)** | **0.8536 ± 0.0053** | **0.0738 ± 0.0015** |

## 3. Ablation Study
* **Direct Certainty Estimator:** Fails to capture temporal variance, proving that distilling the *full trajectory* is required for active error correction.
