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
| | **Our Method (Stage 1: Cartography / Teacher)** | [Wait for evaluation] | 100 | - |
| | **Our Method (Stage 2: Distillation / Student)** | [Wait for evaluation] | - | - |
| **CIFAR-100N (Noisy)** | *Baseline (Standard ResNet-34)* | - | 100 | - |
| | **Our Method (Full Pipeline)** | - | - | - |

## Ablation Study (Data Cartography Metrics)

Analysis of the sample distribution mapped during Stage 1 and the effectiveness of Stage 2 feature distillation.

| Component | Value | Interpretation |
| :--- | :---: | :--- |
| **Distilled Samples** | **44,928** | Samples selected for teacher-to-student feature transfer |
| **Mean Feature Distance** | **9.366** | Average teacher–student embedding distance after distillation |
| **Median Feature Distance** | **9.074** | Typical feature alignment between teacher and student |
| **Minimum Distance** | **4.829** | Best-aligned feature representation |
| **Maximum Distance** | **27.687** | Hardest samples remaining after distillation |

---
*Stage 2 successfully completed feature distillation on 44,928 selected samples. The obtained embedding distances indicate that the student network effectively learned the teacher's feature representations, while the largest distances correspond to difficult or ambiguous samples identified during Data Cartography. Final classification accuracy and comparison with competing methods will be reported after the evaluation stage.*