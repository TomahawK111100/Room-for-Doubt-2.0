# Experimental Results: Room for Doubt (ICOMP-2026)

**Project:** Distilling training dynamics for inference-time uncertainty estimation  
**Dataset:** CIFAR-10N (Worse labels - extreme asymmetric noise)  
**Status:** Week 4 Complete (Deterministic Baseline Established)

## 1. Main Results: Selective Inference Performance
The primary goal is to improve model reliability by rejecting samples with high predicted uncertainty (Training Dynamics Distillation).

| Rejection Rate (%) | Kept Samples | **Test Accuracy (%)** | Accuracy Gain |
|:-------------------|:-------------|:----------------------|:--------------|
| 0% (Baseline)      | 10000        | **74.06%**            | +0.00%        |
| 10%                | 9000         | 75.60%                | +1.54%        |
| 20%                | 8000         | 77.08%                | +3.02%        |
| 30%                | 7000         | 78.61%                | +4.55%        |
| 40%                | 6000         | 80.27%                | +6.21%        |
| 50%                | 5000         | **81.66%**            | **+7.60%**    |

**Key Insight:** Our distilled meta-regressor effectively identifies "Hard" and "Noisy" samples. Rejecting the top 30% of uncertain samples yields a significant +4.55% boost in accuracy.

## 2. Uncertainty Estimation Quality (Distillation Analysis)
We evaluate how well the Meta-Regressor (ResNet-18) captures the historical training dynamics (Flips) of the Teacher model (ResNet-34).

| Metric | Value | Interpretation |
|:-------|:------|:---------------|
| **Spearman Rank Correlation** | **0.289** | Statistically significant link between visual features and learning instability. |
| **Data Cartography (Hard/Noise)** | 50.0% | High concentration of noise in CIFAR-10N Worse split. |
| **Data Cartography (Ambiguous)** | 10.7% | Samples at the decision boundary, successfully targeted by our method. |

## 3. Comparative Analysis (Benchmark Mockup)
*Note: This table compares our current deterministic approach with standard baselines and future probabilistic extensions.*

| Method | Uncertainty Source | Test Acc (0% Rej) | AURC (↓) |
|:-------|:-------------------|:------------------|:---------|
| Standard ResNet-34 | None (Baseline) | 74.06% | TBD |
| MSP (Max Softmax Prob) | Static (Final Layer) | TBD | TBD |
| **Our Method (Det.)** | **Distilled Flips** | **74.06%** | **0.184** (est) |
| *Our Method (Prob.)* | *Flow Matching* | *Upcoming* | *Upcoming* |

## 4. Visual Artifacts
- **Training Dynamics Scatter:** `plots/week3_correlation_scatter.png`
- **Selective Inference Curve:** `plots/selective_inference_curve.png`
- **Data Cartography Map:** `plots/stage1_cartography_scatter_zones.png`