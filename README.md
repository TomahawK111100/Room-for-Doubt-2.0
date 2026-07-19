# Room for Doubt

**Distilling training dynamics for inference-time error correction and uncertainty estimation**

## Overview

Standard neural networks often produce confident predictions even for mislabeled, ambiguous, or unstable samples. Classical uncertainty scores such as maximum softmax probability, entropy, or energy are computed only from the final model output. They ignore how each sample behaved during training.

This project studies a different hypothesis:

> A model's doubt may be partially encoded in the temporal trajectory of learning.

Instead of asking only *what does the model predict now?*, we also ask:

* was this sample learned early or late?
* was its assigned-label confidence stable?
* did its margin remain positive or oscillate?
* can this historical behavior be predicted at inference time to correct errors on the fly?

The project aims to convert training dynamics into useful uncertainty indicators for noisy-label detection, selective prediction, uncertainty-weighted test-time augmentation (TTA), and active error correction.

---

## Main Hypothesis

Let $\mathcal{D} = \{(x_i, \tilde{y}_i)\}_{i=1}^{N}$ be a dataset with possibly noisy labels, where $x_i$ is an input image and $\tilde{y}_i$ is the observed label.

During training, each sample produces a temporal learning trajectory $\tau_i$. In our implementation, we track the **prediction margin** over $E$ epochs:
$$ \tau_i = [m_i^{(1)}, m_i^{(2)}, \dots, m_i^{(E)}] $$

The main hypothesis is that noise memorization and true learning have distinct mathematical signatures in the shape of this trajectory. 

Instead of compressing this trajectory into a single scalar (like mean or variance), we train a **Conditioned Meta-Model** to predict the *entire trajectory* from a static image and a proposed class label:
$$ \hat{\tau}_{i, c} = g_\phi(x_i, c) $$

By evaluating this predicted trajectory using a trained **Badness Detector** (e.g., Linear Discriminant Analysis), we can quantify the probability that the model's prediction is a result of noise memorization rather than clean learning.

---

## Research Questions

1. Do training trajectories of cleanly labeled and noisy samples form distinct, separable clusters?
2. Can full historical training trajectories be predicted from static visual features at inference time?
3. Can a trajectory-based "Badness Detector" accurately identify corrupted samples?
4. Can we use predicted trajectories to actively correct model errors (Top-2 Correction) without discarding data?
5. Does distilled uncertainty improve Selective Inference and Uncertainty-Weighted TTA?

---

## Project Scope

The project focuses on **extreme noisy-label uncertainty** using the **CIFAR-10N (Worse split)** dataset.

Primary goals:
* Train a classifier on noisy labels and log per-sample margin trajectories.
* Visualize the trajectory space (Data Cartography & t-SNE).
* Train a conditioned meta-head to predict full trajectories from frozen visual features.
* Train an LDA "Badness Detector" to classify trajectories as clean or noisy.
* Implement **Top-2 Trajectory Correction** to fix errors on the fly.
* Evaluate Selective Inference and Uncertainty-Weighted TTA.

---

## Methodological Stages

The project consists of three main stages.

### Stage 1: Training Dynamics Mining & Cartography
We train a standard ResNet-34 classifier on noisy labels. During training, we log the logit margin for the assigned label at each epoch:
$$ m_i^{(t)} = \log(p_i^{(t)}(\tilde{y}_i)) - \max_{k \neq \tilde{y}_i} \log(p_i^{(t)}(k)) $$

This yields a trajectory matrix. Visualizing these trajectories via t-SNE reveals a natural clustering of "Easy", "Ambiguous", and "Hard" samples, proving that noise memorization has a distinct topological signature.

### Stage 2: Trajectory Distillation & Badness Detection
We freeze the trained ResNet-34 backbone and attach a new lightweight MLP head. This head takes the image features and a class embedding as input, and predicts the 50-epoch margin trajectory for that specific class.

Simultaneously, we train a **Linear Discriminant Analysis (LDA)** classifier on the real trajectories from Stage 1. The LDA learns to separate clean trajectories from noisy ones, outputting a **Badness Score** $\in [0, 1]$.

### Stage 3: Inference-Time Error Correction
At inference time, we do not have access to the training history. For a new test image $x^*$:
1. The base classifier outputs probabilities, giving a Top-1 class ($c_1$) and Top-2 class ($c_2$).
2. The Meta-Model predicts the virtual training trajectories for both $c_1$ and $c_2$.
3. The LDA evaluates both trajectories.
4. **Top-2 Correction:** If the trajectory for $c_1$ looks like noise memorization (Badness > 0.5) and $c_2$ looks like clean learning (Badness < 0.5), the system actively overrides the prediction to $c_2$.

We also apply the Badness Score to:
* **Selective Inference:** Rejecting samples with high badness scores.
* **Uncertainty-Weighted TTA:** Down-weighting augmented views that yield high badness scores.

---

## Experimental Results (CIFAR-10N Worse)

* **Data Cartography:** LDA separates clean and noisy trajectories with **80.3% accuracy**.
* **Top-2 Correction:** Actively correcting errors on the fly yields a **+2.77%** absolute accuracy gain over the baseline, without discarding any data.
* **Selective Inference:** Rejecting the top 50% most uncertain samples increases accuracy from 74.06% to **81.66% (+7.60%)**.
* **Uncertainty-Weighted TTA:** Dynamically weighting augmented views outperforms standard equal-weight TTA.

---

## References

[1] J. Wei, Z. Zhu, H. Cheng, T. Liu, G. Niu, and Y. Liu.  
**Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations.**  
ICLR, 2022.

[2] S. Swayamdipta, R. Schwartz, N. A. Smith, and Y. Choi.  
**Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.**  
EMNLP, 2020.

[3] G. Pleiss, T. Zhang, E. R. Elibol, and K. Q. Weinberger.  
**Identifying Mislabeled Data using the Area Under the Margin Ranking.**  
NeurIPS, 2020.