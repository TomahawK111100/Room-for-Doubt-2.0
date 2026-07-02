# Room for Doubt

**Distilling training dynamics for inference-time uncertainty estimation**

## Overview

Standard neural networks often produce confident predictions even for mislabeled, ambiguous, or unstable samples. Classical uncertainty scores such as maximum softmax probability, entropy, or energy are computed only from the final model output. They ignore how each sample behaved during training.

This project studies a different hypothesis:

> A model's doubt may be partially encoded in the temporal trajectory of learning.

Instead of asking only *what does the model predict now?*, we also ask:

* was this sample learned early or late?
* was it forgotten during training?
* was its assigned-label confidence stable?
* did its margin remain positive or oscillate?
* can this historical behavior be predicted at inference time?

The project aims to convert training dynamics into useful uncertainty indicators for noisy-label detection, selective prediction, and uncertainty-aware inference.

---

## Main Hypothesis

Let

$$
\mathcal{D} = {(x_i, \tilde{y}*i)}*{i=1}^{N}
$$

be a dataset with possibly noisy labels, where $x_i$ is an input image and $\tilde{y}_i$ is the observed label.

During training, each sample produces a temporal learning trajectory:

$$
\tau_i = {q_i^{(t)}}_{t=1}^{T}.
$$

In the simplest implementation, the logged state $q_i^{(t)}$ contains primitive signals such as per-sample loss and predicted class probabilities:

$$
q_i^{(t)} =
\left[
\ell_i^{(t)},,
p_i^{(t)}(1),,
p_i^{(t)}(2),,
\dots,,
p_i^{(t)}(C)
\right].
$$

Thus, each sample is represented not only by its final prediction, but by a whole trajectory matrix:

$$
\tau_i \in \mathbb{R}^{T \times (C+1)}.
$$

The main hypothesis is that uncertainty, ambiguity, or label noise may be encoded in the **shape of this trajectory**, not only in its final value.

There are several possible ways to use $\tau_i$.

First, the trajectory can be compressed into hand-crafted statistical descriptors:

$$
s_i = A(\tau_i),
$$

for example:

$$
s_i =
\begin{bmatrix}
\mathrm{AUM}_i, \
\mathrm{MeanConf}_i, \
\mathrm{VarConf}_i, \
\mathrm{Forgetting}_i
\end{bmatrix}.
$$

These descriptors are interpretable and provide a strong first baseline.

However, this is not the only possible representation. The full trajectory may contain information that is lost when averaging across epochs or reducing dynamics to a few scalar statistics. Therefore, the project also allows learned trajectory representations:

$$
r_i = E_\psi(\tau_i),
$$

where $E_\psi$ may be a small sequence encoder, temporal model, or generative model trained on the collected dynamics.

The uncertainty score may then be constructed from either hand-crafted descriptors, learned trajectory representations, or both:

$$
u_i = U(s_i, r_i).
$$

For inference-time use, the project studies whether this historical information can be distilled from a final representation:

$$
h_i = f_\theta(x_i).
$$

A simple version predicts descriptor statistics:

$$
g_\phi(h_i) \approx s_i.
$$

A more expressive version may predict a learned trajectory representation:

$$
g_\phi(h_i) \approx r_i,
$$

or model a conditional distribution over descriptors or trajectory-derived representations:

$$
p_\psi(s_i \mid h_i)
\quad \text{or} \quad
p_\psi(r_i \mid h_i).
$$

This is where conditional flow matching may become useful: not as a replacement for simple descriptors, but as a way to model uncertainty over possible training-dynamics summaries.

Therefore, the broader research question is:

> Can the temporal behavior of a sample during training — represented either by compact descriptors or by learned trajectory models — provide uncertainty signals that are stronger than static final-output confidence, and can these signals be distilled for inference-time use?

The descriptor-based approach is the minimal interpretable baseline. The full-trajectory and flow-matching approaches are advanced extensions that test whether richer temporal information improves uncertainty estimation.


---

## Research Questions

1. Can training-dynamics descriptors identify mislabeled or unstable samples better than static confidence scores?

2. Can historical training-dynamics descriptors be predicted from final representations?

3. Can predicted descriptors serve as useful uncertainty indicators at inference time?

4. Can these uncertainty indicators improve selective prediction or uncertainty-weighted prediction?

5. Does conditional flow matching provide useful probabilistic information beyond deterministic descriptor regression?

---

## Project Scope

The first version of the project focuses on **noisy-label uncertainty**.

The project does **not** focus on general OOD detection in the initial version.

Primary goals:

* train a classifier on noisy labels;
* log simple per-sample training trajectories;
* compute trajectory descriptors offline;
* evaluate post-hoc dynamics as uncertainty indicators;
* distill these descriptors into an inference-time predictor;
* evaluate predicted uncertainty against static baselines;
* optionally use predicted uncertainty with weak TTA or dropout;
* explore conditional flow matching as an advanced extension.

The expected result is a reproducible empirical study suitable for a short student research paper.

---

## Dataset

The primary dataset is **CIFAR-10N**, which provides human-annotated noisy labels for CIFAR-10 images.

Recommended splits:

* `aggregate`;
* `worse`;
* optionally other CIFAR-10N noisy-label variants.

Clean CIFAR-10 labels may be used for evaluation, but not for training the base classifier or constructing uncertainty targets.

---

## Methodological Stages

The project has three methodological stages.

---

## Stage 1: Training-Dynamics Logging and Post-Hoc Analysis

Train a standard classifier on noisy labels.

During training, do not compute every possible metric inside the training loop. Instead, log simple primitive signals that are sufficient to reconstruct many descriptors later.

Minimal per-sample logging:

$$
q_i^{(t)} = \begin{bmatrix}
\ell_i^{(t)} \\
p_i^{(t)}
\end{bmatrix},
$$

where:

* $\ell_i^{(t)}$ is the loss for sample $x_i$ at epoch $t$;
* $p_i^{(t)} \in \mathbb{R}^{C}$ is the predicted probability vector over classes.

The log should preserve:

* sample ID;
* epoch;
* noisy label;
* clean label for evaluation only;
* per-sample loss;
* predicted class probabilities.

From these primitive logs, descriptors can be computed offline.

Examples:

### Assigned-label confidence

$$
\mathrm{Conf}_i^{(t)} = p_i^{(t)}(\tilde{y}_i).
$$

### Maximum softmax probability

$$
\mathrm{MSP}_i^{(t)} = \max_k p_i^{(t)}(k).
$$

### Assigned-label margin

If full probabilities are stored, a log-probability margin can be computed as:

$$
m_i^{(t)} = \log(p_i^{(t)}(\tilde{y}_i) + \epsilon) - \max_{k \neq \tilde{y}_i} \log(p_i^{(t)}(k) + \epsilon).
$$

This is equivalent to the corresponding logit margin up to numerical precision.

### Area Under the Margin

$$
\mathrm{AUM}_i = \frac{1}{T} \sum_{t=1}^{T} m_i^{(t)}.
$$

### Confidence variability

$$
\mathrm{VarConf}_i = \mathrm{Std}_{t} \left[ p_i^{(t)}(\tilde{y}_i) \right].
$$

### Forgetting count

Let

$$
c_i^{(t)} = \mathbb{1} \left[ \arg\max_k p_i^{(t)}(k) = \tilde{y}_i \right].
$$

Then

$$
\mathrm{Forgetting}_i = \sum_{t=1}^{T-1} \mathbb{1} \left[ c_i^{(t)} = 1 \land c_i^{(t+1)} = 0 \right].
$$

This design keeps the training loop simple and makes descriptor design flexible.

Students can add new descriptors later without retraining the classifier, as long as the required primitive signals were logged.

Possible post-hoc descriptors:

* Area Under the Margin;
* mean assigned-label confidence;
* confidence variability;
* forgetting count;
* mean loss;
* final loss;
* final confidence;
* final margin;
* first learned epoch;
* last forgotten epoch;
* entropy statistics across epochs.

Possible static baselines:

* maximum softmax probability;
* entropy;
* final loss;
* final margin;
* final assigned-label confidence.

The main evaluation target is noisy-label detection and uncertainty ranking.

---

## Stage 2: Descriptor Distillation for Inference-Time Uncertainty

After Stage 1, each sample has a descriptor vector:

$$
s_i = A(\tau_i).
$$

Then freeze the trained encoder and extract final representations:

$$
h_i = f_\theta(x_i).
$$

Train a lightweight descriptor predictor:

$$
g_\phi(h_i) = \hat{s}_i.
$$

The goal is to predict post-hoc training-dynamics descriptors without access to the full training trajectory.

This tests whether historical learning behavior is encoded in the final representation space.

The predicted descriptor vector can be converted into an uncertainty score:

$$
u_i = U(\hat{s}_i).
$$

The score may be:

* hand-designed;
* calibrated on a validation set;
* learned by a shallow model;
* optimized for ranking suspicious samples;
* optimized for selective prediction.

A simple first version may combine normalized predicted descriptors:

$$
u_i = \alpha_1(-\widehat{\mathrm{AUM}}_i) + \alpha_2\widehat{\mathrm{VarConf}}_i + \alpha_3\widehat{\mathrm{Forgetting}}_i + \alpha_4(1-\widehat{\mathrm{MeanConf}}_i).
$$

This formula is not fixed. It is only a reasonable starting point.

A possible additional experiment is uncertainty-aware prediction aggregation.

For weak test-time augmentations or dropout samples,

$$
p_1, p_2, \dots, p_K
$$

can be combined using uncertainty-dependent weights:

$$
\bar{p}(y \mid x) = \frac{\sum_{k=1}^{K} w_k p_k(y \mid x)}{\sum_{k=1}^{K} w_k},
$$

where

$$
w_k = \exp(-\tau u_k)
$$

or another decreasing function of predicted uncertainty.

The question is whether uncertain augmented predictions should contribute less to the final decision.

---

## Stage 3: Conditional Flow Matching Extension

The deterministic descriptor predictor estimates one vector:

$$
g_\phi(h_i) = \hat{s}_i.
$$

The advanced extension studies a probabilistic alternative:

$$
p_\psi(s_i \mid h_i).
$$

A conditional flow-matching model can generate multiple plausible descriptor vectors for the same representation:

$$
\hat{s}_i^{(1)}, \hat{s}_i^{(2)}, \dots, \hat{s}_i^{(M)}.
$$

The variance or disagreement among generated descriptors may provide an additional uncertainty signal.

This stage is exploratory. It should not block the main deterministic pipeline.

The main question is not whether flow matching is fashionable, but whether it adds measurable value beyond deterministic descriptor regression.

---

## Compact Experimental Outputs

Experiment outputs should be designed for later analysis with ChatGPT and for writing the Results section of the paper.

Each important experiment should generate:

* compact result tables;
* plots;
* saved configuration;
* trained checkpoints when relevant;
* `report.md`.

The compact tables should save tokens during later analysis.

Guidelines for compact tables:

* include only essential columns;
* use short method names;
* round metrics to 3-4 meaningful digits;
* keep detailed raw artifacts separately;
* report mean and standard deviation for multi-seed experiments;
* avoid unnecessary per-sample rows in summaries.

Example compact table:

```csv
method,split,seed,auroc,auprc,aurc,ece,acc
msp,worse,0,0.712,0.481,0.184,0.092,0.821
entropy,worse,0,0.728,0.496,0.177,0.087,0.821
aum,worse,0,0.803,0.586,0.142,0.071,0.821
pred_desc,worse,0,0.819,0.604,0.136,0.068,0.821
```

## References

[1] J. Wei, Z. Zhu, H. Cheng, T. Liu, G. Niu, and Y. Liu.  
**Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations.**  
ICLR, 2022. arXiv:2110.12088.

[2] S. Swayamdipta, R. Schwartz, N. A. Smith, and Y. Choi.  
**Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.**  
EMNLP, 2020.

[3] G. Pleiss, T. Zhang, E. R. Elibol, and K. Q. Weinberger.  
**Identifying Mislabeled Data using the Area Under the Margin Ranking.**  
NeurIPS, 2020.

[4] P. Maini, S. Garg, Z. C. Lipton, and J. Z. Kolter.  
**Characterizing Optimization Trajectories under Label Noise.**  
ICML, 2024.

[5] S. Qian, C. M. D. Sa, and C. Cardie.  
**MIND: Decoupling Model-Induced Label Noise via Latent Manifold Disentanglement.**  
arXiv:2605.16081, 2026.
