"""Controlled representation ablation, distinct from real-trajectory separation.

Both detectors use the same fit/evaluation IDs and target. Dynamic inputs must
be predicted trajectories of a meta-model that did not train on either set.
"""

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score


@torch.no_grad()
def paired_representations(model, images, candidate_labels):
    """Extract once; share image features and class embedding across both arms."""
    if model.features.training:
        raise ValueError("Matched ablation requires an eval-mode frozen backbone")
    if any(p.requires_grad for p in model.features.parameters()):
        raise ValueError("Matched ablation requires frozen backbone parameters")
    features = model.extract_features(images)
    class_features = model.class_emb(candidate_labels)
    static = torch.cat([features, class_features], dim=1)
    dynamic = model.predict_from_features(features, candidate_labels)
    return static.cpu().numpy(), dynamic.cpu().numpy()


def evaluate_matched_lda(
    static, dynamic, labels, sample_ids, fit_ids, eval_ids, meta_train_ids
):
    """Fit identical LDA settings and evaluate each on the same unseen samples."""
    static, dynamic, labels = (
        np.asarray(static),
        np.asarray(dynamic),
        np.asarray(labels),
    )
    ids, fit_ids, eval_ids, meta_train_ids = [
        np.asarray(x) for x in (sample_ids, fit_ids, eval_ids, meta_train_ids)
    ]
    if static.ndim != 2 or dynamic.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected two feature matrices and a one-dimensional target")
    if (
        len(ids) != len(set(ids.tolist()))
        or len(ids) != len(labels)
        or len(static) != len(ids)
        or len(dynamic) != len(ids)
    ):
        raise ValueError("Sample IDs and paired representations must align uniquely")
    fit, evaluation, trained = map(
        lambda x: set(x.tolist()), (fit_ids, eval_ids, meta_train_ids)
    )
    if not fit or not evaluation or fit & evaluation:
        raise ValueError("Fit/evaluation IDs must be nonempty and disjoint")
    if (fit | evaluation) & trained:
        raise ValueError("Ablation fit/evaluation IDs overlap meta-model training")
    if not (fit | evaluation) <= set(ids.tolist()):
        raise ValueError("Requested sample IDs are absent from paired arrays")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(
            "Ablation requires binary 0/1 labels with identical meaning in both arms"
        )
    if not np.isfinite(static).all() or not np.isfinite(dynamic).all():
        raise ValueError("Representations must be finite")
    index = {sample_id: i for i, sample_id in enumerate(ids.tolist())}
    fit_rows = [index[i] for i in fit_ids.tolist()]
    eval_rows = [index[i] for i in eval_ids.tolist()]
    if len(fit) != len(fit_rows) or len(evaluation) != len(eval_rows):
        raise ValueError("Split IDs must not contain duplicates")
    if len(np.unique(labels[fit_rows])) != 2 or len(np.unique(labels[eval_rows])) != 2:
        raise ValueError("Each split must contain both target classes")
    result = {}
    for name, features in [("static_shared", static), ("predicted_traj", dynamic)]:
        detector = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        detector.fit(features[fit_rows], labels[fit_rows])
        probability = detector.predict_proba(features[eval_rows])[
            :, list(detector.classes_).index(1)
        ]
        result[name] = {
            "acc": float(
                accuracy_score(labels[eval_rows], detector.predict(features[eval_rows]))
            ),
            "auroc": float(roc_auc_score(labels[eval_rows], probability)),
            "fit_n": len(fit_rows),
            "eval_n": len(eval_rows),
        }
    return result
