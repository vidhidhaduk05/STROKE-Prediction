"""
Shared utilities for the stroke treatment-benefit prediction pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve,
    confusion_matrix,
)


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_metrics(y_true, y_pred_prob, y_pred_label=None, threshold=0.5):
    """Compute the full metric suite used in the paper (Table 2)."""
    if y_pred_label is None:
        y_pred_label = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label, labels=[0, 1]).ravel()
    metrics = {
        "AUC": roc_auc_score(y_true, y_pred_prob),
        "Brier": brier_score_loss(y_true, y_pred_prob),
        "Accuracy": accuracy_score(y_true, y_pred_label),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # recall
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "PPV": tp / (tp + fp) if (tp + fp) > 0 else 0.0,  # precision
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "F1": f1_score(y_true, y_pred_label, zero_division=0),
    }
    return metrics


def _npv(y_true, y_pred):
    """Negative predictive value (kept for backwards compat)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def bootstrap_metrics(y_true, y_pred_prob, n_iter=1000, threshold=0.5, rng=None):
    """Bootstrap 95% CIs for all metrics."""
    if rng is None:
        rng = np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)
    n = len(y_true)
    boot_results = {k: [] for k in ["AUC", "Brier", "Accuracy", "Sensitivity",
                                     "Specificity", "PPV", "NPV", "F1"]}
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_pred_prob[idx], threshold=threshold)
        for k in boot_results:
            boot_results[k].append(m[k])
    summary = {}
    for k, vals in boot_results.items():
        vals = np.array(vals)
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }
    return summary


def smd(group1, group2):
    """Standardized mean difference for covariate balance checking."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    m1, m2 = np.nanmean(g1), np.nanmean(g2)
    v1, v2 = np.nanvar(g1, ddof=1), np.nanvar(g2, ddof=1)
    pooled_sd = np.sqrt((v1 + v2) / 2)
    if pooled_sd == 0:
        return 0.0
    return (m1 - m2) / pooled_sd


def benefit_score(cate_values, min_score=1, max_score=20):
    """
    Map CATE (differential benefit) to a 1-20 clinical score.
    CATE near 0 -> middle score; positive CATE (treatment benefit) -> higher score.
    """
    cate = np.asarray(cate_values, dtype=float)
    c_min, c_max = np.percentile(cate, 1), np.percentile(cate, 99)
    if c_max == c_min:
        return np.full_like(cate, (min_score + max_score) / 2)
    normalized = (cate - c_min) / (c_max - c_min)
    return np.clip(min_score + normalized * (max_score - min_score), min_score, max_score)
