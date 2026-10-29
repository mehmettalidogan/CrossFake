from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (EER, threshold). EER is where FPR ~= FNR."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thr


def accuracy_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    preds = (np.asarray(y_score) >= threshold).astype(int)
    return float(accuracy_score(np.asarray(y_true).astype(int), preds))


def summarize_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    eer, eer_thr = equal_error_rate(y_true, y_score)
    return {
        "auc": binary_auc(y_true, y_score),
        "eer": eer,
        "eer_threshold": eer_thr,
        "acc@0.5": accuracy_at_threshold(y_true, y_score, 0.5),
        "acc@eer": accuracy_at_threshold(y_true, y_score, eer_thr if not np.isnan(eer_thr) else threshold),
        "n": int(len(y_true)),
        "n_fake": int(np.sum(np.asarray(y_true) == 1)),
        "n_real": int(np.sum(np.asarray(y_true) == 0)),
    }
