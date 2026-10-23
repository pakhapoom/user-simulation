"""Metrics for slates with exactly one positive.

Ties are resolved as their expectation under random ordering, so a predictor
that outputs constant scores earns exactly chance-level numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def slate_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pos = scores[labels == 1][0]
    neg = scores[labels == 0]
    greater = int((neg > pos).sum())
    ties = int((neg == pos).sum())
    rank = 1 + greater + ties / 2  # expected rank of the positive
    return {
        "hit@1": (greater == 0) / (1 + ties),
        "mrr": 1 / rank,
        "ndcg": 1 / np.log2(rank + 1),
        "gauc": ((neg < pos).sum() + ties / 2) / len(neg),
    }


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Weighted gap between predicted and observed click rate across probability bins."""
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        error += in_bin.mean() * abs(probs[in_bin].mean() - labels[in_bin].mean())
    return float(error)


def summarize(
    per_slate: list[dict[str, float]],
    scores: np.ndarray,
    labels: np.ndarray,
    probabilistic: bool,
) -> dict[str, float]:
    summary = {key: float(np.mean([m[key] for m in per_slate])) for key in per_slate[0]}
    # Pooled AUC compares scores across users, i.e. pointwise click prediction.
    summary["auc"] = float(roc_auc_score(labels, scores))
    if probabilistic:
        summary["log_loss"] = log_loss(scores, labels)
        summary["ece"] = expected_calibration_error(scores, labels)
    return summary


def bootstrap_ci(
    values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of per-slate `values` with a (1 - alpha) percentile interval.

    Resamples whole groups (users), since one user's slates are not independent. For a
    paired comparison pass the per-slate difference between two predictors on the same
    episodes.
    """
    _, codes = np.unique(groups, return_inverse=True)
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    picks = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
