"""Statistical and machine learning evaluation metrics for Meta-Labeling."""

from __future__ import annotations

import math

from xau_bot.ml.models import ModelMetrics


def compute_log_loss(y_true: list[int], y_probs: list[float], eps: float = 1e-15) -> float:
    """Calculates binary cross-entropy (log-loss)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    if len(y_probs) != n:
        raise ValueError("Lengths of y_true and y_probs must match.")
    total = 0.0
    for y, p in zip(y_true, y_probs, strict=True):
        clipped_p = max(eps, min(1.0 - eps, float(p)))
        total += -(y * math.log(clipped_p) + (1 - y) * math.log(1.0 - clipped_p))
    return total / n


def calculate_roc_auc(y_true: list[int], y_probs: list[float]) -> float:
    """Calculates exact ROC-AUC score using Wilcoxon-Mann-Whitney rank sum algorithm."""
    n = len(y_true)
    if n == 0 or len(y_probs) != n:
        return 0.5

    pos_count = sum(1 for y in y_true if y == 1)
    neg_count = n - pos_count
    if pos_count == 0 or neg_count == 0:
        return 0.5

    paired = sorted(zip(y_probs, y_true, strict=True), key=lambda item: item[0])

    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(paired[j + 1][0] - paired[j][0]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_pos = sum(ranks[idx] for idx in range(n) if paired[idx][1] == 1)
    u_stat = rank_sum_pos - (pos_count * (pos_count + 1)) / 2.0
    auc = u_stat / (pos_count * neg_count)
    return max(0.0, min(1.0, float(auc)))


def calculate_metrics(
    y_true: list[int],
    y_probs: list[float],
    r_values: list[float] | None = None,
    threshold: float = 0.5,
) -> ModelMetrics:
    """Calculates comprehensive classification and financial metrics for meta-label predictions."""
    n = len(y_true)
    if n == 0:
        return ModelMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            roc_auc=0.5,
            brier_score=0.0,
            delta_r=0.0,
            sample_size=0,
            positive_rate=0.0,
        )

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    brier_sum = 0.0

    for yt, yp in zip(y_true, y_probs, strict=True):
        pred_label = 1 if yp >= threshold else 0
        if pred_label == 1 and yt == 1:
            tp += 1
        elif pred_label == 1 and yt == 0:
            fp += 1
        elif pred_label == 0 and yt == 0:
            tn += 1
        else:
            fn += 1
        brier_sum += (yp - float(yt)) ** 2

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    roc_auc = calculate_roc_auc(y_true, y_probs)
    brier_score = brier_sum / n
    positive_rate = sum(y_true) / n

    delta_r = 0.0
    if r_values and len(r_values) == n:
        base_mean_r = sum(r_values) / n
        filtered_r = [r_values[idx] for idx in range(n) if y_probs[idx] >= threshold]
        if filtered_r:
            filtered_mean_r = sum(filtered_r) / len(filtered_r)
            delta_r = filtered_mean_r - base_mean_r

    return ModelMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        roc_auc=roc_auc,
        brier_score=brier_score,
        delta_r=delta_r,
        sample_size=n,
        positive_rate=positive_rate,
    )
