"""Deterministic statistical functions for proportions, Z-score and R-expectancy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist


def two_proportion_z_test(count1: int, n1: int, count2: int, n2: int) -> tuple[float, float]:
    """Computes two-proportion pooled Z-test and two-tailed p-value.

    Returns:
        tuple[float, float]: (z_score, p_value) rounded to 4 and 6 decimal places.
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0

    p1 = count1 / n1
    p2 = count2 / n2

    pooled_p = (count1 + count2) / (n1 + n2)
    if pooled_p <= 0.0 or pooled_p >= 1.0 or p1 == p2:
        return 0.0, 1.0

    variance = pooled_p * (1.0 - pooled_p) * ((1.0 / n1) + (1.0 / n2))
    se = math.sqrt(variance)
    z = (p1 - p2) / se
    p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z)))
    return round(z, 4), round(p_value, 6)


def welch_t_test(sample1: Sequence[float], sample2: Sequence[float]) -> tuple[float, float]:
    """Computes Welch's t-test for two independent samples with unequal variances.

    Returns:
        tuple[float, float]: (t_stat, p_value) rounded to 4 and 6 decimal places.
    """
    n1 = len(sample1)
    n2 = len(sample2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    mean1 = sum(sample1) / n1
    mean2 = sum(sample2) / n2

    var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1)

    se1 = var1 / n1
    se2 = var2 / n2
    denom = math.sqrt(se1 + se2)
    if denom == 0.0:
        return 0.0, 1.0 if mean1 == mean2 else 0.0

    t_stat = (mean2 - mean1) / denom

    df_denom = ((se1**2) / (n1 - 1)) + ((se2**2) / (n2 - 1))
    df = ((se1 + se2) ** 2) / df_denom if df_denom > 0 else 1.0

    z = abs(t_stat)
    z_adj = z * (1.0 - (1.0 / (4.0 * df))) / math.sqrt(1.0 + (z * z) / (2.0 * df)) if df > 2 else z
    p_value = 2.0 * (1.0 - NormalDist().cdf(z_adj))
    p_value = max(0.0, min(1.0, p_value))
    return round(t_stat, 4), round(p_value, 6)


def calculate_expectancy(values: Sequence[float]) -> float:
    """Computes arithmetic mean of realized R values."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def calculate_win_loss_rates(wins: int, losses: int, total: int) -> tuple[float, float]:
    """Computes win rate and loss rate bounded between 0.0 and 1.0."""
    if total <= 0:
        return 0.0, 0.0
    return round(wins / total, 4), round(losses / total, 4)


def fdr_benjamini_hochberg(p_values: Sequence[float], q_threshold: float = 0.05) -> list[bool]:
    """Applies Benjamini-Hochberg procedure for False Discovery Rate (FDR) control.

    Args:
        p_values: Sequence of p-values in [0.0, 1.0].
        q_threshold: False discovery rate threshold in (0.0, 1.0).

    Returns:
        list[bool]: True for hypotheses that pass FDR control, matching input order.
    """
    if not (0.0 < q_threshold < 1.0):
        raise ValueError(f"q_threshold must be between 0.0 and 1.0, got {q_threshold}")

    if not p_values:
        return []

    for p in p_values:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p-values must be between 0.0 and 1.0, got {p}")

    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])

    max_k = 0
    for rank, (_idx, p) in enumerate(indexed_p, start=1):
        critical_value = (rank / m) * q_threshold
        if p <= critical_value:
            max_k = rank

    significant_indices = {
        orig_idx
        for orig_idx, _p in [item for rank, item in enumerate(indexed_p, start=1) if rank <= max_k]
    }

    return [i in significant_indices for i in range(m)]
