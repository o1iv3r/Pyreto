"""CDF, PDF, quantile and random generation of the piecewise Pareto distribution."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyreto._validation import valid_parameters_piecewise_pareto
from pyreto.pareto import (
    p_pareto,
    r_pareto,
)

# ---------------------------------------------------------------------------
# CDF
# ---------------------------------------------------------------------------


def _p_piecewise_pareto_s(
    x: float,
    t_arr: np.ndarray,
    alpha_arr: np.ndarray,
    trunc: float | None,
    truncation_type: str,
) -> float:
    """Scalar CDF of the piecewise Pareto distribution (mirrors pPiecewisePareto_s in R)."""
    n = len(t_arr)

    if n == 1:
        return float(p_pareto(x, t_arr[0], alpha_arr[0], truncation=trunc))

    if x <= t_arr[0]:
        return 0.0

    def _untrunc_cdf(x_val: float, t_v: np.ndarray, a_v: np.ndarray) -> float:
        """CDF of the piecewise Pareto without truncation, evaluated at x_val."""
        t_filt = np.append(t_v[t_v < x_val], x_val)
        k = len(t_filt) - 1
        if k == 0:
            return 0.0
        factors = t_filt[1:] / t_filt[:-1]
        return 1.0 - float(np.prod((1.0 / factors) ** a_v[:k]))

    if trunc is None:
        return _untrunc_cdf(x, t_arr, alpha_arr)

    if truncation_type == "wd":
        if x >= trunc:
            return 1.0
        factors_all = t_arr[1:] / t_arr[:-1]
        ep_last = float(np.prod((1.0 / factors_all) ** alpha_arr[:-1]))
        denom = 1.0 - ep_last * (t_arr[-1] / trunc) ** alpha_arr[-1]
        return _untrunc_cdf(x, t_arr, alpha_arr) / denom

    else:  # "lp"
        if x >= trunc:
            return 1.0
        if x <= t_arr[-1]:
            return _untrunc_cdf(x, t_arr, alpha_arr)
        # x is in the last (truncated) piece
        factors_all = t_arr[1:] / t_arr[:-1]
        excess_prob_last = float(np.prod((1.0 / factors_all) ** alpha_arr[:-1]))
        lp_frac = (1.0 - (t_arr[-1] / x) ** alpha_arr[-1]) / (
            1.0 - (t_arr[-1] / trunc) ** alpha_arr[-1]
        )
        return 1.0 - excess_prob_last * (1.0 - lp_frac)


def p_piecewise_pareto(
    x: ArrayLike,
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None = None,
    truncation_type: str = "lp",
) -> np.ndarray:
    """CDF of the piecewise Pareto distribution.

    Parameters
    ----------
    x:
        Values at which to evaluate the CDF.
    t:
        Thresholds of the piecewise Pareto distribution (positive, strictly ascending).
    alpha:
        Pareto alphas for each piece (non-negative, last entry must be positive).
    truncation:
        Upper truncation point. None means no truncation.
    truncation_type:
        ``'lp'`` — truncated Pareto in the last piece;
        ``'wd'`` — whole distribution truncated and renormalised.

    Returns
    -------
    np.ndarray
        CDF values.
    """
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    if not valid_parameters_piecewise_pareto(
        t_arr, alpha_arr, truncation, truncation_type, comment=True
    ):
        return np.full(x_arr.shape if x_arr.ndim > 0 else (), np.nan)
    _vfun = np.vectorize(
        lambda xi: _p_piecewise_pareto_s(xi, t_arr, alpha_arr, truncation, truncation_type)
    )
    return _vfun(x_arr)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


def _d_piecewise_pareto_s(
    x: float,
    t_arr: np.ndarray,
    alpha_arr: np.ndarray,
    trunc: float | None,
    truncation_type: str,
) -> float:
    """Scalar density of the piecewise Pareto (mirrors dPiecewisePareto_s in R)."""
    n = len(t_arr)

    if n == 1:
        # Delegate to scalar Pareto density
        if trunc is not None and x >= trunc:
            return 0.0
        if x < t_arr[0]:
            return 0.0
        # Pareto density: alpha/x * (t/x)^alpha, renormalised for truncation
        raw = alpha_arr[0] / x * (t_arr[0] / x) ** alpha_arr[0]
        if trunc is None:
            return raw
        norm = 1.0 - (t_arr[0] / trunc) ** alpha_arr[0]
        return raw / norm

    if x < t_arr[0]:
        return 0.0

    if trunc is None:
        t_sub = t_arr[t_arr <= x]
        k = len(t_sub)
        a_sub = alpha_arr[:k]
        excess_prob = 1.0 - _p_piecewise_pareto_s(x, t_sub, a_sub, None, "lp")
        return excess_prob * a_sub[-1] / x

    if truncation_type == "wd":
        if x >= trunc:
            return 0.0
        # scaling = 1 / CDF_untruncated_at_trunc
        cdf_trunc = _p_piecewise_pareto_s(trunc, t_arr, alpha_arr, None, "lp")
        scaling = 1.0 / cdf_trunc
        t_sub = t_arr[t_arr <= x]
        k = len(t_sub)
        a_sub = alpha_arr[:k]
        excess_prob = 1.0 - _p_piecewise_pareto_s(x, t_sub, a_sub, None, "lp")
        return excess_prob * a_sub[-1] / x * scaling

    else:  # "lp"
        if x >= trunc:
            return 0.0
        if x <= t_arr[-1]:
            t_sub = t_arr[t_arr <= x]
            k = len(t_sub)
            a_sub = alpha_arr[:k]
            excess_prob = 1.0 - _p_piecewise_pareto_s(x, t_sub, a_sub, None, "lp")
            return excess_prob * a_sub[-1] / x
        # x is beyond last threshold — truncated last piece
        ep_at_tn = 1.0 - _p_piecewise_pareto_s(t_arr[-1], t_arr, alpha_arr, None, "lp")
        ep_at_trunc = 1.0 - _p_piecewise_pareto_s(trunc, t_arr, alpha_arr, None, "lp")
        scaling = ep_at_tn / (ep_at_tn - ep_at_trunc)
        # Pareto density at x with threshold t[-1], alpha[-1]
        pareto_density = alpha_arr[-1] / x * (t_arr[-1] / x) ** alpha_arr[-1]
        return ep_at_tn * scaling * pareto_density


def d_piecewise_pareto(
    x: ArrayLike,
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None = None,
    truncation_type: str = "lp",
) -> np.ndarray:
    """Density of the piecewise Pareto distribution.

    Parameters
    ----------
    x:
        Values at which to evaluate the density.
    t:
        Thresholds of the piecewise Pareto distribution.
    alpha:
        Pareto alphas for each piece.
    truncation:
        Upper truncation point.
    truncation_type:
        ``'lp'`` or ``'wd'``.

    Returns
    -------
    np.ndarray
        Density values.
    """
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    if not valid_parameters_piecewise_pareto(
        t_arr, alpha_arr, truncation, truncation_type, comment=True
    ):
        return np.full(x_arr.shape if x_arr.ndim > 0 else (), np.nan)
    _vfun = np.vectorize(
        lambda xi: _d_piecewise_pareto_s(xi, t_arr, alpha_arr, truncation, truncation_type)
    )
    return _vfun(x_arr)


# ---------------------------------------------------------------------------
# Quantile
# ---------------------------------------------------------------------------


def _q_piecewise_pareto_s(
    y: float,
    t_arr: np.ndarray,
    alpha_arr: np.ndarray,
    trunc: float | None,
    truncation_type: str,
) -> float:
    """Scalar quantile of the piecewise Pareto (mirrors qPiecewisePareto_s in R)."""
    n = len(t_arr)

    if n == 1:
        return float(p_pareto(y, t_arr[0], alpha_arr[0], truncation=trunc))

    if y == 1.0:
        return trunc if trunc is not None else np.inf

    # CDF values at each threshold, without truncation
    cdf_at_t = np.array([_p_piecewise_pareto_s(ti, t_arr, alpha_arr, None, "lp") for ti in t_arr])

    y_eff = y  # effective quantile level to invert
    if trunc is not None:
        if np.isinf(trunc):
            cdf_at_trunc = 1.0
        else:
            cdf_at_trunc = _p_piecewise_pareto_s(trunc, t_arr, alpha_arr, None, "lp")

        if truncation_type == "wd":
            y_eff = y * cdf_at_trunc
        else:  # "lp"
            cdf_tn = cdf_at_t[-1]
            if y > cdf_tn:
                y_eff = cdf_tn + (y - cdf_tn) * (cdf_at_trunc - cdf_tn) / (1.0 - cdf_tn)

    # Find which piece y_eff falls in (0-indexed)
    k0 = int(np.sum(cdf_at_t <= y_eff)) - 1
    k0 = max(k0, 0)
    t0 = t_arr[k0]
    f0 = cdf_at_t[k0]

    return t0 / ((1.0 - y_eff) / (1.0 - f0)) ** (1.0 / alpha_arr[k0])


def q_piecewise_pareto(
    p: ArrayLike,
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None = None,
    truncation_type: str = "lp",
) -> np.ndarray:
    """Quantile function of the piecewise Pareto distribution.

    Parameters
    ----------
    p:
        Probability values in (0, 1].
    t:
        Thresholds of the piecewise Pareto distribution.
    alpha:
        Pareto alphas for each piece.
    truncation:
        Upper truncation point.
    truncation_type:
        ``'lp'`` or ``'wd'``.

    Returns
    -------
    np.ndarray
        Quantile values.
    """
    p_arr = np.asarray(p, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    if not valid_parameters_piecewise_pareto(
        t_arr, alpha_arr, truncation, truncation_type, comment=True
    ):
        return np.full(p_arr.shape if p_arr.ndim > 0 else (), np.nan)
    _vfun = np.vectorize(
        lambda yi: _q_piecewise_pareto_s(yi, t_arr, alpha_arr, truncation, truncation_type)
    )
    return _vfun(p_arr)


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------


def r_piecewise_pareto(
    n: int,
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None = None,
    truncation_type: str = "lp",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random samples from the piecewise Pareto distribution.

    Parameters
    ----------
    n:
        Number of samples.
    t:
        Thresholds of the piecewise Pareto distribution.
    alpha:
        Pareto alphas for each piece.
    truncation:
        Upper truncation point.
    truncation_type:
        ``'lp'`` or ``'wd'``.
    rng:
        Numpy random generator. None uses a new default_rng.

    Returns
    -------
    np.ndarray
        Array of n samples.
    """
    t_arr = np.asarray(t, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    if not valid_parameters_piecewise_pareto(
        t_arr, alpha_arr, truncation, truncation_type, comment=True
    ):
        return np.full(n, np.nan)

    if rng is None:
        rng = np.random.default_rng()

    k = len(t_arr)

    if k == 1:
        return r_pareto(n, t_arr[0], alpha_arr[0], truncation=truncation, rng=rng)

    # Compute unnormalised piece selection probabilities
    factors_t = t_arr[1:] / t_arr[:-1]
    excess_prob = np.ones(k)
    excess_prob[1:] = np.cumprod((1.0 / factors_t) ** alpha_arr[:-1])

    prob_pieces = np.empty(k)
    prob_pieces[:-1] = -np.diff(excess_prob)
    prob_pieces[-1] = excess_prob[-1]

    if truncation is not None and truncation_type == "wd":
        prob_pieces[-1] = excess_prob[-1] * (1.0 - (t_arr[-1] / truncation) ** alpha_arr[-1])
        prob_pieces /= prob_pieces.sum()

    # Sample which piece each observation is drawn from
    piece_idx = rng.choice(k, size=n, replace=True, p=prob_pieces / prob_pieces.sum())

    result = np.empty(n)
    for i in range(k):
        mask = piece_idx == i
        m = int(mask.sum())
        if m == 0:
            continue
        if i == k - 1 and truncation is not None:
            cdf_max = 1.0 - (t_arr[i] / truncation) ** alpha_arr[i]
        else:
            cdf_max = 1.0 - (t_arr[i] / t_arr[i + 1]) ** alpha_arr[i] if i < k - 1 else 1.0
        u = rng.uniform(0.0, cdf_max, m)
        result[mask] = t_arr[i] / (1.0 - u) ** (1.0 / alpha_arr[i])

    return result
