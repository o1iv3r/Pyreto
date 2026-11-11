"""Maximum-likelihood estimation of generalized Pareto alphas."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from pyreto._validation import (
    is_positive_finite_number,
)


def gen_pareto_ml_estimator_alpha(
    losses: ArrayLike,
    t: float,
    truncation: float | None = None,
    reporting_thresholds: ArrayLike | None = None,
    is_censored: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    alpha_min: float = 0.001,
    alpha_max: float = 10.0,
) -> np.ndarray:
    """Maximum likelihood estimator for (alpha_ini, alpha_tail) of a generalized Pareto.

    Parameters
    ----------
    losses:
        Observed losses (must be > t, or > reporting_thresholds).
    t:
        Single threshold. Must be positive.
    truncation:
        Upper truncation point (losses must be strictly less than truncation).
    reporting_thresholds:
        Per-loss reporting thresholds. Defaults to t for all losses.
    is_censored:
        Boolean array; True means the loss has been censored at the policy limit.
    weights:
        Per-loss observation weights.
    alpha_min:
        Lower bound for alpha estimates.
    alpha_max:
        Upper bound for alpha estimates.

    Returns
    -------
    np.ndarray
        Array [alpha_ini_hat, alpha_tail_hat].
    """
    losses = np.asarray(losses, dtype=float)
    nan_result = np.full(2, np.nan)

    if not is_positive_finite_number(t):
        warnings.warn("t must be positive.")
        return nan_result
    t = float(t)

    if reporting_thresholds is None:
        rt = np.full(len(losses), t)
    else:
        rt = np.asarray(reporting_thresholds, dtype=float)
        if rt.ndim == 0:
            rt = np.repeat(rt, len(losses))
        if len(rt) == 1:
            rt = np.repeat(rt, len(losses))
        if len(rt) != len(losses):
            warnings.warn("reporting_thresholds must have the same length as losses.")
            return nan_result

    if is_censored is None:
        censored = np.zeros(len(losses), dtype=bool)
    else:
        censored = np.asarray(is_censored, dtype=bool)
        if censored.ndim == 0:
            censored = np.repeat(censored, len(losses))
        if len(censored) == 1:
            censored = np.repeat(censored, len(losses))
        if len(censored) != len(losses):
            warnings.warn("is_censored must have the same length as losses.")
            return nan_result

    if weights is None:
        w = np.ones(len(losses))
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(losses):
            warnings.warn("weights must have the same length as losses.")
            return nan_result

    trunc = np.inf if truncation is None else float(truncation)
    if not np.isinf(trunc) and trunc <= t:
        warnings.warn("truncation must be larger than t.")
        return nan_result
    if np.any(losses >= trunc):
        warnings.warn("Losses must be < truncation.")
        return nan_result

    # Filter: losses > max(t, reporting_threshold)
    threshold = np.maximum(t, rt)
    idx = losses > threshold
    if not np.any(idx):
        warnings.warn("No losses larger than reporting_thresholds and t.")
        return nan_result

    losses = losses[idx]
    w = w[idx]
    rt = np.maximum(rt[idx], t)
    censored = censored[idx]

    has_censored = np.any(censored)

    def neg_ll(a: np.ndarray) -> float:
        # Survival-type terms u(x)^(-alpha_tail) with u(x) = 1 + alpha_ini/alpha_tail * (x/t - 1).
        # Without reporting thresholds u(rt) = 1, and without truncation u(trunc)^(-a) = 0,
        # so this single formula covers every combination of rt / truncation / censoring.
        u_l = 1.0 + a[0] / a[1] * (losses / t - 1.0)
        u_rt = 1.0 + a[0] / a[1] * (rt / t - 1.0)
        u_tr = 1.0 + a[0] / a[1] * (trunc / t - 1.0)
        log_denom = np.log(u_rt ** (-a[1]) - u_tr ** (-a[1]))
        ll = np.log(a[0]) + (-a[1] - 1.0) * np.log(u_l) - log_denom
        if has_censored:
            ll_c = np.log(u_l ** (-a[1]) - u_tr ** (-a[1])) - log_denom
            ll = np.where(censored, ll_c, ll)
        return -float(np.sum(w * ll))

    result = None
    try:
        result = optimize.minimize(
            neg_ll,
            x0=np.array([1.0, 1.0]),
            bounds=[(alpha_min, alpha_max), (alpha_min, alpha_max)],
            method="L-BFGS-B",
            options={"ftol": 1e-12, "gtol": 1e-8},
        )
    except Exception:
        pass

    if result is None or (result.success is False and result.fun == np.inf):
        warnings.warn("No solution found.")
        return nan_result

    return np.array(result.x)
