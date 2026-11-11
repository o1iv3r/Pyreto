"""Maximum-likelihood estimation of piecewise Pareto alphas."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from pyreto.pareto import (
    pareto_ml_estimator_alpha,
)

# ---------------------------------------------------------------------------
# ML estimator
# ---------------------------------------------------------------------------


def piecewise_pareto_ml_estimator_alpha(
    losses: ArrayLike,
    t: ArrayLike,
    truncation: float | None = None,
    truncation_type: str = "lp",
    reporting_thresholds: ArrayLike | None = None,
    is_censored: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    alpha_min: float = 0.001,
    alpha_max: float = 10.0,
) -> np.ndarray:
    """Maximum likelihood estimator for piecewise Pareto alphas.

    Parameters
    ----------
    losses:
        Observed losses.
    t:
        Thresholds of the piecewise Pareto distribution.
    truncation:
        Upper truncation point. None means no truncation.
    truncation_type:
        ``'lp'`` or ``'wd'``.
    reporting_thresholds:
        Per-loss reporting thresholds.
    is_censored:
        Boolean array indicating censored observations.
    weights:
        Per-loss weights.
    alpha_min:
        Lower bound for optimisation (truncated case only).
    alpha_max:
        Upper bound for optimisation (truncated case only).

    Returns
    -------
    np.ndarray
        MLE of alpha for each piece (length k).
    """
    losses_arr = np.asarray(losses, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    k = len(t_arr)

    trunc_eff = np.inf if truncation is None else float(truncation)

    # --- Default optional inputs ---
    n_orig = len(losses_arr)
    if reporting_thresholds is None:
        rt_arr = np.full(n_orig, t_arr[0])
    else:
        rt_arr = np.asarray(reporting_thresholds, dtype=float)
        if rt_arr.ndim == 0:
            rt_arr = np.full(n_orig, float(rt_arr))
        elif len(rt_arr) == 1:
            rt_arr = np.full(n_orig, rt_arr[0])

    if is_censored is None:
        cens_arr = np.zeros(n_orig, dtype=bool)
    else:
        cens_arr = np.asarray(is_censored, dtype=bool)
        if cens_arr.ndim == 0:
            cens_arr = np.full(n_orig, bool(cens_arr))
        elif len(cens_arr) == 1:
            cens_arr = np.full(n_orig, cens_arr[0])

    if weights is None:
        w_arr = np.ones(n_orig)
    else:
        w_arr = np.asarray(weights, dtype=float)

    if k == 1:
        alpha_single = pareto_ml_estimator_alpha(
            losses_arr,
            t_arr[0],
            truncation=truncation,
            reporting_thresholds=rt_arr,
            is_censored=cens_arr,
            weights=w_arr,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        return np.array([alpha_single])

    # --- Filter: keep only losses > max(t[0], rt) ---
    idx = losses_arr > np.maximum(t_arr[0], rt_arr)
    if not idx.any():
        warnings.warn("No losses larger than reporting_thresholds and t[0].")
        return np.full(k, np.nan)

    ls = losses_arr[idx]
    w = w_arr[idx]
    rt = np.maximum(rt_arr[idx], t_arr[0])
    cens = cens_arr[idx]

    if ls.max() <= t_arr[-1]:
        warnings.warn("Number of losses > max(t) must be positive.")
        return np.full(k, np.nan)

    has_censored = bool(cens.any())

    # Extended threshold array (k+1 elements, last = effective truncation)
    t_ext = np.append(t_arr, trunc_eff)

    alpha_hat = np.zeros(k)

    # Closed-form estimator per piece. Censored losses count in the exposure
    # (denominator) but not as observed claims (numerator). Without reporting
    # thresholds rt == t[0], so the rt terms below drop out.
    w_nc = np.where(cens, 0.0, w)
    for i in range(k):
        reported = rt < t_ext[i + 1]
        mask = (ls >= t_ext[i]) & reported
        w_nc_ip1 = w_nc[(ls >= t_ext[i + 1]) & reported].sum()
        exposure = np.log(np.minimum(ls[mask], t_ext[i + 1]) / np.maximum(t_ext[i], rt[mask]))
        alpha_hat[i] = (w_nc[mask].sum() - w_nc_ip1) / np.sum(w[mask] * exposure)

    # --- Truncation adjustment ---
    if np.isinf(trunc_eff):
        return alpha_hat

    if truncation_type == "lp":
        alpha_hat[-1] = pareto_ml_estimator_alpha(
            ls,
            t_arr[-1],
            truncation=truncation,
            reporting_thresholds=rt,
            is_censored=cens,
            weights=w,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        return alpha_hat

    # --- "wd" truncation: joint optimisation ---
    idx_l = np.clip(np.searchsorted(t_arr, ls, side="right") - 1, 0, k - 1)
    idx_rt = np.clip(np.searchsorted(t_arr, rt, side="right") - 1, 0, k - 1)

    if not has_censored:

        def _nll(alpha_v: np.ndarray) -> float:
            ratios = (t_arr / t_ext[1:]) ** alpha_v
            surv = np.ones(k + 1)
            surv[1:] = np.cumprod(ratios)
            surv_l = surv[idx_l]
            surv_rt = surv[idx_rt] * (t_arr[idx_rt] / rt) ** alpha_v[idx_rt]
            surv_trunc = surv[-1]
            a_l = alpha_v[idx_l]
            t_l = t_arr[idx_l]
            dens = surv_l * a_l / t_l * (t_l / ls) ** (a_l + 1.0)
            return -float(np.sum(w * (np.log(dens) - np.log(surv_rt - surv_trunc))))
    else:

        def _nll(alpha_v: np.ndarray) -> float:
            ratios = (t_arr / t_ext[1:]) ** alpha_v
            surv = np.ones(k + 1)
            surv[1:] = np.cumprod(ratios)
            surv_l = surv[idx_l]
            surv_at_loss = surv_l * (t_arr[idx_l] / ls) ** alpha_v[idx_l]
            surv_rt = surv[idx_rt] * (t_arr[idx_rt] / rt) ** alpha_v[idx_rt]
            surv_trunc = surv[-1]
            a_l = alpha_v[idx_l]
            t_l = t_arr[idx_l]
            dens = surv_l * a_l / t_l * (t_l / ls) ** (a_l + 1.0)
            return -float(
                np.sum(
                    w
                    * np.where(
                        cens,
                        np.log(surv_at_loss - surv_trunc) - np.log(surv_rt - surv_trunc),
                        np.log(dens) - np.log(surv_rt - surv_trunc),
                    )
                )
            )

    res = optimize.minimize(
        _nll,
        np.ones(k),
        bounds=[(alpha_min, alpha_max)] * k,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-10},
    )
    if not res.success:
        warnings.warn("No solution found for wd ML estimator.")
        return np.full(k, np.nan)

    return res.x
