"""Pareto extrapolation, alpha matching and maximum-likelihood estimation."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from pyreto.pareto.distribution import p_pareto
from pyreto.pareto.layer import _pareto_layer_mean_scalar, pareto_layer_mean

# ---------------------------------------------------------------------------
# Extrapolation and alpha-finding
# ---------------------------------------------------------------------------


def _pareto_extrapolation_scalar(
    cover_1: float,
    ap_1: float,
    cover_2: float,
    ap_2: float,
    alpha: float,
    exp_loss_1: float,
    truncation: float,
) -> float:
    """Scalar Pareto extrapolation."""
    smaller_ap = min(ap_1, ap_2)

    lm1 = _pareto_layer_mean_scalar(cover_1, ap_1, alpha, smaller_ap, truncation)
    if np.isinf(lm1):
        warnings.warn("Pareto layer mean of layer 1 must be finite.")
        return float("nan")

    lm2 = _pareto_layer_mean_scalar(cover_2, ap_2, alpha, smaller_ap, truncation)
    return exp_loss_1 * lm2 / lm1


def pareto_extrapolation(
    cover_1: ArrayLike,
    attachment_point_1: ArrayLike,
    cover_2: ArrayLike,
    attachment_point_2: ArrayLike,
    alpha: ArrayLike,
    exp_loss_1: ArrayLike | None = None,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Extrapolate expected layer loss using a Pareto distribution.

    Parameters
    ----------
    cover_1:
        Cover of the source layer.
    attachment_point_1:
        Attachment point of the source layer.
    cover_2:
        Cover of the target layer.
    attachment_point_2:
        Attachment point of the target layer.
    alpha:
        Pareto shape parameter.
    exp_loss_1:
        Expected loss of the source layer. If None, defaults to 1 (ratio mode).
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        Expected losses of the target layer.
    """
    _vfun = np.vectorize(_pareto_extrapolation_scalar)
    c1 = np.asarray(cover_1, dtype=float)
    ap1 = np.asarray(attachment_point_1, dtype=float)
    c2 = np.asarray(cover_2, dtype=float)
    ap2 = np.asarray(attachment_point_2, dtype=float)
    a = np.asarray(alpha, dtype=float)
    el1 = np.asarray(exp_loss_1, dtype=float) if exp_loss_1 is not None else np.ones_like(c1)
    trunc = (
        np.asarray(truncation, dtype=float) if truncation is not None else np.full_like(c1, np.inf)
    )
    return _vfun(c1, ap1, c2, ap2, a, el1, trunc)


def pareto_find_alpha_btw_layers(
    cover_1: float,
    attachment_point_1: float,
    exp_loss_1: float,
    cover_2: float,
    attachment_point_2: float,
    exp_loss_2: float,
    max_alpha: float = 100.0,
    tolerance: float = 1e-10,
    truncation: float | None = None,
) -> float:
    """Find Pareto alpha that matches two layer expected losses.

    Parameters
    ----------
    cover_1, attachment_point_1, exp_loss_1:
        Source layer geometry and expected loss.
    cover_2, attachment_point_2, exp_loss_2:
        Target layer geometry and expected loss.
    max_alpha:
        Upper search bound.
    tolerance:
        Root-finding tolerance.
    truncation:
        Upper truncation point.

    Returns
    -------
    float
        Pareto alpha.
    """
    trunc_val = float(truncation) if truncation is not None else np.inf
    finite_trunc = np.isfinite(trunc_val)
    min_alpha = tolerance if finite_trunc else 0.0

    eff_c1 = min(trunc_val - attachment_point_1, cover_1) if finite_trunc else cover_1
    eff_c2 = min(trunc_val - attachment_point_2, cover_2) if finite_trunc else cover_2

    def _f(alpha: float) -> float:
        return (
            float(
                pareto_extrapolation(
                    eff_c1,
                    attachment_point_1,
                    eff_c2,
                    attachment_point_2,
                    alpha,
                    exp_loss_1,
                    truncation,
                )
            )
            - exp_loss_2
        )

    result: float | None = None
    is_inf_cover = np.isinf(cover_1) or np.isinf(cover_2)
    search_lo = 1.0 + tolerance if is_inf_cover else min_alpha
    try:
        result = optimize.brentq(_f, search_lo, max_alpha, xtol=tolerance)
    except ValueError:
        # brentq failed — check boundary cases
        f_max = _f(max_alpha)
        ap1, ap2 = attachment_point_1, attachment_point_2
        ep1, ep2 = ap1 + cover_1, ap2 + cover_2
        if not is_inf_cover:
            if ap1 > ap2 and ep1 >= ep2 and f_max < 0:
                result = max_alpha
            elif ap1 < ap2 and ep1 <= ep2 and f_max > 0:
                result = max_alpha
        else:
            if np.isinf(cover_1) and np.isinf(cover_2):
                if ap1 < ap2 and f_max > 0:
                    result = max_alpha
                elif ap1 > ap2 and f_max < 0:
                    result = max_alpha
            elif np.isinf(cover_1) and ap1 > ap2 and f_max < 0:
                result = max_alpha
            elif np.isinf(cover_2) and ap1 < ap2 and f_max > 0:
                result = max_alpha

    if result is None:
        warnings.warn("pareto_find_alpha_btw_layers: did not find a solution.")
        return float("nan")
    return result


def pareto_find_alpha_btw_fq_layer(
    threshold: float,
    frequency: float,
    cover: float,
    attachment_point: float,
    exp_loss: float,
    max_alpha: float = 100.0,
    tolerance: float = 1e-10,
    truncation: float | None = None,
) -> float:
    """Find Pareto alpha matching an excess frequency and a layer expected loss.

    Parameters
    ----------
    threshold:
        Frequency threshold.
    frequency:
        Expected frequency excess of threshold.
    cover:
        Cover of the layer.
    attachment_point:
        Attachment point of the layer.
    exp_loss:
        Expected loss of the layer.
    max_alpha:
        Upper search bound.
    tolerance:
        Root-finding tolerance.
    truncation:
        Upper truncation point.

    Returns
    -------
    float
        Pareto alpha.
    """
    trunc_val = float(truncation) if truncation is not None else np.inf
    finite_trunc = np.isfinite(trunc_val)
    eff_cover = min(cover, trunc_val - attachment_point) if finite_trunc else cover

    def _f(alpha: float) -> float:
        if attachment_point < threshold:
            denom = float(1 - p_pareto(threshold, attachment_point, alpha, truncation=truncation))
            fq_factor = np.inf if denom == 0.0 else 1.0 / denom
        else:
            fq_factor = float(
                1 - p_pareto(attachment_point, threshold, alpha, truncation=truncation)
            )

        lm = float(pareto_layer_mean(eff_cover, attachment_point, alpha, truncation=truncation))
        return lm * fq_factor * frequency - exp_loss

    if np.isinf(cover):
        min_alpha = 1.0 + tolerance
    elif finite_trunc:
        min_alpha = tolerance
    else:
        min_alpha = 0.0

    # Avoid infinite f(max_alpha)
    while np.isinf(_f(max_alpha)):
        max_alpha /= 2

    result: float | None = None
    try:
        result = optimize.brentq(_f, min_alpha, max_alpha, xtol=tolerance)
    except ValueError:
        f_max = _f(max_alpha)
        if attachment_point >= threshold and f_max > 0:
            result = max_alpha
        elif attachment_point + cover <= threshold and f_max < 0:
            result = max_alpha

    if result is None:
        warnings.warn("pareto_find_alpha_btw_fq_layer: did not find a solution.")
        return float("nan")
    return result


def pareto_find_alpha_btw_fqs(
    threshold_1: float,
    frequency_1: float,
    threshold_2: float,
    frequency_2: float,
    max_alpha: float = 100.0,
    tolerance: float = 1e-10,
    truncation: float | None = None,
) -> float:
    """Find Pareto alpha matching two excess frequencies.

    Parameters
    ----------
    threshold_1, frequency_1:
        First frequency point.
    threshold_2, frequency_2:
        Second frequency point.
    max_alpha:
        Upper search bound.
    tolerance:
        Root-finding tolerance.
    truncation:
        Upper truncation point.

    Returns
    -------
    float
        Pareto alpha.
    """
    # Ensure threshold_1 <= threshold_2
    if threshold_1 > threshold_2:
        threshold_1, frequency_1, threshold_2, frequency_2 = (
            threshold_2,
            frequency_2,
            threshold_1,
            frequency_1,
        )

    finite_trunc = truncation is not None and np.isfinite(truncation)

    if truncation is None or not finite_trunc:
        return np.log(frequency_1 / frequency_2) / np.log(threshold_2 / threshold_1)

    # Truncated case — root-find
    def _f(alpha: float) -> float:
        surv = float(1 - p_pareto(threshold_2, threshold_1, alpha, truncation=truncation))
        return surv - frequency_2 / frequency_1

    result: float | None = None
    min_alpha = tolerance
    try:
        if _f(max_alpha) > 0:
            result = max_alpha
        else:
            result = optimize.brentq(_f, min_alpha, max_alpha, xtol=tolerance)
    except ValueError:
        pass

    if result is None:
        warnings.warn("pareto_find_alpha_btw_fqs: did not find a solution.")
        return float("nan")
    return result


# ---------------------------------------------------------------------------
# Maximum likelihood estimator
# ---------------------------------------------------------------------------


def pareto_ml_estimator_alpha(
    losses: ArrayLike,
    t: float,
    truncation: float | None = None,
    reporting_thresholds: ArrayLike | None = None,
    is_censored: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    alpha_min: float = 0.001,
    alpha_max: float = 10.0,
) -> float:
    """Maximum likelihood estimator for the Pareto shape parameter alpha.

    Parameters
    ----------
    losses:
        Observed losses.
    t:
        Threshold (all losses must be > t to be used).
    truncation:
        Upper truncation of the Pareto distribution. None means no truncation.
    reporting_thresholds:
        Per-loss reporting thresholds. Losses below their threshold are ignored.
    is_censored:
        Boolean array indicating censored losses.
    weights:
        Per-loss weights.
    alpha_min:
        Lower bound for optimisation (truncated case only).
    alpha_max:
        Upper bound for optimisation (truncated case only).

    Returns
    -------
    float
        MLE of alpha.
    """
    losses_arr = np.asarray(losses, dtype=float)
    n_orig = len(losses_arr)
    t_arr = np.full(n_orig, float(t))

    if reporting_thresholds is not None:
        rt = np.asarray(reporting_thresholds, dtype=float)
        t_arr = np.maximum(t_arr, rt)

    if weights is None:
        w = np.ones(n_orig)
    else:
        w = np.asarray(weights, dtype=float)

    if is_censored is None:
        censored = np.zeros(n_orig, dtype=bool)
    else:
        censored = np.asarray(is_censored, dtype=bool)

    # Keep only losses strictly above (possibly adjusted) threshold
    index = losses_arr > t_arr
    losses_arr = losses_arr[index]
    w = w[index]
    t_arr = t_arr[index]
    censored = censored[index]

    if len(losses_arr) == 0:
        warnings.warn("No loss is larger than t and the specific reporting_threshold.")
        return float("nan")

    trunc = np.inf if truncation is None else float(truncation)

    has_censored = bool(np.any(censored))

    if not has_censored:
        alpha_hat = float(np.sum(w) / np.sum(w * np.log(losses_arr / t_arr)))
        if np.isinf(trunc):
            return alpha_hat

        def _nll(alpha: float) -> float:
            log_pdf = np.log(alpha) + alpha * np.log(t_arr) - (alpha + 1) * np.log(losses_arr)
            log_norm = np.log(1 - (t_arr / trunc) ** alpha)
            return -float(np.sum(w * (log_pdf - log_norm)))

    else:
        alpha_hat = float(np.sum(w[~censored]) / np.sum(w * np.log(losses_arr / t_arr)))
        if np.isinf(trunc):
            return alpha_hat

        def _nll(alpha: float) -> float:
            log_norm = np.log(1 - (t_arr / trunc) ** alpha)
            ll_uncensored = (
                np.log(alpha) + alpha * np.log(t_arr) - (alpha + 1) * np.log(losses_arr) - log_norm
            )
            ll_censored = (
                np.log((t_arr / losses_arr) ** alpha - (t_arr / trunc) ** alpha) - log_norm
            )
            ll = np.where(censored, ll_censored, ll_uncensored)
            return -float(np.sum(w * ll))

    result = optimize.minimize_scalar(_nll, bounds=(alpha_min, alpha_max), method="bounded")
    if not result.success and result.message not in ("Solution found.", ""):
        warnings.warn(f"pareto_ml_estimator_alpha optimisation: {result.message}")
    return float(result.x)
