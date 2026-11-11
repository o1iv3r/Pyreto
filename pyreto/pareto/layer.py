"""Expected value, second moment and variance of Pareto layer losses."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from pyreto.pareto.distribution import p_pareto

# ---------------------------------------------------------------------------
# Layer moments
# ---------------------------------------------------------------------------


def _pareto_layer_mean_scalar(
    cover: float,
    attachment: float,
    alpha: float,
    t: float,
    truncation: float,
) -> float:
    """Scalar Pareto layer mean. truncation=inf means no truncation."""
    if attachment < 0 or not np.isfinite(attachment):
        warnings.warn("attachment must be a non-negative finite number.")
        return float("nan")
    if cover < 0:
        warnings.warn("cover must be a non-negative number.")
        return float("nan")
    if alpha < 0 or not np.isfinite(alpha):
        warnings.warn("alpha must be a non-negative finite number.")
        return float("nan")
    if t <= 0 or not np.isfinite(t):
        warnings.warn("t must be a positive finite number.")
        return float("nan")

    finite_trunc = np.isfinite(truncation)
    if finite_trunc:
        if truncation <= t:
            warnings.warn("truncation must be larger than t.")
            return float("nan")
        if truncation <= attachment:
            return 0.0
        if attachment + cover > truncation:
            cover = truncation - attachment

    if np.isinf(cover):
        if alpha <= 1:
            return float("inf")
        if t <= attachment:
            ep = -((t / attachment) ** alpha) / (1 - alpha) * attachment
        else:
            ep = t - attachment - t / (1 - alpha)
        return ep

    # Finite cover — compute ignoring truncation first
    if t <= attachment:
        if alpha == 0:
            ep = cover
        elif alpha == 1:
            ep = t * (np.log(cover + attachment) - np.log(attachment))
        else:
            ep = (
                t
                / (1 - alpha)
                * (((cover + attachment) / t) ** (1 - alpha) - (attachment / t) ** (1 - alpha))
            )
    elif t >= attachment + cover:
        ep = cover
    else:
        ep = t - attachment
        if alpha == 0:
            ep = cover
        elif alpha == 1:
            ep += t * (np.log(cover + attachment) - np.log(t))
        else:
            ep += t / (1 - alpha) * (((cover + attachment) / t) ** (1 - alpha) - 1)

    if finite_trunc:
        if alpha < 1e-6:

            def _indef(x: float) -> float:
                return x - 1 / np.log(truncation / t) * (x * np.log(x / t) - x)

            if t <= attachment:
                ep = _indef(cover + attachment) - _indef(attachment)
            elif t >= cover + attachment:
                ep = cover
            else:
                ep = t - attachment + _indef(cover + attachment) - _indef(t)
        else:
            fq_at_trunc = (t / truncation) ** alpha
            ep = (ep - fq_at_trunc * cover) / (1 - fq_at_trunc)

    return ep


def pareto_layer_mean(
    cover: ArrayLike,
    attachment_point: ArrayLike,
    alpha: ArrayLike,
    t: ArrayLike | None = None,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Expected loss of Pareto(t, alpha) in reinsurance layer cover xs attachment_point.

    Parameters
    ----------
    cover:
        Cover of the reinsurance layer. Use np.inf for unlimited.
    attachment_point:
        Attachment point of the layer.
    alpha:
        Pareto shape parameter (non-negative).
    t:
        Threshold of the distribution. Defaults to attachment_point if None.
    truncation:
        Upper truncation point. None means no truncation.

    Returns
    -------
    np.ndarray
        Expected layer losses.

    Examples
    --------
    >>> pareto_layer_mean(8000, 2000, alpha=2)
    1600.0
    """
    _vfun = np.vectorize(_pareto_layer_mean_scalar)
    cover = np.asarray(cover, dtype=float)
    attachment_point = np.asarray(attachment_point, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    t_arr = np.asarray(t, dtype=float) if t is not None else attachment_point.copy()
    trunc_arr = (
        np.asarray(truncation, dtype=float)
        if truncation is not None
        else np.full_like(cover, np.inf)
    )
    return _vfun(cover, attachment_point, alpha, t_arr, trunc_arr)


def _pareto_layer_sm_simple_scalar(cover: float, attachment: float, alpha: float) -> float:
    """Second moment of Pareto layer when t == attachment (no threshold shift)."""
    if attachment <= 0 or not np.isfinite(attachment):
        return float("nan")
    if cover < 0:
        return float("nan")
    if cover == 0:
        return 0.0
    if alpha < 0 or not np.isfinite(alpha):
        return float("nan")

    if np.isinf(cover):
        if alpha <= 2:
            return float("inf")
        return 2 * attachment**2 * (1 / (alpha - 2) - 1 / (alpha - 1))

    if alpha == 0:
        return cover**2
    if alpha == 1:
        return 2 * attachment**2 * (cover / attachment - np.log(1 + cover / attachment))
    if alpha == 2:
        return 2 * attachment**2 * (-cover / (cover + attachment) + np.log(1 + cover / attachment))
    return (
        2
        * attachment**2
        * (
            ((1 + cover / attachment) ** (2 - alpha) - 1) / (2 - alpha)
            - ((1 + cover / attachment) ** (1 - alpha) - 1) / (1 - alpha)
        )
    )


def _pareto_layer_var_scalar(
    cover: float,
    attachment: float,
    alpha: float,
    t: float,
    truncation: float,
) -> float:
    """Scalar Pareto layer variance."""
    if attachment < 0 or not np.isfinite(attachment):
        warnings.warn("attachment must be a non-negative finite number.")
        return float("nan")
    if cover < 0:
        warnings.warn("cover must be a non-negative number.")
        return float("nan")
    if alpha < 0 or not np.isfinite(alpha):
        warnings.warn("alpha must be a non-negative finite number.")
        return float("nan")
    if t <= 0 or not np.isfinite(t):
        warnings.warn("t must be a positive finite number.")
        return float("nan")

    finite_trunc = np.isfinite(truncation)
    if finite_trunc:
        if truncation <= t:
            warnings.warn("truncation must be larger than t.")
            return float("nan")
        if truncation <= attachment:
            return 0.0
        if attachment + cover > truncation:
            cover = truncation - attachment

    exit_point = cover + attachment
    if t >= exit_point:
        return 0.0

    eff_attachment = max(attachment, t)
    eff_cover = exit_point - eff_attachment

    if np.isinf(eff_cover) and alpha <= 2:
        return float("inf")

    # SM at t == eff_attachment, ignoring any threshold shift
    sm = _pareto_layer_sm_simple_scalar(eff_cover, eff_attachment, alpha)

    if finite_trunc and alpha > 0:
        p = 1 - p_pareto(truncation, eff_attachment, alpha)
        sm = (sm - p * eff_cover**2) / (1 - p)

    # Account for t < attachment: multiply by survival probability.
    # Skip when finite_trunc and alpha == 0: p_pareto would produce 0/0 and
    # the special case below overwrites sm entirely anyway.
    if not (finite_trunc and alpha == 0):
        trunc_arg = truncation if finite_trunc else None
        p_surv = 1 - p_pareto(eff_attachment, t, alpha, truncation=trunc_arg)
        sm = float(p_surv) * sm

    if finite_trunc and alpha == 0:
        # Special case: use indefinite integral formula
        def _indef_var(x: float) -> float:
            if x <= t:
                return x**2 + 0.5 * t**2 / np.log(truncation / t)
            log_ratio = np.log(truncation / t)
            return x**2 - x**2 / log_ratio * np.log(x / t) + 0.5 * x**2 / log_ratio

        sm = (
            _indef_var(eff_cover + eff_attachment)
            - _indef_var(eff_attachment)
            - 2
            * eff_attachment
            * _pareto_layer_mean_scalar(eff_cover, eff_attachment, alpha, t, truncation)
        )

    # R modifies Cover/AttachmentPoint before computing the final mean for variance.
    # Use the effective (threshold-adjusted) values here to match R exactly.
    mean_eff = _pareto_layer_mean_scalar(eff_cover, eff_attachment, alpha, t, truncation)
    return sm - mean_eff**2


def pareto_layer_var(
    cover: ArrayLike,
    attachment_point: ArrayLike,
    alpha: ArrayLike,
    t: ArrayLike | None = None,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Variance of Pareto(t, alpha) in reinsurance layer cover xs attachment_point.

    Parameters
    ----------
    cover:
        Cover of the reinsurance layer.
    attachment_point:
        Attachment point.
    alpha:
        Pareto shape parameter.
    t:
        Threshold of the distribution.
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        Layer variances.
    """
    _vfun = np.vectorize(_pareto_layer_var_scalar)
    cover = np.asarray(cover, dtype=float)
    attachment_point = np.asarray(attachment_point, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    t_arr = np.asarray(t, dtype=float) if t is not None else attachment_point.copy()
    trunc_arr = (
        np.asarray(truncation, dtype=float)
        if truncation is not None
        else np.full_like(cover, np.inf)
    )
    return _vfun(cover, attachment_point, alpha, t_arr, trunc_arr)


def pareto_layer_sm(
    cover: ArrayLike,
    attachment_point: ArrayLike,
    alpha: ArrayLike,
    t: ArrayLike | None = None,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Second moment of Pareto(t, alpha) in reinsurance layer cover xs attachment_point.

    Parameters
    ----------
    cover:
        Cover of the reinsurance layer.
    attachment_point:
        Attachment point.
    alpha:
        Pareto shape parameter.
    t:
        Threshold of the distribution.
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        Layer second moments.
    """
    var = pareto_layer_var(cover, attachment_point, alpha, t, truncation)
    mean = pareto_layer_mean(cover, attachment_point, alpha, t, truncation)
    # handle inf
    result = np.where(np.isinf(var), np.inf, var + mean**2)
    return result
