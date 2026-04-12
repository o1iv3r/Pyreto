"""Generalized Pareto distribution functions for reinsurance pricing.

Mirrors the GenPareto section of the R Pareto package (Functions.R, lines 3381-4261).
All public functions accept scalar or array inputs via np.vectorize dispatch.

The generalized Pareto distribution is parameterised by (t, alpha_ini, alpha_tail)
and has survival function:

    S(x) = (1 + alpha_ini/alpha_tail * (x/t - 1))^(-alpha_tail)   for x > t

When alpha_ini == alpha_tail the distribution reduces to the standard Pareto(t, alpha).

R -> Python name mapping (see docs/function_mapping.md):
  pGenPareto                    -> p_gen_pareto
  dGenPareto                    -> d_gen_pareto
  qGenPareto                    -> q_gen_pareto
  rGenPareto                    -> r_gen_pareto
  GenPareto_Layer_Mean          -> gen_pareto_layer_mean
  GenPareto_Layer_SM            -> gen_pareto_layer_sm
  GenPareto_Layer_Var           -> gen_pareto_layer_var
  GenPareto_ML_Estimator_Alpha  -> gen_pareto_ml_estimator_alpha
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from pyreto._validation import (
    is_nonnegative_finite_number,
    is_nonnegative_number,
    is_positive_finite_number,
    is_positive_finite_vector,
    is_positive_number,
    is_positive_vector,
    valid_parameters_gen_pareto,
)

# ---------------------------------------------------------------------------
# Private scalar helpers
# ---------------------------------------------------------------------------


def _psi(x: float, t: float, alpha_ini: float, alpha_tail: float) -> float:
    """Survival function value S(x) = (1 + alpha_ini/alpha_tail*(x/t-1))^(-alpha_tail)."""
    return (1.0 + alpha_ini / alpha_tail * (x / t - 1.0)) ** (-alpha_tail)


def _p_gen_pareto_s(
    x: float,
    t: float,
    alpha_ini: float,
    alpha_tail: float,
    truncation: float | None,
) -> float:
    """Scalar CDF of the generalized Pareto distribution."""
    if not valid_parameters_gen_pareto(t, alpha_ini, alpha_tail, truncation, comment=True):
        return np.nan
    x = float(x)
    if np.isnan(x):
        warnings.warn("x must be a number.")
        return np.nan
    if x <= t:
        return 0.0
    if truncation is None:
        return 1.0 - _psi(x, t, alpha_ini, alpha_tail)
    if x >= truncation:
        return 1.0
    num = 1.0 - _psi(x, t, alpha_ini, alpha_tail)
    denom = 1.0 - _psi(truncation, t, alpha_ini, alpha_tail)
    return num / denom


def _d_gen_pareto_s(
    x: float,
    t: float,
    alpha_ini: float,
    alpha_tail: float,
    truncation: float | None,
) -> float:
    """Scalar PDF of the generalized Pareto distribution."""
    if not valid_parameters_gen_pareto(t, alpha_ini, alpha_tail, truncation, comment=True):
        return np.nan
    x = float(x)
    if np.isnan(x):
        warnings.warn("x must be a number.")
        return np.nan
    if x < t:
        return 0.0
    if truncation is not None and x >= truncation:
        return 0.0
    u = 1.0 + alpha_ini / alpha_tail * (x / t - 1.0)
    density = alpha_ini / t * u ** (-alpha_tail - 1.0)
    if truncation is not None:
        density /= _p_gen_pareto_s(truncation, t, alpha_ini, alpha_tail, None)
    return density


def _q_gen_pareto_s(
    y: float,
    t: float,
    alpha_ini: float,
    alpha_tail: float,
    truncation: float | None,
) -> float:
    """Scalar quantile function of the generalized Pareto distribution."""
    if not valid_parameters_gen_pareto(t, alpha_ini, alpha_tail, truncation, comment=True):
        return np.nan
    y = float(y)
    if np.isnan(y) or not np.isfinite(y) or y < 0.0 or y > 1.0:
        warnings.warn("y must be a number in [0, 1].")
        return np.nan
    if y == 1.0:
        return float(truncation) if truncation is not None else np.inf
    if truncation is not None and not np.isinf(truncation):
        scale = _p_gen_pareto_s(truncation, t, alpha_ini, alpha_tail, None)
        y = y * scale
    return t * (1.0 + alpha_tail / alpha_ini * ((1.0 - y) ** (-1.0 / alpha_tail) - 1.0))


def _gen_pareto_layer_mean_s(
    cover: float,
    att: float,
    t: float,
    alpha_ini: float,
    alpha_tail: float,
    truncation: float | None,
) -> float:
    """Scalar layer mean E[min(max(X-att,0), cover)] for X ~ GenPareto(t, alpha_ini, alpha_tail)."""
    if not is_nonnegative_finite_number(att):
        warnings.warn("AttachmentPoint must be a non-negative number.")
        return np.nan
    if not is_nonnegative_number(cover):
        warnings.warn("Cover must be a non-negative number ('Inf' allowed).")
        return np.nan
    if not is_positive_finite_number(alpha_ini):
        warnings.warn("alpha_ini must be a positive number.")
        return np.nan
    if not is_positive_finite_number(alpha_tail):
        warnings.warn("alpha_tail must be a positive number.")
        return np.nan
    if not is_positive_finite_number(t):
        warnings.warn("t must be a positive number.")
        return np.nan

    if truncation is not None:
        if not is_positive_number(truncation):
            warnings.warn("truncation must be NULL or a positive number ('Inf' allowed).")
            return np.nan
        if not np.isinf(truncation) and truncation <= t:
            warnings.warn("truncation must be larger than t.")
            return np.nan
        if not np.isinf(truncation) and truncation <= att:
            return 0.0
        if not np.isinf(truncation) and att + cover > truncation:
            cover = truncation - att

    if np.isinf(cover):
        if alpha_tail <= 1.0:
            return np.inf
        if t <= att:
            u_att = 1.0 + alpha_ini / alpha_tail * (att / t - 1.0)
            ep = -t * alpha_tail / (alpha_ini * (1.0 - alpha_tail)) * u_att ** (1.0 - alpha_tail)
        else:
            ep = (t - att) - t * alpha_tail / (alpha_ini * (1.0 - alpha_tail))
        return ep

    # Finite cover
    upper = att + cover

    if t <= att:
        if alpha_tail == 1.0:
            ep = (
                t
                * alpha_tail
                / alpha_ini
                * (
                    np.log(1.0 + alpha_ini / alpha_tail * (upper / t - 1.0))
                    - np.log(1.0 + alpha_ini / alpha_tail * (att / t - 1.0))
                )
            )
        else:
            ep = (
                t
                * alpha_tail
                / (alpha_ini * (1.0 - alpha_tail))
                * (
                    (1.0 + alpha_ini / alpha_tail * (upper / t - 1.0)) ** (1.0 - alpha_tail)
                    - (1.0 + alpha_ini / alpha_tail * (att / t - 1.0)) ** (1.0 - alpha_tail)
                )
            )
    elif t >= upper:
        ep = cover
    else:
        ep = t - att
        if alpha_tail == 1.0:
            u_upper = 1.0 + alpha_ini / alpha_tail * (upper / t - 1.0)
            ep += t * alpha_tail / alpha_ini * np.log(u_upper)
        else:
            ep += (
                t
                * alpha_tail
                / (alpha_ini * (1.0 - alpha_tail))
                * ((1.0 + alpha_ini / alpha_tail * (upper / t - 1.0)) ** (1.0 - alpha_tail) - 1.0)
            )

    if truncation is not None and is_positive_finite_number(truncation):
        psi_trunc = _psi(truncation, t, alpha_ini, alpha_tail)
        ep = (ep - psi_trunc * cover) / (1.0 - psi_trunc)

    return ep


def _gen_pareto_layer_sm_s(
    cover: float,
    att: float,
    t: float,
    alpha_ini: float,
    alpha_tail: float,
    truncation: float | None,
) -> float:
    """Scalar second moment E[(min(max(X-att,0),cover))^2] for X ~ GenPareto."""
    if not is_nonnegative_finite_number(att):
        warnings.warn("AttachmentPoint must be a non-negative number.")
        return np.nan
    if not is_nonnegative_number(cover):
        warnings.warn("Cover must be a non-negative number ('Inf' allowed).")
        return np.nan
    if not is_positive_finite_number(alpha_ini):
        warnings.warn("alpha_ini must be a positive number.")
        return np.nan
    if not is_positive_finite_number(alpha_tail):
        warnings.warn("alpha_tail must be a positive number.")
        return np.nan
    if not is_positive_finite_number(t):
        warnings.warn("t must be a positive number.")
        return np.nan

    if truncation is not None:
        if not is_positive_number(truncation):
            warnings.warn("truncation must be NULL or a positive number ('Inf' allowed).")
            return np.nan
        if not np.isinf(truncation) and truncation <= t:
            warnings.warn("truncation must be larger than t.")
            return np.nan
        if not np.isinf(truncation) and truncation <= att:
            return 0.0
        if not np.isinf(truncation) and att + cover > truncation:
            cover = truncation - att

    if np.isinf(cover) and alpha_tail <= 2.0:
        return np.inf

    # Helper antiderivatives (R's G and H)
    def _g(x: float) -> float:
        u = 1.0 + alpha_ini / alpha_tail * (x / t - 1.0)
        if alpha_tail != 1.0:
            c = t * alpha_tail / (alpha_ini * (1.0 - alpha_tail))
            return c * u ** (1.0 - alpha_tail) if x > t else x - t + c
        else:
            return t / alpha_ini * np.log(u) if x > t else x - t

    def _h(x: float) -> float:
        u = 1.0 + alpha_ini / alpha_tail * (x / t - 1.0)
        if alpha_tail == 1.0:
            if x > t:
                return t * x / alpha_ini * np.log(u) - t**2 / alpha_ini**2 * (u * np.log(u) - u)
            return x**2 / 2.0 - t**2 / 2.0 + t**2 / alpha_ini**2
        elif alpha_tail == 2.0:
            c1 = t * alpha_tail / (alpha_ini * (1.0 - alpha_tail))
            c2 = t**2 * alpha_tail**2 / (alpha_ini**2 * (1.0 - alpha_tail))
            if x > t:
                return c1 * x * u ** (1.0 - alpha_tail) - c2 * np.log(u)
            return x**2 / 2.0 - t**2 / 2.0 + c1 * t
        else:
            c1 = t * alpha_tail / (alpha_ini * (1.0 - alpha_tail))
            c2 = t**2 * alpha_tail**2 / (alpha_ini**2 * (1.0 - alpha_tail))
            c3 = c2 / (2.0 - alpha_tail)
            if np.isinf(x):
                return 0.0
            if x > t:
                return c1 * x * u ** (1.0 - alpha_tail) - c3 * u ** (2.0 - alpha_tail)
            return x**2 / 2.0 - t**2 / 2.0 + c1 * t - c3

    def phi_1(x: float) -> float:
        if np.isinf(x):
            return 0.0
        return x * (1.0 - _p_gen_pareto_s(x, t, alpha_ini, alpha_tail, None))

    def phi_2(x: float) -> float:
        if np.isinf(x):
            return 0.0
        return x**2 * (1.0 - _p_gen_pareto_s(x, t, alpha_ini, alpha_tail, None))

    upper = att + cover
    f_upper = _p_gen_pareto_s(upper, t, alpha_ini, alpha_tail, None)
    f_att = _p_gen_pareto_s(att, t, alpha_ini, alpha_tail, None)

    result = att**2 * (f_upper - f_att)
    result -= 2.0 * att * (_g(upper) - _g(att) - phi_1(upper) + phi_1(att))
    result += 2.0 * (_h(upper) - _h(att)) - phi_2(upper) + phi_2(att)
    if not np.isinf(cover):
        result += cover**2 * (1.0 - f_upper)

    if truncation is not None:
        psi_trunc = _psi(truncation, t, alpha_ini, alpha_tail)
        result = (result - psi_trunc * cover**2) / (1.0 - psi_trunc)

    return result


def _gen_pareto_layer_var_s(
    cover: float,
    att: float,
    t: float,
    alpha_ini: float,
    alpha_tail: float,
    truncation: float | None,
) -> float:
    """Scalar layer variance."""
    if not is_nonnegative_finite_number(att):
        warnings.warn("AttachmentPoint must be a non-negative number.")
        return np.nan
    if not is_nonnegative_number(cover):
        warnings.warn("Cover must be a non-negative number ('Inf' allowed).")
        return np.nan
    if not is_positive_finite_number(alpha_ini):
        warnings.warn("alpha_ini must be a positive number.")
        return np.nan
    if not is_positive_finite_number(alpha_tail):
        warnings.warn("alpha_tail must be a positive number.")
        return np.nan
    if not is_positive_finite_number(t):
        warnings.warn("t must be a positive number.")
        return np.nan

    if truncation is not None:
        if not is_positive_number(truncation):
            warnings.warn("truncation must be NULL or a positive number ('Inf' allowed).")
            return np.nan
        if not np.isinf(truncation) and truncation <= t:
            warnings.warn("truncation must be larger than t.")
            return np.nan
        if not np.isinf(truncation) and truncation <= att:
            return 0.0
        if not np.isinf(truncation) and att + cover > truncation:
            cover = truncation - att

    if np.isinf(cover) and alpha_tail <= 2.0:
        return np.inf

    sm = _gen_pareto_layer_sm_s(cover, att, t, alpha_ini, alpha_tail, truncation)
    mean = _gen_pareto_layer_mean_s(cover, att, t, alpha_ini, alpha_tail, truncation)
    return sm - mean**2


# ---------------------------------------------------------------------------
# Public vectorized functions
# ---------------------------------------------------------------------------


def p_gen_pareto(
    x: ArrayLike,
    t: ArrayLike,
    alpha_ini: ArrayLike,
    alpha_tail: ArrayLike,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """CDF of the generalized Pareto(t, alpha_ini, alpha_tail) distribution.

    Parameters
    ----------
    x:
        Values at which to evaluate the CDF.
    t:
        Threshold parameter. Must be positive.
    alpha_ini:
        Initial Pareto alpha (at threshold t). Must be positive.
    alpha_tail:
        Tail Pareto alpha. Must be positive.
    truncation:
        Upper truncation point. If provided, the CDF is renormalised.

    Returns
    -------
    np.ndarray
        CDF values in [0, 1].
    """
    if truncation is None:
        vf = np.vectorize(_p_gen_pareto_s, excluded=["truncation"], otypes=[float])
        return vf(x, t, alpha_ini, alpha_tail, truncation=None)
    vf = np.vectorize(_p_gen_pareto_s, otypes=[float])
    return vf(x, t, alpha_ini, alpha_tail, truncation)


def d_gen_pareto(
    x: ArrayLike,
    t: ArrayLike,
    alpha_ini: ArrayLike,
    alpha_tail: ArrayLike,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """PDF of the generalized Pareto(t, alpha_ini, alpha_tail) distribution.

    Parameters
    ----------
    x:
        Values at which to evaluate the PDF.
    t:
        Threshold parameter.
    alpha_ini:
        Initial Pareto alpha.
    alpha_tail:
        Tail Pareto alpha.
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        PDF values (zero outside support).
    """
    if truncation is None:
        vf = np.vectorize(_d_gen_pareto_s, excluded=["truncation"], otypes=[float])
        return vf(x, t, alpha_ini, alpha_tail, truncation=None)
    vf = np.vectorize(_d_gen_pareto_s, otypes=[float])
    return vf(x, t, alpha_ini, alpha_tail, truncation)


def q_gen_pareto(
    p: ArrayLike,
    t: ArrayLike,
    alpha_ini: ArrayLike,
    alpha_tail: ArrayLike,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Quantile function of the generalized Pareto(t, alpha_ini, alpha_tail) distribution.

    Parameters
    ----------
    p:
        Probability values in [0, 1].
    t:
        Threshold parameter.
    alpha_ini:
        Initial Pareto alpha.
    alpha_tail:
        Tail Pareto alpha.
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        Quantile values.
    """
    if truncation is None:
        vf = np.vectorize(_q_gen_pareto_s, excluded=["truncation"], otypes=[float])
        return vf(p, t, alpha_ini, alpha_tail, truncation=None)
    vf = np.vectorize(_q_gen_pareto_s, otypes=[float])
    return vf(p, t, alpha_ini, alpha_tail, truncation)


def r_gen_pareto(
    n: int,
    t: ArrayLike,
    alpha_ini: ArrayLike,
    alpha_tail: ArrayLike,
    truncation: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random samples from the generalized Pareto distribution.

    Parameters
    ----------
    n:
        Number of samples.
    t:
        Threshold parameter(s). Recycled to length n if shorter.
    alpha_ini:
        Initial Pareto alpha(s). Recycled to length n.
    alpha_tail:
        Tail Pareto alpha(s). Recycled to length n.
    truncation:
        Upper truncation point(s), or None. Recycled to length n.
    rng:
        NumPy random Generator. If None, uses ``np.random.default_rng()``.

    Returns
    -------
    np.ndarray
        Array of n samples.
    """
    n = int(np.ceil(n))
    if rng is None:
        rng = np.random.default_rng()

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    a_ini = np.atleast_1d(np.asarray(alpha_ini, dtype=float))
    a_tail = np.atleast_1d(np.asarray(alpha_tail, dtype=float))

    if not is_positive_finite_vector(t_arr):
        warnings.warn("t must be a positive vector.")
        return np.full(n, np.nan)
    if not is_positive_finite_vector(a_ini):
        warnings.warn("alpha_ini must be a positive vector.")
        return np.full(n, np.nan)
    if not is_positive_finite_vector(a_tail):
        warnings.warn("alpha_tail must be a positive vector.")
        return np.full(n, np.nan)

    # Broadcast / tile to length n
    def _tile(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 1:
            return np.repeat(arr, n)
        if n % len(arr) != 0:
            warnings.warn("n is not a multiple of parameter vector length.")
            return np.tile(arr, int(np.ceil(n / len(arr))))[:n]
        return np.tile(arr, n // len(arr))

    t_arr = _tile(t_arr)
    a_ini = _tile(a_ini)
    a_tail = _tile(a_tail)

    u_lo = np.zeros(n)
    u_hi = np.ones(n)

    if truncation is not None:
        trunc_arr = np.atleast_1d(np.asarray(truncation, dtype=float))
        if not is_positive_vector(trunc_arr):
            warnings.warn("truncation must be NULL or a positive vector.")
            return np.full(n, np.nan)
        trunc_arr = _tile(trunc_arr)
        if np.any(trunc_arr <= t_arr):
            warnings.warn("truncation must be > t.")
            return np.full(n, np.nan)
        u_hi = 1.0 - (1.0 + a_ini / a_tail * (trunc_arr / t_arr - 1.0)) ** (-a_tail)

    u = rng.uniform(u_lo, u_hi, n)
    return t_arr * (1.0 + a_tail / a_ini * ((1.0 - u) ** (-1.0 / a_tail) - 1.0))


def gen_pareto_layer_mean(
    cover: ArrayLike,
    att: ArrayLike,
    t: ArrayLike,
    alpha_ini: ArrayLike,
    alpha_tail: ArrayLike,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Expected layer loss for GenPareto(t, alpha_ini, alpha_tail) in layer cover xs att.

    Parameters
    ----------
    cover:
        Layer cover. Use ``np.inf`` for unlimited layers.
    att:
        Attachment point of the layer.
    t:
        Threshold of the generalized Pareto distribution.
    alpha_ini:
        Initial Pareto alpha (at threshold t).
    alpha_tail:
        Tail Pareto alpha.
    truncation:
        Upper truncation point (``np.inf`` or ``None`` means no truncation).

    Returns
    -------
    np.ndarray
        Expected layer loss values.
    """
    if truncation is None:
        vf = np.vectorize(_gen_pareto_layer_mean_s, excluded=["truncation"], otypes=[float])
        return vf(cover, att, t, alpha_ini, alpha_tail, truncation=None)
    vf = np.vectorize(_gen_pareto_layer_mean_s, otypes=[float])
    return vf(cover, att, t, alpha_ini, alpha_tail, truncation)


def gen_pareto_layer_sm(
    cover: ArrayLike,
    att: ArrayLike,
    t: ArrayLike,
    alpha_ini: ArrayLike,
    alpha_tail: ArrayLike,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Second moment of layer loss for GenPareto(t, alpha_ini, alpha_tail).

    Parameters
    ----------
    cover:
        Layer cover. Use ``np.inf`` for unlimited layers.
    att:
        Attachment point of the layer.
    t:
        Threshold of the generalized Pareto distribution.
    alpha_ini:
        Initial Pareto alpha.
    alpha_tail:
        Tail Pareto alpha.
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        Second moment of layer loss values.
    """
    if truncation is None:
        vf = np.vectorize(_gen_pareto_layer_sm_s, excluded=["truncation"], otypes=[float])
        return vf(cover, att, t, alpha_ini, alpha_tail, truncation=None)
    vf = np.vectorize(_gen_pareto_layer_sm_s, otypes=[float])
    return vf(cover, att, t, alpha_ini, alpha_tail, truncation)


def gen_pareto_layer_var(
    cover: ArrayLike,
    att: ArrayLike,
    t: ArrayLike,
    alpha_ini: ArrayLike,
    alpha_tail: ArrayLike,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Variance of layer loss for GenPareto(t, alpha_ini, alpha_tail).

    Parameters
    ----------
    cover:
        Layer cover. Use ``np.inf`` for unlimited layers.
    att:
        Attachment point of the layer.
    t:
        Threshold of the generalized Pareto distribution.
    alpha_ini:
        Initial Pareto alpha.
    alpha_tail:
        Tail Pareto alpha.
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        Variance of layer loss values.
    """
    if truncation is None:
        vf = np.vectorize(_gen_pareto_layer_var_s, excluded=["truncation"], otypes=[float])
        return vf(cover, att, t, alpha_ini, alpha_tail, truncation=None)
    vf = np.vectorize(_gen_pareto_layer_var_s, otypes=[float])
    return vf(cover, att, t, alpha_ini, alpha_tail, truncation)


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

    use_rt = np.any(rt > t)
    has_censored = np.any(censored)

    def _u(x: np.ndarray) -> np.ndarray:
        return 1.0 + alpha[0] / alpha[1] * (x / t - 1.0)

    # Closure reference - will be set inside the optimizer
    alpha = np.ones(2)

    if not use_rt and not has_censored:
        if np.isinf(trunc):

            def neg_ll(a: np.ndarray) -> float:
                u_l = 1.0 + a[0] / a[1] * (losses / t - 1.0)
                return -float(np.sum(w * (np.log(a[0]) + (-a[1] - 1.0) * np.log(u_l))))

        else:

            def neg_ll(a: np.ndarray) -> float:
                u_l = 1.0 + a[0] / a[1] * (losses / t - 1.0)
                u_tr = 1.0 + a[0] / a[1] * (trunc / t - 1.0)
                ll_i = np.log(a[0]) + (-a[1] - 1.0) * np.log(u_l) - np.log(1.0 - u_tr ** (-a[1]))
                return -float(np.sum(w * ll_i))

    elif not has_censored:
        if np.isinf(trunc):

            def neg_ll(a: np.ndarray) -> float:
                u_l = 1.0 + a[0] / a[1] * (losses / t - 1.0)
                u_rt = 1.0 + a[0] / a[1] * (rt / t - 1.0)
                ll_i = np.log(a[0]) + (-a[1] - 1.0) * np.log(u_l) - np.log(u_rt ** (-a[1]))
                return -float(np.sum(w * ll_i))

        else:

            def neg_ll(a: np.ndarray) -> float:
                u_l = 1.0 + a[0] / a[1] * (losses / t - 1.0)
                u_rt = 1.0 + a[0] / a[1] * (rt / t - 1.0)
                u_tr = 1.0 + a[0] / a[1] * (trunc / t - 1.0)
                denom = u_rt ** (-a[1]) - u_tr ** (-a[1])
                ll_i = np.log(a[0]) + (-a[1] - 1.0) * np.log(u_l) - np.log(denom)
                return -float(np.sum(w * ll_i))

    else:

        def neg_ll(a: np.ndarray) -> float:
            u_l = 1.0 + a[0] / a[1] * (losses / t - 1.0)
            u_rt = 1.0 + a[0] / a[1] * (rt / t - 1.0)
            u_tr = 1.0 + a[0] / a[1] * (trunc / t - 1.0)
            denom = u_rt ** (-a[1]) - u_tr ** (-a[1])
            ll_c = np.log(u_l ** (-a[1]) - u_tr ** (-a[1])) - np.log(denom)
            ll_u = np.log(a[0]) + (-a[1] - 1.0) * np.log(u_l) - np.log(denom)
            return -float(np.sum(w * np.where(censored, ll_c, ll_u)))

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
