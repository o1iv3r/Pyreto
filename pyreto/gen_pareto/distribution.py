"""CDF, PDF, quantile and random generation of the generalized Pareto distribution."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from pyreto._validation import (
    is_positive_finite_vector,
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
