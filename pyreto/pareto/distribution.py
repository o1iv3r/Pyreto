"""CDF, PDF, quantile and random generation of the Pareto distribution."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# CDF / PDF / quantile / random
# ---------------------------------------------------------------------------


def p_pareto(
    x: ArrayLike,
    t: ArrayLike,
    alpha: float,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """CDF of the Pareto(t, alpha) distribution.

    Parameters
    ----------
    x:
        Values at which to evaluate the CDF.
    t:
        Threshold (scale) parameter. Must be positive.
    alpha:
        Shape parameter. Must be positive.
    truncation:
        Upper truncation point. If provided, the distribution is truncated at
        this value and the CDF is renormalised.

    Returns
    -------
    np.ndarray
        CDF values in [0, 1].

    Examples
    --------
    >>> p_pareto(np.array([2000.0]), t=1000.0, alpha=2.0)
    array([0.75])
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    alpha = float(alpha)
    result = np.where(x <= t, 0.0, 1.0 - (t / x) ** alpha)

    if truncation is not None:
        trunc = np.asarray(truncation, dtype=float)
        p_trunc = 1.0 - (t / trunc) ** alpha
        result = np.where(x >= trunc, 1.0, result / p_trunc)

    return result


def d_pareto(
    x: ArrayLike,
    t: ArrayLike,
    alpha: float,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """PDF of the Pareto(t, alpha) distribution.

    Parameters
    ----------
    x:
        Values at which to evaluate the PDF.
    t:
        Threshold parameter.
    alpha:
        Shape parameter.
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        PDF values (zero outside support).

    Examples
    --------
    >>> d_pareto(np.array([2000.0]), t=1000.0, alpha=2.0)
    array([0.00025])
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    alpha = float(alpha)
    pdf = np.where(x < t, 0.0, alpha * t**alpha / x ** (alpha + 1))

    if truncation is not None:
        trunc = np.asarray(truncation, dtype=float)
        p_trunc = 1.0 - (t / trunc) ** alpha
        pdf = np.where(x > trunc, 0.0, pdf / p_trunc)

    return pdf


def q_pareto(
    p: ArrayLike,
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Quantile function of the Pareto(t, alpha) distribution.

    Parameters
    ----------
    p:
        Probabilities in [0, 1].
    t:
        Threshold parameter.
    alpha:
        Shape parameter.
    truncation:
        Upper truncation point.

    Returns
    -------
    np.ndarray
        Quantiles.

    Examples
    --------
    >>> q_pareto(np.array([0.75]), t=1000.0, alpha=2.0)
    array([2000.])
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    if truncation is None:
        return t / (1.0 - p) ** (1.0 / alpha)

    trunc = np.asarray(truncation, dtype=float)
    p_trunc = 1.0 - (t / trunc) ** alpha
    p_adj = p * p_trunc
    return np.minimum(trunc, t / (1.0 - p_adj) ** (1.0 / alpha))


def r_pareto(
    n: int,
    t: float,
    alpha: float,
    truncation: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw random samples from the Pareto(t, alpha) distribution.

    Parameters
    ----------
    n:
        Number of samples.
    t:
        Threshold parameter.
    alpha:
        Shape parameter.
    truncation:
        Upper truncation point.
    rng:
        Random number generator (for reproducibility). If None, uses default_rng().

    Returns
    -------
    np.ndarray
        Array of n samples, all >= t.

    Examples
    --------
    >>> r_pareto(5, t=1000.0, alpha=2.0, rng=np.random.default_rng(42))
    array([...])
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(0, 1, n)
    return q_pareto(u, t=t, alpha=alpha, truncation=truncation)
