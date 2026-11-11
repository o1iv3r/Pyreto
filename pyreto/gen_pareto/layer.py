"""Expected value, second moment and variance of generalized Pareto layer losses."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from pyreto._validation import (
    is_nonnegative_finite_number,
    is_nonnegative_number,
    is_positive_finite_number,
    is_positive_number,
)
from pyreto.gen_pareto.distribution import _p_gen_pareto_s, _psi


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
