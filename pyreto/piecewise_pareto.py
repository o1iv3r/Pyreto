"""PiecewisePareto distribution functions for reinsurance pricing.

Mirrors the PiecewisePareto section of the R Pareto package (Functions.R).
All layer-moment functions accept scalar or array inputs for cover/attachment.

R -> Python name mapping (see docs/function_mapping.md):
  pPiecewisePareto                   -> p_piecewise_pareto
  dPiecewisePareto                   -> d_piecewise_pareto
  qPiecewisePareto                   -> q_piecewise_pareto
  rPiecewisePareto                   -> r_piecewise_pareto
  PiecewisePareto_Layer_Mean         -> piecewise_pareto_layer_mean
  PiecewisePareto_Layer_SM           -> piecewise_pareto_layer_sm
  PiecewisePareto_Layer_Var          -> piecewise_pareto_layer_var
  PiecewisePareto_ML_Estimator_Alpha -> piecewise_pareto_ml_estimator_alpha
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from pyreto._validation import valid_parameters_piecewise_pareto
from pyreto.pareto import (
    _pareto_layer_mean_scalar,
    _pareto_layer_var_scalar,
    p_pareto,
    pareto_ml_estimator_alpha,
    r_pareto,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _excess_probs(t_arr: np.ndarray, alpha_arr: np.ndarray) -> np.ndarray:
    """P(X > t[i]) with P(X > t[0]) = 1 (unnormalised)."""
    n = len(t_arr)
    ep = np.ones(n)
    if n > 1:
        factors = t_arr[1:] / t_arr[:-1]
        ep[1:] = np.cumprod((1.0 / factors) ** alpha_arr[:-1])
    return ep


def _pareto_sm_scalar(cover: float, att: float, alpha: float, t: float, trunc_inf: float) -> float:
    """Pareto_Layer_SM(cover, att, alpha, t, trunc_inf). trunc_inf=np.inf = no truncation."""
    var = _pareto_layer_var_scalar(cover, att, alpha, t, trunc_inf)
    mean = _pareto_layer_mean_scalar(cover, att, alpha, t, trunc_inf)
    if np.isinf(var):
        return np.inf
    return var + mean * mean


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


# ---------------------------------------------------------------------------
# Layer moments — scalar implementations
# ---------------------------------------------------------------------------


def _piecewise_pareto_layer_mean_scalar(
    cover: float,
    attachment: float,
    t_arr: np.ndarray,
    alpha_arr: np.ndarray,
    trunc: float | None,
    truncation_type: str,
) -> float:
    """Scalar layer mean (mirrors PiecewisePareto_Layer_Mean_s in R)."""
    n = len(t_arr)

    if n == 1:
        trunc_inf = np.inf if trunc is None else float(trunc)
        return _pareto_layer_mean_scalar(cover, attachment, alpha_arr[0], t_arr[0], trunc_inf)

    trunc_val = trunc  # tracks truncation (may become None if inf)

    if trunc_val is not None:
        if trunc_val <= attachment:
            return 0.0
        cover = min(cover, trunc_val - attachment)
        if np.isinf(trunc_val):
            trunc_val = None

    cover_orig = cover  # post-truncation-cap cover (matches R: Cover_orig <- Cover after capping)

    if cover == 0.0:
        return 0.0
    if np.isinf(cover) and alpha_arr[-1] <= 1.0:
        return np.inf

    excess_prob = _excess_probs(t_arr, alpha_arr)

    k1 = int(np.sum(t_arr <= attachment))
    k2 = n if np.isinf(attachment + cover) else int(np.sum(t_arr < attachment + cover))

    result = 0.0
    att = attachment  # effective attachment (updated if k1 == 0)

    if k1 == 0 and k2 == 0:
        return float(cover)
    if k1 == 0:
        result = float(t_arr[0]) - attachment
        att = t_arr[0]
        if np.isfinite(cover):
            cover = cover - result
        k1 = 1

    for i in range(k1 - 1, k2):
        att_i = max(t_arr[i], att)
        next_t = t_arr[i + 1] if i + 1 < n else np.inf

        if np.isfinite(cover):
            exit_i = min(next_t, att + cover)
        else:
            exit_i = next_t if i < n - 1 else np.inf

        piece_cover = exit_i - att_i

        if np.isfinite(cover):
            is_lp_last = trunc_val is not None and truncation_type == "lp" and i == n - 1
            t_arg: float = float(trunc_val) if trunc_val is not None and is_lp_last else np.inf
            contrib = _pareto_layer_mean_scalar(piece_cover, att_i, alpha_arr[i], t_arr[i], t_arg)
        else:
            if i < n - 1:
                contrib = _pareto_layer_mean_scalar(
                    piece_cover, att_i, alpha_arr[i], t_arr[i], np.inf
                )
            else:
                contrib = _pareto_layer_mean_scalar(np.inf, att_i, alpha_arr[-1], t_arr[-1], np.inf)

        result += contrib * excess_prob[i]

    if trunc_val is not None and truncation_type == "wd":
        p = (1.0 - float(p_pareto(trunc_val, t_arr[-1], alpha_arr[-1]))) * excess_prob[-1]
        result = (result - p * cover_orig) / (1.0 - p)

    return result


def _piecewise_pareto_layer_sm_scalar(
    cover: float,
    attachment: float,
    t_arr: np.ndarray,
    alpha_arr: np.ndarray,
    trunc: float | None,
    truncation_type: str,
) -> float:
    """Scalar layer second moment (mirrors PiecewisePareto_Layer_SM_s in R)."""
    n = len(t_arr)

    if n == 1:
        trunc_inf = np.inf if trunc is None else float(trunc)
        return _pareto_sm_scalar(cover, attachment, alpha_arr[0], t_arr[0], trunc_inf)

    att_orig = attachment
    trunc_val = trunc

    if trunc_val is not None:
        if trunc_val <= attachment:
            return 0.0
        cover = min(cover, trunc_val - attachment)
        if np.isinf(trunc_val):
            trunc_val = None

    cover_orig = cover  # post-truncation-cap cover (matches R: Cover_orig <- Cover after capping)

    if cover == 0.0:
        return 0.0
    if np.isinf(cover) and alpha_arr[-1] <= 2.0:
        return np.inf

    excess_prob = _excess_probs(t_arr, alpha_arr)
    prob = np.empty(n)
    prob[:-1] = -np.diff(excess_prob)
    prob[-1] = excess_prob[-1]

    k1 = int(np.sum(t_arr <= attachment))
    k2 = n if np.isinf(attachment + cover) else int(np.sum(t_arr < attachment + cover))

    att = attachment  # effective attachment

    if k1 == 0 and k2 == 0:
        return float(cover) ** 2
    if k1 == 0:
        att = t_arr[0]
        if np.isfinite(cover):
            cover = cover - (att - att_orig)
        k1 = 1

    result = 0.0

    for i in range(k1 - 1, k2):
        att_i = max(t_arr[i], att)
        next_t = t_arr[i + 1] if i + 1 < n else np.inf

        # exit for piece i
        if np.isfinite(cover):
            exit_i = min(next_t, att + cover)
        else:
            exit_i = next_t if i < n - 1 else np.inf

        piece_cover = exit_i - att_i
        shift = att_i - att_orig
        factor = (t_arr[i] / att_i) ** alpha_arr[i] if att_i > 0 else 1.0

        is_lp_last = trunc_val is not None and truncation_type == "lp" and i == n - 1
        is_k2_piece = i == k2 - 1

        if is_lp_last:
            w = prob[i]
            assert trunc_val is not None  # is_lp_last guarantees this
            trunc_float = float(trunc_val)
            sm = _pareto_sm_scalar(piece_cover, att_i, alpha_arr[i], t_arr[i], trunc_float)
            mean = _pareto_layer_mean_scalar(
                piece_cover, att_i, alpha_arr[i], t_arr[i], trunc_float
            )
            result += sm * w
            result += shift**2 * w * factor
            result += 2.0 * shift * mean * w
        elif is_k2_piece:
            w = excess_prob[i]
            sm = _pareto_sm_scalar(piece_cover, att_i, alpha_arr[i], t_arr[i], np.inf)
            mean = _pareto_layer_mean_scalar(piece_cover, att_i, alpha_arr[i], t_arr[i], np.inf)
            result += sm * w
            result += shift**2 * w * factor
            result += 2.0 * shift * mean * w
        else:
            w = prob[i]
            sm = _pareto_sm_scalar(piece_cover, att_i, alpha_arr[i], t_arr[i], exit_i)
            mean = _pareto_layer_mean_scalar(piece_cover, att_i, alpha_arr[i], t_arr[i], exit_i)
            result += sm * w
            result += shift**2 * w * factor
            result += 2.0 * shift * mean * w

    if trunc_val is not None and truncation_type == "wd":
        p = (1.0 - float(p_pareto(trunc_val, t_arr[-1], alpha_arr[-1]))) * excess_prob[-1]
        result = (result - p * cover_orig**2) / (1.0 - p)

    return result


# ---------------------------------------------------------------------------
# Layer moments — public vectorised API
# ---------------------------------------------------------------------------


def piecewise_pareto_layer_mean(
    cover: ArrayLike,
    attachment_point: ArrayLike,
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None = None,
    truncation_type: str = "lp",
) -> np.ndarray:
    """Expected loss of the piecewise Pareto in reinsurance layer cover xs attachment_point.

    Parameters
    ----------
    cover:
        Cover of the reinsurance layer. Use np.inf for unlimited.
    attachment_point:
        Attachment point of the layer.
    t:
        Thresholds of the piecewise Pareto distribution.
    alpha:
        Pareto alphas for each piece.
    truncation:
        Upper truncation point. None means no truncation.
    truncation_type:
        ``'lp'`` or ``'wd'``.

    Returns
    -------
    np.ndarray
        Expected layer losses.
    """
    cover_arr = np.asarray(cover, dtype=float)
    att_arr = np.asarray(attachment_point, dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
    if not valid_parameters_piecewise_pareto(
        t_arr, alpha_arr, truncation, truncation_type, comment=True
    ):
        return np.full(np.broadcast(cover_arr, att_arr).shape, np.nan)
    _vfun = np.vectorize(
        lambda c, a: _piecewise_pareto_layer_mean_scalar(
            c, a, t_arr, alpha_arr, truncation, truncation_type
        )
    )
    return _vfun(cover_arr, att_arr)


def piecewise_pareto_layer_sm(
    cover: ArrayLike,
    attachment_point: ArrayLike,
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None = None,
    truncation_type: str = "lp",
) -> np.ndarray:
    """Second moment of the piecewise Pareto in reinsurance layer cover xs attachment_point.

    Parameters
    ----------
    cover:
        Cover of the reinsurance layer.
    attachment_point:
        Attachment point of the layer.
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
        Layer second moments.
    """
    cover_arr = np.asarray(cover, dtype=float)
    att_arr = np.asarray(attachment_point, dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
    if not valid_parameters_piecewise_pareto(
        t_arr, alpha_arr, truncation, truncation_type, comment=True
    ):
        return np.full(np.broadcast(cover_arr, att_arr).shape, np.nan)
    _vfun = np.vectorize(
        lambda c, a: _piecewise_pareto_layer_sm_scalar(
            c, a, t_arr, alpha_arr, truncation, truncation_type
        )
    )
    return _vfun(cover_arr, att_arr)


def piecewise_pareto_layer_var(
    cover: ArrayLike,
    attachment_point: ArrayLike,
    alpha: ArrayLike,
    t: ArrayLike,
    truncation: float | None = None,
    truncation_type: str = "lp",
) -> np.ndarray:
    """Variance of the piecewise Pareto in reinsurance layer cover xs attachment_point.

    Parameters
    ----------
    cover:
        Cover of the reinsurance layer.
    attachment_point:
        Attachment point.
    alpha:
        Pareto alphas for each piece.
    t:
        Thresholds of the piecewise Pareto distribution.
    truncation:
        Upper truncation point.
    truncation_type:
        ``'lp'`` or ``'wd'``.

    Returns
    -------
    np.ndarray
        Layer variances.

    Notes
    -----
    Argument order matches R's ``PiecewisePareto_Layer_Var(Cover, AP, alpha, t)``.
    """
    sm = piecewise_pareto_layer_sm(cover, attachment_point, t, alpha, truncation, truncation_type)
    mean = piecewise_pareto_layer_mean(
        cover, attachment_point, t, alpha, truncation, truncation_type
    )
    return np.where(np.isinf(sm), np.inf, sm - mean**2)


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

    use_rt = rt.max() > t_arr[0]
    has_censored = bool(cens.any())

    # Extended threshold array (k+1 elements, last = effective truncation)
    t_ext = np.append(t_arr, trunc_eff)

    alpha_hat = np.zeros(k)

    if not use_rt and not has_censored:
        # Simple closed-form (no reporting thresholds, no censoring)
        for i in range(k):
            mask_i = ls >= t_ext[i]
            ls_i = ls[mask_i]
            w_i = w[mask_i]
            w_sum_ip1 = w[ls >= t_ext[i + 1]].sum()
            denom = np.sum(w_i * np.log(np.minimum(ls_i, t_ext[i + 1]) / t_ext[i]))
            alpha_hat[i] = (w_i.sum() - w_sum_ip1) / denom

    elif not has_censored:
        # Reporting thresholds, no censoring
        for i in range(k):
            mask = (ls >= t_ext[i]) & (rt < t_ext[i + 1])
            ls_i = ls[mask]
            w_i = w[mask]
            rt_i = rt[mask]
            w_sum_ip1 = w[(ls >= t_ext[i + 1]) & (rt < t_ext[i + 1])].sum()
            denom = np.sum(
                w_i * np.log(np.minimum(ls_i, t_ext[i + 1]) / np.maximum(t_ext[i], rt_i))
            )
            alpha_hat[i] = (w_i.sum() - w_sum_ip1) / denom

    else:
        # Reporting thresholds + censoring
        w_nc = w.copy()
        w_nc[cens] = 0.0
        for i in range(k):
            mask = (ls >= t_ext[i]) & (rt < t_ext[i + 1])
            ls_i = ls[mask]
            w_i = w[mask]
            w_nc_i = w_nc[mask]
            rt_i = rt[mask]
            w_nc_ip1 = w_nc[(ls >= t_ext[i + 1]) & (rt < t_ext[i + 1])].sum()
            denom = np.sum(
                w_i * np.log(np.minimum(ls_i, t_ext[i + 1]) / np.maximum(t_ext[i], rt_i))
            )
            alpha_hat[i] = (w_nc_i.sum() - w_nc_ip1) / denom

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
