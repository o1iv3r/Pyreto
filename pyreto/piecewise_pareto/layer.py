"""Expected value, second moment and variance of piecewise Pareto layer losses."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyreto._validation import valid_parameters_piecewise_pareto
from pyreto.pareto.distribution import p_pareto
from pyreto.pareto.layer import _pareto_layer_mean_scalar, _pareto_layer_var_scalar

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
