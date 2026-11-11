"""Fit a piecewise Pareto distribution to frequencies and layer losses (port of FitPP.R)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from pyreto.pareto import (
    pareto_find_alpha_btw_fq_layer,
)

# ---------------------------------------------------------------------------
# Alpha-fitting engine (Task 11b) — port of FitPP.R
# ---------------------------------------------------------------------------


@dataclass
class _FitPPResult:
    """Result of the alpha-fitting engine (_fit_pp).

    Attributes
    ----------
    t:
        Breakpoints of the piecewise-Pareto distribution (same convention as
        the R Pareto package: one attachment point per segment).
    alpha:
        Pareto alpha for each segment (same length as ``t``).
    status:
        ``"OK"`` on success, or an error description string.
    """

    t: list[float]
    alpha: list[float]
    status: str = "OK"


# ---------------------------------------------------------------------------
# Private math helpers
# ---------------------------------------------------------------------------


def _ll(a: float, b: float, alpha: float) -> float:
    """Integral of (a/x)^alpha from a to b.

    Port of the inner ``LL`` function in FitPP.R / lp_functions.R.

    LL(a, b, alpha) =
        a * log(b/a)           if alpha == 1
        a/(1-alpha) * ((b/a)^(1-alpha) - 1)   otherwise
    """
    if alpha == 1.0:
        return a * (np.log(b) - np.log(a))
    return a / (1.0 - alpha) * ((b / a) ** (1.0 - alpha) - 1.0)


def _lambda_fn(
    t: float,
    alpha: float,
    s_0: float,
    s_1: float,
    a_0: float,
    a_1: float,
) -> float:
    """Expected loss of layer [a_0, a_1) xs a_0 with a two-piece Pareto.

    The break between the two pieces is at ``t`` with alphas ``alpha`` (below)
    and ``alpha_1`` (above), where ``alpha_1`` is determined by the requirement
    that frequency at ``a_1`` equals ``s_1``:

        alpha_1 = (log(s_1/s_0) - alpha*log(a_0/t)) / log(t/a_1)

    Port of the inner ``lambda`` function in FitPP.R.
    """
    log_t_a1 = np.log(t / a_1)
    if log_t_a1 == 0.0:
        alpha_1 = alpha
    else:
        alpha_1 = (np.log(s_1 / s_0) - alpha * np.log(a_0 / t)) / log_t_a1
    return s_0 * _ll(a_0, t, alpha) + s_0 * (a_0 / t) ** alpha * _ll(t, a_1, alpha_1)


def _calculate_taus(
    s_0: float,
    s_1: float,
    a_0: float,
    a_1: float,
    l_0: float,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    """Find the valid breakpoint range [tau_l, tau_u] for a single segment.

    Port of ``Calculate_taus`` from ``FitPP.R``.

    ``tau_u`` is the breakpoint at which both pieces have the *same* alpha (the
    single-alpha solution); ``tau_l`` is the breakpoint at which the first
    piece has alpha=0 (uniform distribution).

    Returns
    -------
    (tau_l, tau_u) : tuple[float, float]
    """
    delta = tolerance * (a_1 - a_0)
    tol_abs = tolerance * (a_1 - a_0)

    # --- tau_u: solve f(x) = lambda(x, log(s1/s0)/log(a0/x)) - l_0 = 0 ---
    def f(x: float) -> float:
        log_a0_x = np.log(a_0 / x)
        if log_a0_x == 0.0:
            alpha = 0.0
        else:
            alpha = np.log(s_1 / s_0) / log_a0_x
        return _lambda_fn(x, alpha, s_0, s_1, a_0, a_1) - l_0

    tau_u: float | None = None
    try:
        lo, hi = a_0 + delta, a_1 - delta
        if lo < hi and np.sign(f(lo)) != np.sign(f(hi)):
            tau_u = float(brentq(f, lo, hi, xtol=tol_abs))
    except (ValueError, ZeroDivisionError):
        pass

    if tau_u is None:
        mid = (a_0 + a_1) / 2.0
        tau_u = a_1 if f(mid) < 0 else a_0

    # --- tau_l: solve g(x) = lambda(x, 0) - l_0 = 0 ---
    def g(x: float) -> float:
        return _lambda_fn(x, 0.0, s_0, s_1, a_0, a_1) - l_0

    tau_l: float | None = None
    try:
        lo2, hi2 = a_0, a_1
        if np.sign(g(lo2)) != np.sign(g(hi2)):
            tau_l = float(brentq(g, lo2, hi2, xtol=tol_abs))
    except (ValueError, ZeroDivisionError):
        pass

    if tau_l is None:
        mid = (a_0 + a_1) / 2.0
        tau_l = a_0 if g(mid) > 0 else a_1

    return float(tau_l), float(tau_u)


def _calculate_alphas(
    s_0: float,
    s_1: float,
    a_0: float,
    a_1: float,
    l_0: float,
    t: float,
    alpha_max: float = 100.0,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    """Find (alpha_0, alpha_1) for a two-piece Pareto matching layer loss l_0.

    Port of ``Calculate_alphas`` from ``FitPP.R``.

    Given the breakpoint ``t``, alpha_0 is found by root-finding
    (lambda(t, alpha_0) == l_0) and alpha_1 is derived analytically.
    """
    tol_abs = tolerance * (a_1 - a_0)

    def f(alpha: float) -> float:
        val = _lambda_fn(t, alpha, s_0, s_1, a_0, a_1) - l_0
        if not np.isnan(val):
            return val
        # Fallback: nudge alpha down (as in R source)
        for i in range(1, 21):
            val2 = _lambda_fn(t, alpha / 1.1**i, s_0, s_1, a_0, a_1) - l_0
            if not np.isnan(val2):
                return val2
        return val  # still nan — let caller handle

    alpha_0: float
    try:
        f0, fmax = f(0.0), f(alpha_max)
        if np.sign(f0) != np.sign(fmax):
            alpha_0 = float(brentq(f, 0.0, alpha_max, xtol=tol_abs))
        else:
            # Pick the endpoint closest to zero
            alpha_0 = 0.0 if abs(f0) <= abs(fmax) else alpha_max
    except (ValueError, ZeroDivisionError):
        alpha_0 = 0.0

    log_t_a1 = np.log(t / a_1)
    if log_t_a1 == 0.0:
        alpha_1 = alpha_0
    else:
        alpha_1 = (np.log(s_1 / s_0) - alpha_0 * np.log(a_0 / t)) / log_t_a1
        alpha_1 = min(alpha_max, alpha_1)

    return float(max(alpha_0, 0.0)), float(max(alpha_1, 0.0))


def _fit_pp(
    a: np.ndarray,
    s: np.ndarray,
    layer_losses: np.ndarray,
    truncation: float | None,
    tolerance: float = 1e-10,
    alpha_max: float = 100.0,
    minimize_ratios: bool = True,
    merge_tolerance: float = 1e-6,
) -> _FitPPResult:
    """Fit a piecewise-Pareto distribution to layer frequencies and expected losses.

    Port of ``Fit_PP`` from ``FitPP.R``.

    Parameters
    ----------
    a:
        Attachment points (n values, strictly ascending, positive).
    s:
        Excess frequencies at each attachment point (n values, strictly
        descending, all positive).
    layer_losses:
        Expected loss of each contiguous layer [a[k], a[k+1]) xs a[k].
        Accepts n-1 values (finite layers only) or n values (full R convention
        where the last element is the infinite-tail layer EL).
    truncation:
        Upper truncation point, or ``None`` for no truncation.
    tolerance:
        Root-finding tolerance.
    alpha_max:
        Maximum alpha value used during root-finding.
    minimize_ratios:
        If True, choose the breakpoint within [tau_l, tau_u] that minimises
        the ratio max(alpha_0, alpha_1) / min(alpha_0, alpha_1).
    merge_tolerance:
        Consecutive segments whose alphas differ by less than this are merged.

    Returns
    -------
    _FitPPResult
        .t      - breakpoints (one per segment)
        .alpha  - Pareto alpha for each segment
        .status - "OK" or error description
    """
    a = np.asarray(a, dtype=float)
    s = np.asarray(s, dtype=float)
    ll = np.asarray(layer_losses, dtype=float)

    n = len(a)
    result_err = _FitPPResult(t=[], alpha=[], status="")

    if len(s) != n:
        result_err.status = "a and s must have same length!"
        return result_err
    # ll must have n elements: n-1 finite layers + 1 infinite-tail layer (R convention).
    # Also accepts n-1 elements; in that case the topmost alpha is derived from
    # the last two frequencies via a simple power-law.
    if len(ll) not in (n - 1, n):
        result_err.status = "layer_losses must have n-1 or n elements (n = len(a))."
        return result_err
    if n < 2 or np.min(np.diff(a)) <= 0:
        result_err.status = "a must be ascending"
        return result_err
    if np.max(np.diff(s)) >= 0:
        result_err.status = "s must be descending"
        return result_err
    if a[0] <= 0:
        result_err.status = "a must be positive."
        return result_err
    if s[n - 1] <= 0:
        result_err.status = "s must be positive."
        return result_err

    ll_has_tail = len(ll) == n  # True -> R convention; False -> tail alpha from freqs

    # Output arrays of length 2*n - 1 (as in R source).
    # Positions 0, 2, ..., 2*(n-1) hold attachment points a[k].
    # Positions 1, 3, ..., 2*(n-1)-1 hold intermediate breakpoints.
    q = 2 * n - 1
    t_arr = np.zeros(q)
    alpha_arr = np.zeros(q)

    # Place the n attachment points at even positions (0-based: 0, 2, 4, ...)
    t_arr[0::2] = a

    # Alpha for the topmost segment (position q-1)
    if ll_has_tail:
        # Full R convention: use pareto_find_alpha_btw_fq_layer with ll[n-1]
        alpha_arr[q - 1] = pareto_find_alpha_btw_fq_layer(
            a[n - 1],
            s[n - 1],
            np.inf,
            a[n - 1],
            ll[n - 1],
            max_alpha=alpha_max,
            tolerance=tolerance,
            truncation=truncation,
        )
    else:
        # Derive topmost alpha from the last two frequencies (power-law)
        log_ratio = np.log(a[n - 1] / a[n - 2])
        if log_ratio > 0:
            alpha_arr[q - 1] = float(np.log(s[n - 2] / s[n - 1]) / log_ratio)
        else:
            alpha_arr[q - 1] = alpha_max
        alpha_arr[q - 1] = max(0.0, min(alpha_max, alpha_arr[q - 1]))

    # For each consecutive pair of attachment points, find the breakpoint and alphas
    for k in range(n - 1):
        taus = list(_calculate_taus(s[k], s[k + 1], a[k], a[k + 1], ll[k], tolerance=tolerance))

        # Clamp taus to [lower_bound, upper_bound] as in R
        lower_bound = min(
            a[k] * (s[k] / s[k + 1]) ** (1.0 / alpha_max),
            (a[k] + a[k + 1]) / 2.0,
        )
        upper_bound = max(
            a[k + 1] * (s[k + 1] / s[k]) ** (1.0 / alpha_max),
            (a[k] + a[k + 1]) / 2.0,
        )

        if taus[1] < lower_bound:
            taus = [lower_bound, lower_bound]
        elif taus[0] > upper_bound:
            taus = [upper_bound, upper_bound]
        else:
            taus[0] = max(taus[0], lower_bound)
            taus[1] = min(taus[1], upper_bound)

        t_mid = (taus[0] + taus[1]) / 2.0

        if minimize_ratios and taus[0] < taus[1]:

            def _penalty(tv: float, _k: int = k) -> float:
                al0, al1 = _calculate_alphas(
                    s[_k],
                    s[_k + 1],
                    a[_k],
                    a[_k + 1],
                    ll[_k],
                    tv,
                    tolerance=tolerance,
                    alpha_max=alpha_max,
                )
                if al0 == 0.0 and al1 == 0.0:
                    return 1.0
                mn = min(al0, al1)
                mx = max(al0, al1)
                if mn == 0.0:
                    return 1000.0
                ratio = mx / mn
                return min(ratio, 1000.0)

            opt = minimize_scalar(
                _penalty,
                bounds=(taus[0], taus[1]),
                method="bounded",
                options={"xatol": tolerance * (a[k + 1] - a[k])},
            )
            t_mid = float(opt.x)
        elif not minimize_ratios:
            t_mid = taus[0]

        # t_arr index for breakpoint between a[k] and a[k+1] is 2k+1
        t_arr[2 * k + 1] = t_mid

        al0, al1 = _calculate_alphas(
            s[k],
            s[k + 1],
            a[k],
            a[k + 1],
            ll[k],
            t_mid,
            tolerance=tolerance,
            alpha_max=alpha_max,
        )
        alpha_arr[2 * k] = al0  # segment below breakpoint (starts at a[k])
        alpha_arr[2 * k + 1] = al1  # segment above breakpoint (starts at t_mid)

    # Merge consecutive segments with equal alpha
    is_equal = np.concatenate(([False], np.abs(np.diff(alpha_arr)) < merge_tolerance))
    t_out = t_arr[~is_equal].tolist()
    alpha_out = alpha_arr[~is_equal].tolist()

    return _FitPPResult(t=t_out, alpha=alpha_out, status="OK")
