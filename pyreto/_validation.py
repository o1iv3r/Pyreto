"""Internal input validation helpers.

Mirrors R's ValidationFunctions.R. All functions are private (prefixed _).
These are not part of the public API.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike


def is_positive_vector(x: ArrayLike) -> bool:
    """Return True if x is an array of strictly positive numbers (NaN treated as non-positive)."""
    arr = np.asarray(x, dtype=float)
    return bool(np.all(arr > 0))


def is_nonnegative_finite_vector(x: ArrayLike) -> bool:
    """Return True if x is an array of non-negative finite numbers."""
    arr = np.asarray(x, dtype=float)
    return bool(np.all(arr >= 0) and np.all(np.isfinite(arr)))


def is_positive_finite_vector(x: ArrayLike) -> bool:
    """Return True if x is an array of strictly positive finite numbers."""
    arr = np.asarray(x, dtype=float)
    return bool(np.all(arr > 0) and np.all(np.isfinite(arr)))


def is_positive_or_na_finite_vector(x: ArrayLike) -> bool:
    """Return True if x contains only positive finite numbers or NaN."""
    arr = np.asarray(x, dtype=float)
    return bool(np.all((arr > 0) | np.isnan(arr)) and np.all(np.isfinite(arr) | np.isnan(arr)))


def is_nonnegative_or_na_finite_vector(x: ArrayLike) -> bool:
    """Return True if x contains only non-negative finite numbers or NaN."""
    arr = np.asarray(x, dtype=float)
    return bool(np.all((arr >= 0) | np.isnan(arr)) and np.all(np.isfinite(arr) | np.isnan(arr)))


def is_positive_finite_number(x: float) -> bool:
    """Return True if x is a single strictly positive finite number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) and x > 0


def is_nonnegative_finite_number(x: float) -> bool:
    """Return True if x is a single non-negative finite number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) and x >= 0


def is_positive_number(x: float) -> bool:
    """Return True if x is a single strictly positive number (inf allowed)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not np.isnan(x) and x > 0


def is_nonnegative_number(x: float) -> bool:
    """Return True if x is a single non-negative number (inf allowed)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not np.isnan(x) and x >= 0


def valid_parameters_pareto(
    t: float,
    alpha: float,
    truncation: float | None,
    allow_alpha_zero: bool = False,
    comment: bool = False,
) -> bool:
    """Validate Pareto parameters.

    Returns True if valid, False otherwise (with optional warning).
    """
    ok = True
    msgs: list[str] = []

    if not is_positive_finite_number(t):
        msgs.append("t must be a positive finite number.")
        ok = False
    if allow_alpha_zero:
        if not is_nonnegative_finite_number(alpha):
            msgs.append("alpha must be a non-negative finite number.")
            ok = False
    else:
        if not is_positive_number(alpha):
            msgs.append("alpha must be a positive number.")
            ok = False
    if truncation is not None:
        if not is_positive_finite_number(truncation):
            msgs.append("truncation must be a positive finite number.")
            ok = False
        elif t is not None and np.isfinite(t) and truncation <= t:
            msgs.append("truncation must be greater than t.")
            ok = False

    if comment and msgs:
        warnings.warn(" ".join(msgs))
    return ok


def valid_parameters_piecewise_pareto(
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None,
    truncation_type: str,
    comment: bool = False,
) -> bool:
    """Validate PiecewisePareto parameters."""
    t_arr = np.asarray(t, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    ok = True
    msgs: list[str] = []

    if truncation_type not in ("lp", "wd"):
        msgs.append("truncation_type must be 'lp' or 'wd'.")
        ok = False
    if not is_positive_finite_vector(t_arr):
        msgs.append("t must be a positive finite vector.")
        ok = False
    elif len(t_arr) > 1 and not np.all(np.diff(t_arr) > 0):
        msgs.append("t must be strictly increasing.")
        ok = False
    if not is_nonnegative_finite_vector(alpha_arr):
        msgs.append("alpha must be non-negative and finite.")
        ok = False
    elif float(alpha_arr[-1]) <= 0:
        msgs.append("Last alpha must be strictly positive.")
        ok = False
    if len(t_arr) != len(alpha_arr):
        msgs.append("t and alpha must have the same length.")
        ok = False
    if truncation is not None:
        if not is_positive_number(truncation):
            msgs.append("truncation must be a positive number.")
            ok = False
        elif truncation <= float(t_arr[-1]):
            msgs.append("truncation must be greater than max(t).")
            ok = False

    if comment and msgs:
        warnings.warn(" ".join(msgs))
    return ok


def valid_parameters_gen_pareto(
    t: float,
    alpha_ini: float,
    alpha_tail: float,
    truncation: float | None,
    comment: bool = False,
) -> bool:
    """Validate generalized Pareto parameters."""
    ok = True
    msgs: list[str] = []

    if not is_positive_finite_number(t):
        msgs.append("t must be a positive finite number.")
        ok = False
    if not is_positive_number(alpha_ini):
        msgs.append("alpha_ini must be a positive number.")
        ok = False
    if not is_positive_number(alpha_tail):
        msgs.append("alpha_tail must be a positive number.")
        ok = False
    if truncation is not None:
        if not is_positive_finite_number(truncation):
            msgs.append("truncation must be a positive finite number.")
            ok = False
        elif np.isfinite(t) and truncation <= t:
            msgs.append("truncation must be greater than t.")
            ok = False

    if comment and msgs:
        warnings.warn(" ".join(msgs))
    return ok
