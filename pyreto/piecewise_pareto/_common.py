"""Helpers shared by the piecewise Pareto modules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from pyreto._validation import valid_parameters_piecewise_pareto


def _excess_probs(t_arr: np.ndarray, alpha_arr: np.ndarray) -> np.ndarray:
    """P(X > t[i]) with P(X > t[0]) = 1 (unnormalised)."""
    n = len(t_arr)
    ep = np.ones(n)
    if n > 1:
        factors = t_arr[1:] / t_arr[:-1]
        ep[1:] = np.cumprod((1.0 / factors) ** alpha_arr[:-1])
    return ep


def _vectorize_piecewise(
    scalar_fn: Callable[..., float],
    values: tuple[ArrayLike, ...],
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None,
    truncation_type: str,
) -> np.ndarray:
    """Validate the parameters, then apply ``scalar_fn`` elementwise over ``values``.

    ``scalar_fn`` is called as ``scalar_fn(*value_elements, t, alpha, truncation,
    truncation_type)``. Invalid parameters give a warning and an all-NaN result
    with the broadcast shape of ``values``.
    """
    value_arrs = [np.asarray(v, dtype=float) for v in values]
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
    if not valid_parameters_piecewise_pareto(
        t_arr, alpha_arr, truncation, truncation_type, comment=True
    ):
        return np.full(np.broadcast(*value_arrs).shape, np.nan)
    return np.vectorize(lambda *v: scalar_fn(*v, t_arr, alpha_arr, truncation, truncation_type))(
        *value_arrs
    )
