"""Collective model functions for PPPModel (Panjer & Piecewise Pareto).

Port of PPPModel.R from the R Pareto package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyreto.piecewise_pareto import p_piecewise_pareto, piecewise_pareto_layer_mean
from pyreto.ppp_model import PPPModel


def layer_mean(
    model: PPPModel,
    cover: ArrayLike,
    attachment_point: ArrayLike,
) -> np.ndarray:
    """Expected loss of one or more reinsurance layers for a PPPModel.

    Parameters
    ----------
    model:
        A ``PPPModel`` instance.
    cover:
        Cover of each layer (use ``np.inf`` for unlimited layers).
    attachment_point:
        Attachment point of each layer.

    Returns
    -------
    np.ndarray
        Expected loss of each layer.
    """
    cover_arr = np.asarray(cover, dtype=float)
    ap_arr = np.asarray(attachment_point, dtype=float)

    trunc = getattr(model, "truncation", None)
    trunc_type = getattr(model, "truncation_type", "lp")
    fq = float(getattr(model, "fq", 1.0))

    def _single(cov: float, ap: float) -> float:
        return float(
            fq
            * piecewise_pareto_layer_mean(
                cov,
                ap,
                model.t,
                model.alpha,
                truncation=trunc,
                truncation_type=trunc_type,
            )
        )

    return np.vectorize(_single)(cover_arr, ap_arr)


def excess_frequency(
    model: PPPModel,
    x: ArrayLike,
) -> np.ndarray:
    """Expected frequency in excess of threshold(s) for a PPPModel.

    Parameters
    ----------
    model:
        A ``PPPModel`` instance.
    x:
        Threshold value(s).

    Returns
    -------
    np.ndarray
        Expected excess frequency at each threshold.
    """
    x_arr = np.asarray(x, dtype=float)

    trunc = getattr(model, "truncation", None)
    trunc_type = getattr(model, "truncation_type", "lp")
    fq = float(getattr(model, "fq", 1.0))

    def _single(xi: float) -> float:
        return float(
            fq
            * (
                1.0
                - p_piecewise_pareto(
                    xi,
                    model.t,
                    model.alpha,
                    truncation=trunc,
                    truncation_type=trunc_type,
                )
            )
        )

    return np.vectorize(_single)(x_arr)
