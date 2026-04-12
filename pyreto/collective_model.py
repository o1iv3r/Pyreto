"""Collective model functions for PPPModel and PGPModel.

Port of PPPModel.R and PGPModel.R from the R Pareto package.
Provides free functions that dispatch to PPPModel or PGPModel implementations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyreto.gen_pareto import gen_pareto_layer_mean, gen_pareto_layer_sm
from pyreto.pgp_model import PGPModel
from pyreto.piecewise_pareto import (
    p_piecewise_pareto,
    piecewise_pareto_layer_mean,
    piecewise_pareto_layer_sm,
)
from pyreto.ppp_model import PPPModel

CollectiveModel = PPPModel | PGPModel


# ---------------------------------------------------------------------------
# layer_mean
# ---------------------------------------------------------------------------


def layer_mean(
    model: CollectiveModel,
    cover: ArrayLike = np.inf,
    attachment_point: ArrayLike = 0,
) -> np.ndarray:
    """Expected aggregate loss in a reinsurance layer.

    Parameters
    ----------
    model:
        A ``PPPModel`` or ``PGPModel`` instance.
    cover:
        Cover of each layer (use ``np.inf`` for unlimited layers).
    attachment_point:
        Attachment point of each layer.

    Returns
    -------
    np.ndarray
        Expected aggregate loss for each layer.
    """

    cover_arr = np.asarray(cover, dtype=float)
    ap_arr = np.asarray(attachment_point, dtype=float)
    fq = float(model.fq)

    if isinstance(model, PPPModel):
        trunc = model.truncation
        trunc_type = model.truncation_type

        def _single(cov: float, ap: float) -> float:
            return fq * float(
                piecewise_pareto_layer_mean(
                    cov,
                    ap,
                    model.t,
                    model.alpha,
                    truncation=trunc,
                    truncation_type=trunc_type,
                )
            )

    elif isinstance(model, PGPModel):

        def _single(cov: float, ap: float) -> float:
            return fq * float(
                gen_pareto_layer_mean(cov, ap, model.t, model.alpha_ini, model.alpha_tail)
            )

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    return np.vectorize(_single)(cover_arr, ap_arr)


# ---------------------------------------------------------------------------
# layer_var
# ---------------------------------------------------------------------------


def layer_var(
    model: CollectiveModel,
    cover: ArrayLike = np.inf,
    attachment_point: ArrayLike = 0,
) -> np.ndarray:
    """Variance of aggregate loss in a reinsurance layer.

    Uses the compound distribution formula:
    Var[S] = fq * E[X^2] + fq * (dispersion - 1) * E[X]^2

    Parameters
    ----------
    model:
        A ``PPPModel`` or ``PGPModel`` instance.
    cover:
        Cover of each layer.
    attachment_point:
        Attachment point of each layer.

    Returns
    -------
    np.ndarray
        Aggregate variance for each layer.
    """

    cover_arr = np.asarray(cover, dtype=float)
    ap_arr = np.asarray(attachment_point, dtype=float)
    fq = float(model.fq)
    disp = float(getattr(model, "dispersion", 1.0))

    if isinstance(model, PPPModel):
        trunc = model.truncation
        trunc_type = model.truncation_type

        def _single(cov: float, ap: float) -> float:
            sm = float(
                piecewise_pareto_layer_sm(
                    cov,
                    ap,
                    model.t,
                    model.alpha,
                    truncation=trunc,
                    truncation_type=trunc_type,
                )
            )
            mn = float(
                piecewise_pareto_layer_mean(
                    cov,
                    ap,
                    model.t,
                    model.alpha,
                    truncation=trunc,
                    truncation_type=trunc_type,
                )
            )
            return fq * sm + fq * (disp - 1.0) * mn**2

    elif isinstance(model, PGPModel):

        def _single(cov: float, ap: float) -> float:
            sm = float(gen_pareto_layer_sm(cov, ap, model.t, model.alpha_ini, model.alpha_tail))
            mn = float(gen_pareto_layer_mean(cov, ap, model.t, model.alpha_ini, model.alpha_tail))
            return fq * sm + fq * (disp - 1.0) * mn**2

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    return np.vectorize(_single)(cover_arr, ap_arr)


# ---------------------------------------------------------------------------
# layer_sd
# ---------------------------------------------------------------------------


def layer_sd(
    model: CollectiveModel,
    cover: ArrayLike = np.inf,
    attachment_point: ArrayLike = 0,
) -> np.ndarray:
    """Standard deviation of aggregate loss in a reinsurance layer.

    Parameters
    ----------
    model:
        A ``PPPModel`` or ``PGPModel`` instance.
    cover:
        Cover of each layer.
    attachment_point:
        Attachment point of each layer.

    Returns
    -------
    np.ndarray
        Aggregate standard deviation for each layer.
    """
    return np.sqrt(layer_var(model, cover, attachment_point))


# ---------------------------------------------------------------------------
# excess_frequency
# ---------------------------------------------------------------------------


def excess_frequency(
    model: CollectiveModel,
    x: ArrayLike,
) -> np.ndarray:
    """Expected claim frequency in excess of threshold(s).

    Parameters
    ----------
    model:
        A ``PPPModel`` or ``PGPModel`` instance.
    x:
        Threshold value(s).

    Returns
    -------
    np.ndarray
        Expected excess frequency at each threshold.
    """
    from pyreto.gen_pareto import p_gen_pareto

    x_arr = np.asarray(x, dtype=float)
    fq = float(model.fq)

    if isinstance(model, PPPModel):
        trunc = model.truncation
        trunc_type = model.truncation_type

        def _single(xi: float) -> float:
            return fq * (
                1.0
                - float(
                    p_piecewise_pareto(
                        xi,
                        model.t,
                        model.alpha,
                        truncation=trunc,
                        truncation_type=trunc_type,
                    )
                )
            )

    elif isinstance(model, PGPModel):

        def _single(xi: float) -> float:
            return fq * (1.0 - float(p_gen_pareto(xi, model.t, model.alpha_ini, model.alpha_tail)))

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    return np.vectorize(_single)(x_arr)


# ---------------------------------------------------------------------------
# simulate_losses
# ---------------------------------------------------------------------------


def simulate_losses(
    model: CollectiveModel,
    nyears: int = 1,
    seed: int | None = None,
) -> list[list[float]]:
    """Simulate annual loss realisations from a collective model.

    Parameters
    ----------
    model:
        A ``PPPModel`` or ``PGPModel`` instance.
    nyears:
        Number of years to simulate.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    list[list[float]]
        A list of length ``nyears``.  Each element is a list of individual
        claim severities (may be empty for zero-loss years).
    """
    from pyreto.gen_pareto import r_gen_pareto
    from pyreto.piecewise_pareto import r_piecewise_pareto

    rng = np.random.default_rng(seed)
    fq = float(model.fq)
    disp = float(getattr(model, "dispersion", 1.0))

    # Sample claim counts for each year
    if disp == 1.0:
        counts = rng.poisson(fq, size=nyears)
    elif disp > 1.0:
        # Negative Binomial: mean=fq, variance=fq*disp => r=fq/(disp-1), p=1/disp
        r_nb = fq / (disp - 1.0)
        p_nb = 1.0 / disp
        counts = rng.negative_binomial(r_nb, p_nb, size=nyears)
    else:
        # Binomial: mean=fq, variance=fq*disp => n=fq/(1-disp), p=1-disp
        n_bin = int(np.ceil(fq / (1.0 - disp)))
        p_bin = fq / n_bin
        counts = rng.binomial(n_bin, p_bin, size=nyears)

    total_claims = int(counts.sum())

    # Sample severities
    if isinstance(model, PPPModel):
        trunc = model.truncation
        trunc_type = model.truncation_type
        if total_claims > 0:
            all_sev = r_piecewise_pareto(
                total_claims,
                model.t,
                model.alpha,
                truncation=trunc,
                truncation_type=trunc_type,
            ).tolist()
        else:
            all_sev = []
    elif isinstance(model, PGPModel):
        if total_claims > 0:
            all_sev = r_gen_pareto(
                total_claims,
                model.t,
                model.alpha_ini,
                model.alpha_tail,
            ).tolist()
        else:
            all_sev = []
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    # Split into years
    result: list[list[float]] = []
    idx = 0
    for n in counts:
        result.append(all_sev[idx : idx + n])
        idx += n
    return result
