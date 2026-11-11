"""Collective model functions for PPPModel and PGPModel.

Port of PPPModel.R and PGPModel.R from the R Pareto package.
Provides free functions that dispatch to PPPModel or PGPModel implementations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyreto.fitting import r_panjer
from pyreto.gen_pareto import (
    gen_pareto_layer_mean,
    gen_pareto_layer_sm,
    p_gen_pareto,
    r_gen_pareto,
)
from pyreto.pgp_model import PGPModel
from pyreto.piecewise_pareto import (
    p_piecewise_pareto,
    piecewise_pareto_layer_mean,
    piecewise_pareto_layer_sm,
    r_piecewise_pareto,
)
from pyreto.ppp_model import PPPModel

CollectiveModel = PPPModel | PGPModel


# ---------------------------------------------------------------------------
# Severity dispatch: one place that knows how each model type is evaluated
# ---------------------------------------------------------------------------


def _check_model(model: CollectiveModel) -> None:
    if not isinstance(model, (PPPModel, PGPModel)):
        raise TypeError(f"Unsupported model type: {type(model)}")


def _severity_layer_mean(model: CollectiveModel, cover: float, ap: float) -> float:
    """Expected loss of a single claim in layer ``cover`` xs ``ap``."""
    if isinstance(model, PPPModel):
        return float(
            piecewise_pareto_layer_mean(
                cover,
                ap,
                model.t,
                model.alpha,
                truncation=model.truncation,
                truncation_type=model.truncation_type,
            )
        )
    return float(
        gen_pareto_layer_mean(
            cover, ap, model.t, model.alpha_ini, model.alpha_tail, truncation=model.truncation
        )
    )


def _severity_layer_sm(model: CollectiveModel, cover: float, ap: float) -> float:
    """Second moment of a single claim in layer ``cover`` xs ``ap``."""
    if isinstance(model, PPPModel):
        return float(
            piecewise_pareto_layer_sm(
                cover,
                ap,
                model.t,
                model.alpha,
                truncation=model.truncation,
                truncation_type=model.truncation_type,
            )
        )
    return float(
        gen_pareto_layer_sm(
            cover, ap, model.t, model.alpha_ini, model.alpha_tail, truncation=model.truncation
        )
    )


def _severity_cdf(model: CollectiveModel, x: float) -> float:
    """CDF of a single claim at ``x``."""
    if isinstance(model, PPPModel):
        return float(
            p_piecewise_pareto(
                x,
                model.t,
                model.alpha,
                truncation=model.truncation,
                truncation_type=model.truncation_type,
            )
        )
    return float(
        p_gen_pareto(x, model.t, model.alpha_ini, model.alpha_tail, truncation=model.truncation)
    )


def _severity_sample(model: CollectiveModel, n: int, rng: np.random.Generator) -> list[float]:
    """Draw ``n`` claim severities."""
    if n == 0:
        return []
    if isinstance(model, PPPModel):
        return r_piecewise_pareto(
            n,
            model.t,
            model.alpha,
            truncation=model.truncation,
            truncation_type=model.truncation_type,
            rng=rng,
        ).tolist()
    return r_gen_pareto(
        n, model.t, model.alpha_ini, model.alpha_tail, truncation=model.truncation, rng=rng
    ).tolist()


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
    _check_model(model)
    fq = float(model.fq)
    cover_arr = np.asarray(cover, dtype=float)
    ap_arr = np.asarray(attachment_point, dtype=float)
    return np.vectorize(lambda c, a: fq * _severity_layer_mean(model, c, a))(cover_arr, ap_arr)


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
    _check_model(model)
    fq = float(model.fq)
    disp = float(model.dispersion)
    cover_arr = np.asarray(cover, dtype=float)
    ap_arr = np.asarray(attachment_point, dtype=float)

    def _single(cov: float, ap: float) -> float:
        sm = _severity_layer_sm(model, cov, ap)
        mn = _severity_layer_mean(model, cov, ap)
        return fq * sm + fq * (disp - 1.0) * mn**2

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
    _check_model(model)
    fq = float(model.fq)
    x_arr = np.asarray(x, dtype=float)
    return np.vectorize(lambda xi: fq * (1.0 - _severity_cdf(model, xi)))(x_arr)


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
    _check_model(model)
    rng = np.random.default_rng(seed)
    counts = r_panjer(nyears, mean=float(model.fq), dispersion=float(model.dispersion), rng=rng)
    all_sev = _severity_sample(model, int(counts.sum()), rng)

    # Split into years
    result: list[list[float]] = []
    idx = 0
    for n in counts:
        result.append(all_sev[idx : idx + n])
        idx += n
    return result
