"""Fitting utilities: Panjer distribution, local Pareto alpha, reference fitting.

Port of sections of Functions.R from the R Pareto package.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from pyreto.gen_pareto import d_gen_pareto, p_gen_pareto
from pyreto.matching import piecewise_pareto_match_layer_losses
from pyreto.pareto import d_pareto, p_pareto
from pyreto.piecewise_pareto import d_piecewise_pareto, p_piecewise_pareto

# ---------------------------------------------------------------------------
# Panjer (Poisson / NegBin / Binomial) distribution
# ---------------------------------------------------------------------------


def _binomial_parameters(mean: float, dispersion: float) -> tuple[int, float]:
    """Number of trials and success probability for the Panjer binomial case.

    A binomial distribution has dispersion ``1 - p``, but ``mean / (1 - dispersion)``
    is generally not an integer.  As in R's ``dPanjer`` / ``rPanjer``, the number of
    trials is rounded up and ``p`` is recomputed so the mean is matched exactly; the
    dispersion is adjusted instead (with a warning if it moves by more than 0.01).
    """
    n_bin = max(int(np.ceil(mean / (1.0 - dispersion))), 1)
    p_bin = mean / n_bin
    adjusted = 1.0 - p_bin
    if abs(dispersion - adjusted) > 0.01:
        warnings.warn(
            f"Dispersion has been adjusted from {dispersion:.2f} to {adjusted:.2f}"
            " to obtain a matching binomial distribution."
        )
    return n_bin, p_bin


def d_panjer(k: int, *, mean: float, dispersion: float = 1.0) -> float:
    """Probability mass function of the Panjer (a, b, 0) distribution.

    Parameters
    ----------
    k:
        Non-negative integer value.
    mean:
        Expected value (must be positive).
    dispersion:
        Variance-to-mean ratio.  ``1`` → Poisson, ``>1`` → Negative Binomial,
        ``<1`` → Binomial.

    Returns
    -------
    float
        P(X = k).
    """
    k = int(k)
    if k < 0:
        return 0.0
    if dispersion == 1.0:
        return float(stats.poisson.pmf(k, mu=mean))
    if dispersion > 1.0:
        r = mean / (dispersion - 1.0)
        p = 1.0 / dispersion
        return float(stats.nbinom.pmf(k, n=r, p=p))
    n_bin, p_bin = _binomial_parameters(mean, dispersion)
    return float(stats.binom.pmf(k, n=n_bin, p=p_bin))


def r_panjer(
    n: int,
    *,
    mean: float,
    dispersion: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random samples from the Panjer (a, b, 0) distribution.

    Parameters
    ----------
    n:
        Number of samples to draw.
    mean:
        Expected value.
    dispersion:
        Variance-to-mean ratio.
    rng:
        NumPy random generator.  If ``None``, uses ``np.random.default_rng()``.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n,)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(n)
    if dispersion == 1.0:
        return rng.poisson(mean, size=n)
    if dispersion > 1.0:
        r = mean / (dispersion - 1.0)
        p = 1.0 / dispersion
        return rng.negative_binomial(r, p, size=n)
    n_bin, p_bin = _binomial_parameters(mean, dispersion)
    return rng.binomial(n_bin, p_bin, size=n)


# ---------------------------------------------------------------------------
# Local Pareto alpha
# ---------------------------------------------------------------------------


def _scipy_pdf_sf(
    distribution: str, kwargs: dict[str, float | str | np.ndarray], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """PDF and survival function for the scipy-backed distributions, else None."""
    if distribution == "norm":
        dist = stats.norm(loc=float(kwargs.get("mean", 0.0)), scale=float(kwargs.get("sd", 1.0)))
    elif distribution == "lnorm":
        meanlog = float(kwargs.get("meanlog", 0.0))
        dist = stats.lognorm(s=float(kwargs.get("sdlog", 1.0)), scale=np.exp(meanlog))
    elif distribution == "gamma":
        if "scale" in kwargs:
            scale = float(kwargs["scale"])
        else:
            scale = 1.0 / float(kwargs["rate"])
        dist = stats.gamma(a=float(kwargs["shape"]), scale=scale)
    elif distribution == "weibull":
        dist = stats.weibull_min(c=float(kwargs["shape"]), scale=float(kwargs.get("scale", 1.0)))
    elif distribution == "exp":
        if "rate" in kwargs:
            scale = 1.0 / float(kwargs["rate"])
        else:
            scale = float(kwargs.get("scale", 1.0))
        dist = stats.expon(scale=scale)
    else:
        return None
    return dist.pdf(x), dist.sf(x)


def local_pareto_alpha(
    x: ArrayLike,
    distribution: str,
    **kwargs: float | str | np.ndarray,
) -> np.ndarray:
    """Local Pareto alpha (hazard elasticity) at each point in x.

    Computes alpha(x) = x * f(x) / S(x), where f is the density function
    and S = 1 - F is the survival function.

    Parameters
    ----------
    x:
        Points at which to evaluate the local Pareto alpha.
    distribution:
        Name of the distribution.  Supported values:

        - ``"norm"`` — Normal; kwargs: ``mean``, ``sd``
        - ``"lnorm"`` — Log-normal; kwargs: ``meanlog``, ``sdlog``
        - ``"gamma"`` — Gamma; kwargs: ``shape``, ``scale`` (or ``rate``)
        - ``"weibull"`` — Weibull; kwargs: ``shape``, ``scale``
        - ``"exp"`` — Exponential; kwargs: ``rate`` (or ``scale``)
        - ``"Pareto"`` — Pareto; kwargs: ``t``, ``alpha`` (and optionally
          ``truncation``)
        - ``"GenPareto"`` — Generalized Pareto; kwargs: ``t``, ``alpha_ini``,
          ``alpha_tail`` (and optionally ``truncation``)
        - ``"PiecewisePareto"`` — Piecewise Pareto; kwargs: ``t`` (array),
          ``alpha`` (array), and optionally ``truncation``,
          ``truncation_type``
    **kwargs:
        Distribution parameters (see above).

    Returns
    -------
    np.ndarray
        Local Pareto alpha values at each point in x.
    """
    x_arr = np.asarray(x, dtype=float)

    scipy_vals = _scipy_pdf_sf(distribution, kwargs, x_arr)
    if scipy_vals is not None:
        pdf_vals, sf_vals = scipy_vals

    elif distribution == "Pareto":
        t = float(kwargs["t"])
        alpha_p = float(kwargs["alpha"])
        truncation = kwargs.get("truncation", None)
        trunc = float(truncation) if truncation is not None else None
        pdf_vals = d_pareto(x_arr, t=t, alpha=alpha_p, truncation=trunc)
        sf_vals = 1.0 - p_pareto(x_arr, t=t, alpha=alpha_p, truncation=trunc)

    elif distribution == "GenPareto":
        t = float(kwargs["t"])
        alpha_ini = float(kwargs["alpha_ini"])
        alpha_tail = float(kwargs["alpha_tail"])
        truncation = kwargs.get("truncation", None)
        trunc = float(truncation) if truncation is not None else None
        pdf_vals = d_gen_pareto(
            x_arr, t=t, alpha_ini=alpha_ini, alpha_tail=alpha_tail, truncation=trunc
        )
        sf_vals = 1.0 - p_gen_pareto(
            x_arr, t=t, alpha_ini=alpha_ini, alpha_tail=alpha_tail, truncation=trunc
        )

    elif distribution == "PiecewisePareto":
        t = np.asarray(kwargs["t"], dtype=float)
        alpha_pp = np.asarray(kwargs["alpha"], dtype=float)
        truncation = kwargs.get("truncation", None)
        trunc = float(truncation) if truncation is not None else None
        trunc_type = str(kwargs.get("truncation_type", "lp"))
        pdf_vals = d_piecewise_pareto(
            x_arr, t=t, alpha=alpha_pp, truncation=trunc, truncation_type=trunc_type
        )
        sf_vals = 1.0 - p_piecewise_pareto(
            x_arr, t=t, alpha=alpha_pp, truncation=trunc, truncation_type=trunc_type
        )

    else:
        raise ValueError(
            f"Unknown distribution: {distribution!r}. "
            "Supported: norm, lnorm, gamma, weibull, exp, Pareto, GenPareto, PiecewisePareto."
        )

    pdf_vals = np.asarray(pdf_vals, dtype=float)
    sf_vals = np.asarray(sf_vals, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(sf_vals > 0, x_arr * pdf_vals / sf_vals, np.nan)

    return result


# ---------------------------------------------------------------------------
# fit_references
# ---------------------------------------------------------------------------


def fit_references(
    attachment_points: ArrayLike,
    expected_layer_losses: ArrayLike,
    *,
    frequencies: ArrayLike | None = None,
    truncation: float | None = None,
    truncation_type: str = "lp",
    dispersion: float = 1.0,
    tolerance: float = 1e-10,
    alpha_max: float = 100.0,
    merge_tolerance: float = 1e-6,
    rol_tolerance: float = 1e-6,
    minimize_ratios: bool = True,
) -> object:
    """Fit a PPPModel to a set of reference layer expected losses.

    Port of ``Fit_References`` from Functions.R.

    Parameters
    ----------
    attachment_points:
        Attachment points of the reference layers.
    expected_layer_losses:
        Expected loss for each reference layer.
    frequencies:
        Optional known excess frequencies.
    truncation:
        Upper truncation point.
    truncation_type:
        ``"lp"`` or ``"wd"``.
    dispersion:
        Variance-to-mean ratio for the claim count distribution.
    tolerance:
        Numerical tolerance for root-finding.
    alpha_max:
        Maximum Pareto alpha.
    merge_tolerance:
        Tolerance for merging equal-alpha segments.
    rol_tolerance:
        Tolerance for rate-on-line checks.
    minimize_ratios:
        Whether to minimize breakpoint-optimization ratios.

    Returns
    -------
    PPPModel
        Fitted model.
    """
    return piecewise_pareto_match_layer_losses(
        np.asarray(attachment_points, dtype=float),
        np.asarray(expected_layer_losses, dtype=float),
        frequencies=None if frequencies is None else np.asarray(frequencies, dtype=float),
        truncation=truncation,
        truncation_type=truncation_type,
        dispersion=dispersion,
        tolerance=tolerance,
        alpha_max=alpha_max,
        merge_tolerance=merge_tolerance,
        rol_tolerance=rol_tolerance,
        minimize_ratios=minimize_ratios,
    )


# ---------------------------------------------------------------------------
# fit_pml_curve
# ---------------------------------------------------------------------------


def fit_pml_curve(
    return_periods: ArrayLike,
    pml_values: ArrayLike,
    *,
    frequency: float = 1.0,
    truncation: float | None = None,
    truncation_type: str = "lp",
    dispersion: float = 1.0,
    tolerance: float = 1e-10,
    alpha_max: float = 100.0,
    merge_tolerance: float = 1e-6,
    rol_tolerance: float = 1e-6,
    minimize_ratios: bool = True,
) -> object:
    """Fit a PPPModel to a Probable Maximum Loss (PML) curve.

    Port of ``Fit_PML_Curve`` from Functions.R.

    Converts return-period data to expected layer losses and calls
    ``piecewise_pareto_match_layer_losses``.

    Parameters
    ----------
    return_periods:
        Return periods (years).  Must be strictly increasing.
    pml_values:
        PML at each return period.  Must be strictly increasing.
    frequency:
        Annual claim frequency.
    truncation:
        Upper truncation point.
    truncation_type:
        ``"lp"`` or ``"wd"``.
    dispersion:
        Variance-to-mean ratio for the claim count distribution.
    tolerance:
        Numerical tolerance for root-finding.
    alpha_max:
        Maximum Pareto alpha.
    merge_tolerance:
        Tolerance for merging equal-alpha segments.
    rol_tolerance:
        Tolerance for rate-on-line checks.
    minimize_ratios:
        Whether to minimize breakpoint-optimization ratios.

    Returns
    -------
    PPPModel
        Fitted model.
    """
    rp = np.asarray(return_periods, dtype=float)
    pml = np.asarray(pml_values, dtype=float)

    if len(rp) != len(pml):
        raise ValueError("return_periods and pml_values must have the same length.")
    if np.any(np.diff(rp) <= 0):
        raise ValueError("return_periods must be strictly increasing.")
    if np.any(np.diff(pml) <= 0):
        raise ValueError("pml_values must be strictly increasing.")

    # Excess probabilities: P(X > pml[i]) = 1/rp[i]
    # For a model with frequency `frequency`: P(X > x) = fq * S(x) ≈ fq/rp
    # Expected layer loss between PML[i] and PML[i+1]:
    # ELL[i] = (P(X > PML[i]) - P(X > PML[i+1])) * (PML[i+1] - PML[i]) / 2  (trapezoidal)
    # Exact computation uses the survival function.
    # A simpler approach: convert to excess frequencies and layer ELs.
    # Excess frequency at pml[i] = frequency / rp[i]
    exc_fq = frequency / rp

    # Expected layer losses: approximate as rectangular (exact only for exponential tail)
    k = len(pml)
    ell = np.empty(k, dtype=float)
    for i in range(k - 1):
        # Integral of S(x)/S(pml[0]) between pml[i] and pml[i+1], scaled by exc_fq[i]
        ell[i] = (
            exc_fq[i] * (pml[i + 1] - pml[i])
            - (exc_fq[i] - exc_fq[i + 1]) * (pml[i + 1] - pml[i]) / 2.0
        )
    # Last layer: unlimited
    ell[k - 1] = exc_fq[k - 1] * pml[k - 1]  # approximate tail

    warnings.warn(
        "fit_pml_curve uses a trapezoidal approximation for expected layer losses."
        " Results may be inaccurate for coarse return-period grids."
    )

    return piecewise_pareto_match_layer_losses(
        pml,
        ell,
        frequencies=exc_fq,
        truncation=truncation,
        truncation_type=truncation_type,
        dispersion=dispersion,
        tolerance=tolerance,
        alpha_max=alpha_max,
        merge_tolerance=merge_tolerance,
        rol_tolerance=rol_tolerance,
        minimize_ratios=minimize_ratios,
    )
