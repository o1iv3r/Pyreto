"""Fitting utilities: Panjer distribution, local Pareto alpha, reference fitting.

Port of sections of Functions.R from the R Pareto package.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats

from pyreto._validation import (
    is_nonnegative_finite_vector,
    is_positive_finite_number,
    is_positive_finite_vector,
    is_positive_vector,
)
from pyreto.gen_pareto import d_gen_pareto, gen_pareto_layer_mean, p_gen_pareto
from pyreto.matching import _calculate_layer_losses, piecewise_pareto_match_layer_losses
from pyreto.pareto import (
    d_pareto,
    p_pareto,
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_fqs,
    pareto_find_alpha_btw_layers,
    pareto_layer_mean,
)
from pyreto.pgp_model import PGPModel
from pyreto.piecewise_pareto import d_piecewise_pareto, p_piecewise_pareto
from pyreto.ppp_model import PPPModel

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

_SEVERITY_DISTRIBUTIONS = ("PiecewisePareto", "Pareto", "GenPareto")


def fit_references(
    covers: ArrayLike | None = None,
    attachment_points: ArrayLike | None = None,
    expected_layer_losses: ArrayLike | None = None,
    thresholds: ArrayLike | None = None,
    frequencies: ArrayLike | None = None,
    model_threshold: float | None = None,
    default_alpha: float = 2.0,
    dispersion: float = 1.0,
    alpha_max: float = 100.0,
    severity_distribution: str = "PiecewisePareto",
    ignore_inconsistent_references: bool = False,
) -> PPPModel | PGPModel:
    """Fit a collective model to a wishlist of references.

    Port of ``Fit_References`` from Functions.R.  The references are layers
    (``covers`` xs ``attachment_points`` with ``expected_layer_losses``) and
    excess frequencies (``frequencies`` in excess of ``thresholds``).  Layers
    may overlap and need not be contiguous.

    - ``"PiecewisePareto"``: returns a ``PPPModel`` that matches all references.
      The references are first turned into a contiguous tower of layers (via a
      linear program if they overlap or are inconsistent), which is then
      matched with :func:`piecewise_pareto_match_layer_losses`.
    - ``"Pareto"``: returns a ``PPPModel`` with a single Pareto piece that
      minimises the squared relative deviations from the references.
    - ``"GenPareto"``: returns a ``PGPModel`` that minimises the squared
      relative deviations from the references.

    Parameters
    ----------
    covers:
        Covers of the reference layers (``np.inf`` allowed).
    attachment_points:
        Attachment points of the reference layers.
    expected_layer_losses:
        Expected losses of the reference layers.
    thresholds:
        Thresholds for which excess frequencies are given.
    frequencies:
        Expected frequencies in excess of ``thresholds``.
    model_threshold:
        Lowest threshold of the fitted distribution.  Defaults to the smallest
        attachment point or threshold.
    default_alpha:
        Pareto alpha used where an alpha has to be chosen (must be > 1).
    dispersion:
        Variance-to-mean ratio of the claim-count distribution of the result.
    alpha_max:
        Maximum Pareto alpha used for the fit.
    severity_distribution:
        ``"PiecewisePareto"`` (default), ``"Pareto"`` or ``"GenPareto"``.
    ignore_inconsistent_references:
        If True, inconsistent references are dropped (piecewise Pareto only)
        and the model is fitted to the rest (status 1).  If False, inconsistent
        references give status 2.

    Returns
    -------
    PPPModel or PGPModel
        Fitted model.  ``status`` is 0 on success, 1 if some information was
        ignored and 2 if no solution was found (with a warning and the reason
        in ``comment``).
    """
    cov = np.atleast_1d(np.asarray([] if covers is None else covers, dtype=float))
    att = np.atleast_1d(
        np.asarray([] if attachment_points is None else attachment_points, dtype=float)
    )
    ell = np.atleast_1d(
        np.asarray([] if expected_layer_losses is None else expected_layer_losses, dtype=float)
    )
    thr = np.atleast_1d(np.asarray([] if thresholds is None else thresholds, dtype=float))
    fqs = np.atleast_1d(np.asarray([] if frequencies is None else frequencies, dtype=float))

    gen_pareto = severity_distribution == "GenPareto"

    def _fail(msg: str) -> PPPModel | PGPModel:
        warnings.warn(msg)
        if gen_pareto:
            return PGPModel(
                fq=np.nan,
                t=np.nan,
                alpha_ini=np.nan,
                alpha_tail=np.nan,
                dispersion=dispersion,
                status=2,
                comment=msg,
            )
        return PPPModel(
            fq=np.nan,
            t=np.array([np.nan]),
            alpha=np.array([np.nan]),
            dispersion=dispersion,
            status=2,
            comment=msg,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if severity_distribution not in _SEVERITY_DISTRIBUTIONS:
        gen_pareto = False
        return _fail("severity_distribution not valid.")
    if not is_positive_finite_number(dispersion):
        return _fail("Dispersion must be a positive number.")
    if not is_positive_finite_vector(att):
        return _fail("attachment_points must be positive or None.")
    if not is_positive_vector(cov):
        return _fail("covers must be positive or None.")
    if not is_nonnegative_finite_vector(ell):
        return _fail("expected_layer_losses must be positive or None.")
    if not is_positive_finite_vector(thr):
        return _fail("thresholds must be positive or None.")
    if not is_nonnegative_finite_vector(fqs):
        return _fail("frequencies must be positive or None.")
    if not len(att) == len(cov) == len(ell):
        return _fail(
            "attachment_points, covers and expected_layer_losses must have the same length."
        )
    if len(thr) != len(fqs):
        return _fail("thresholds and frequencies must have the same length.")
    if len(att) == 0 and len(thr) == 0:
        return _fail("Either number of layers or number of thresholds must be positive.")
    if model_threshold is None:
        model_threshold = float(np.min(np.concatenate([att, thr])))
    if not is_positive_finite_number(model_threshold):
        return _fail("model_threshold must be a positive number.")
    if model_threshold > np.min(np.concatenate([att, thr])):
        return _fail(
            "model_threshold must be a less than or equal to the lowest threshold"
            " / attachment point."
        )
    if not is_positive_finite_number(default_alpha) or default_alpha <= 1:
        return _fail("default_alpha must be a positive number > 1.")
    if not isinstance(ignore_inconsistent_references, bool):
        return _fail("ignore_inconsistent_references must be True or False.")

    def _model(
        fq: float, alpha_1: float, alpha_2: float, status: int, comment: str
    ) -> PPPModel | PGPModel:
        if gen_pareto:
            return PGPModel(
                fq=fq,
                t=model_threshold,
                alpha_ini=alpha_1,
                alpha_tail=alpha_2,
                dispersion=dispersion,
                status=status,
                comment=comment,
            )
        return PPPModel(
            fq=fq,
            t=np.array([model_threshold]),
            alpha=np.array([alpha_1]),
            dispersion=dispersion,
            status=status,
            comment=comment,
        )

    # ------------------------------------------------------------------
    # At most two references: a single Pareto piece fits them exactly
    # ------------------------------------------------------------------
    small = _fit_two_references(cov, att, ell, thr, fqs, model_threshold, default_alpha)
    if small is not None:
        fq, alpha = small
        return _model(fq, alpha, alpha, 0, "OK")

    if severity_distribution == "PiecewisePareto":
        return _fit_references_piecewise_pareto(
            cov,
            att,
            ell,
            thr,
            fqs,
            model_threshold=model_threshold,
            default_alpha=default_alpha,
            dispersion=dispersion,
            alpha_max=alpha_max,
            ignore_inconsistent_references=ignore_inconsistent_references,
            fail=_fail,
        )

    # ------------------------------------------------------------------
    # Pareto / GenPareto: least squares on relative deviations
    # ------------------------------------------------------------------
    lb = 1.01 if len(cov) > 0 and np.isinf(np.max(cov)) else 0.01

    def _target(fq: float, alphas: np.ndarray) -> float:
        if gen_pareto:
            a_ini, a_tail = float(alphas[0]), float(alphas[1])
            el_model = gen_pareto_layer_mean(cov, att, model_threshold, a_ini, a_tail)
            fq_model = 1.0 - p_gen_pareto(thr, model_threshold, a_ini, a_tail)
        else:
            alpha = float(alphas[0])
            el_model = pareto_layer_mean(cov, att, alpha, t=model_threshold)
            fq_model = 1.0 - p_pareto(thr, model_threshold, alpha)
        result = float(np.sum((fq * el_model - ell) ** 2 / ell**2))
        result += float(np.sum((fq * fq_model - fqs) ** 2 / fqs**2))
        return result if np.isfinite(result) else np.inf

    def _best_alphas(fq: float) -> tuple[np.ndarray, float] | None:
        """Alpha(s) minimising the target for a given frequency (R: optim)."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if gen_pareto:
                    res = optimize.minimize(
                        lambda x: _target(fq, x),
                        x0=np.array([1.5, 1.5]),
                        method="L-BFGS-B",
                        bounds=[(0.01, alpha_max), (lb, alpha_max)],
                    )
                    if not res.success:
                        return None
                    return np.asarray(res.x, dtype=float), float(res.fun)
                res = optimize.minimize_scalar(
                    lambda a: _target(fq, np.array([a])),
                    bounds=(lb, alpha_max),
                    method="bounded",
                )
                if not res.success:
                    return None
                return np.array([float(res.x)]), float(res.fun)
        except (ValueError, ArithmeticError):
            return None

    def _profile(fq: float) -> float:
        best = _best_alphas(fq)
        return np.inf if best is None else best[1]

    if gen_pareto:
        fq_lo, fq_hi = 1e-4, 1e4
        fq_min = fq_lo
        if len(ell) > 0:
            fq_min = max(fq_min, float(np.max(ell / cov)))
        if len(fqs) > 0:
            fq_min = max(fq_min, float(np.max(fqs)))
        fq_max = fq_hi
        for _ in range(10):
            grid = np.exp(np.log(fq_max / fq_min) * np.arange(6) / 5) * fq_min
            values = np.array([_profile(f) for f in grid])
            best = int(np.argmin(values))
            fq_min = grid[max(0, best - 1)]
            fq_max = grid[min(len(grid) - 1, best + 1)]
        fq_fit = float(grid[best])
    else:
        fq_lo, fq_hi = 1e-6, 1e6
        grid = np.exp(np.log(1e12) * np.arange(101) / 100) * 1e-6
        values = np.array([_profile(f) for f in grid])
        best = int(np.argmin(values))
        lo = grid[max(0, best - 1)]
        hi = grid[min(len(grid) - 1, best + 1)]
        res = optimize.minimize_scalar(_profile, bounds=(lo, hi), method="bounded")
        if not res.success or not np.isfinite(res.fun):
            return _fail("No solution found.")
        fq_fit = float(res.x)

    alphas = _best_alphas(fq_fit)
    if alphas is None:
        return _fail("No solution found.")
    alpha_1 = float(alphas[0][0])
    alpha_2 = float(alphas[0][-1])

    if fq_fit in (fq_lo, fq_hi):
        msg = f"Frequency is only fitted in the interval [{fq_lo:g}, {fq_hi:g}]"
        warnings.warn(msg)
        return _model(fq_fit, alpha_1, alpha_2, 1, msg)
    return _model(fq_fit, alpha_1, alpha_2, 0, "OK")


def _fit_two_references(
    cov: np.ndarray,
    att: np.ndarray,
    ell: np.ndarray,
    thr: np.ndarray,
    fqs: np.ndarray,
    model_threshold: float,
    default_alpha: float,
) -> tuple[float, float] | None:
    """Exact single-Pareto fit (frequency, alpha) for at most two references, else None."""

    def _fq_from_threshold(threshold: float, freq: float, alpha: float) -> float:
        return freq / (1.0 - float(p_pareto(threshold, model_threshold, alpha)))

    n_lay, n_thr = len(att), len(thr)
    alpha = fq = np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n_lay == 1 and n_thr == 0:
                alpha = default_alpha
                fq = ell[0] / float(pareto_layer_mean(cov[0], att[0], alpha, t=model_threshold))
            elif n_lay == 0 and n_thr == 1:
                alpha = default_alpha
                fq = _fq_from_threshold(thr[0], fqs[0], alpha)
            elif n_lay == 1 and n_thr == 1:
                alpha = pareto_find_alpha_btw_fq_layer(thr[0], fqs[0], cov[0], att[0], ell[0])
                fq = _fq_from_threshold(thr[0], fqs[0], alpha)
            elif n_lay == 2 and n_thr == 0:
                alpha = pareto_find_alpha_btw_layers(cov[0], att[0], ell[0], cov[1], att[1], ell[1])
                fq = ell[0] / float(pareto_layer_mean(cov[0], att[0], alpha, t=model_threshold))
            elif n_lay == 0 and n_thr == 2:
                alpha = pareto_find_alpha_btw_fqs(thr[0], fqs[0], thr[1], fqs[1])
                fq = _fq_from_threshold(thr[0], fqs[0], alpha)
    except (ValueError, ArithmeticError):
        return None
    if np.isnan(alpha) or np.isnan(fq):
        return None
    return float(fq), float(alpha)


def _references_overlap(cov: np.ndarray, att: np.ndarray, thr: np.ndarray) -> bool:
    """True if any two references overlap (a threshold inside a layer counts) as in R."""
    entry = np.concatenate([att, thr])
    exit_ = np.concatenate([att + cov, thr])
    order = np.argsort(exit_, kind="stable")
    entry, exit_ = entry[order], exit_[order]
    return bool(np.any(exit_[:-1] > entry[1:])) or len(np.unique(thr)) < len(thr)


def _fit_references_piecewise_pareto(
    cov: np.ndarray,
    att: np.ndarray,
    ell: np.ndarray,
    thr: np.ndarray,
    fqs: np.ndarray,
    *,
    model_threshold: float,
    default_alpha: float,
    dispersion: float,
    alpha_max: float,
    ignore_inconsistent_references: bool,
    fail: Callable[[str], PPPModel | PGPModel],
) -> PPPModel | PGPModel:
    """Turn the references into a contiguous tower (LP if needed) and match it."""
    tower = _calculate_layer_losses(
        att,
        cov,
        ell,
        thr,
        fqs,
        overlapping=_references_overlap(cov, att, thr),
        default_alpha=default_alpha,
        ignore_inconsistent_references=ignore_inconsistent_references,
    )
    if tower.status == 4:
        return fail(
            "References are inconsistent. Use the option ignore_inconsistent_references"
            " = True to ignore the inconsistent references."
        )

    with warnings.catch_warnings():
        if tower.status == 1:
            # The tower is fitted as given; only report the ignored references once.
            warnings.simplefilter("ignore")
        model = piecewise_pareto_match_layer_losses(
            tower.att,
            tower.exp_loss,
            frequencies=tower.frequency,
            dispersion=dispersion,
            alpha_max=alpha_max,
        )
    if model.status == 2:
        return model
    if tower.status == 1:
        msg = "References are not consistent. Some data has been ignored."
        warnings.warn(msg)
        model.status = 1
        model.comment = msg

    if model.t[0] > model_threshold:
        model.fq = model.fq * (model.t[0] / model_threshold) ** model.alpha[0]
        model.t[0] = model_threshold
    elif model.t[0] < model_threshold:
        warnings.warn("model_threshold is ignored.")
    return model


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
