"""Regression tests for the collective-model bugs tracked in issue #4."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyreto.collective_model import excess_frequency, layer_mean, layer_var, simulate_losses
from pyreto.fitting import d_panjer, r_panjer
from pyreto.gen_pareto import gen_pareto_layer_mean, gen_pareto_layer_sm, p_gen_pareto
from pyreto.pgp_model import PGPModel
from pyreto.ppp_model import PPPModel

PPP = PPPModel(fq=2.0, t=np.array([1000.0, 2000.0]), alpha=np.array([1.5, 2.5]))
PGP = PGPModel(fq=2.0, t=1000.0, alpha_ini=1.0, alpha_tail=2.0)
PGP_TRUNC = PGPModel(fq=2.0, t=1000.0, alpha_ini=1.0, alpha_tail=2.0, truncation=5000.0)


# ---------------------------------------------------------------------------
# simulate_losses(seed=...) is reproducible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [PPP, PGP, PGP_TRUNC], ids=["PPP", "PGP", "PGP-truncated"])
@pytest.mark.parametrize("dispersion", [1.0, 2.0, 0.5])
def test_simulate_losses_reproducible_with_seed(
    model: PPPModel | PGPModel, dispersion: float
) -> None:
    model.dispersion = dispersion
    try:
        first = simulate_losses(model, nyears=50, seed=123)
        second = simulate_losses(model, nyears=50, seed=123)
    finally:
        model.dispersion = 1.0
    assert sum(len(year) for year in first) > 0
    assert first == second


def test_simulate_losses_different_seeds_differ() -> None:
    assert simulate_losses(PGP, nyears=50, seed=1) != simulate_losses(PGP, nyears=50, seed=2)


# ---------------------------------------------------------------------------
# PGPModel.truncation is honoured by the collective-model functions
# ---------------------------------------------------------------------------


def test_pgp_layer_mean_uses_truncation() -> None:
    expected = 2.0 * gen_pareto_layer_mean(np.inf, 1000.0, 1000.0, 1.0, 2.0, truncation=5000.0)
    np.testing.assert_allclose(layer_mean(PGP_TRUNC, np.inf, 1000.0), expected)
    assert layer_mean(PGP_TRUNC, np.inf, 1000.0) < layer_mean(PGP, np.inf, 1000.0)
    # Nothing lies above the truncation point.
    assert layer_mean(PGP_TRUNC, np.inf, 6000.0) == pytest.approx(0.0)


def test_pgp_layer_var_uses_truncation() -> None:
    mean = gen_pareto_layer_mean(3000.0, 1000.0, 1000.0, 1.0, 2.0, truncation=5000.0)
    sm = gen_pareto_layer_sm(3000.0, 1000.0, 1000.0, 1.0, 2.0, truncation=5000.0)
    np.testing.assert_allclose(layer_var(PGP_TRUNC, 3000.0, 1000.0), 2.0 * sm)
    assert mean > 0
    assert layer_var(PGP_TRUNC, 3000.0, 1000.0) != pytest.approx(layer_var(PGP, 3000.0, 1000.0))


def test_pgp_excess_frequency_uses_truncation() -> None:
    expected = 2.0 * (1.0 - p_gen_pareto(3000.0, 1000.0, 1.0, 2.0, truncation=5000.0))
    np.testing.assert_allclose(excess_frequency(PGP_TRUNC, 3000.0), expected)
    assert excess_frequency(PGP_TRUNC, 5000.0) == pytest.approx(0.0)


def test_pgp_simulate_losses_respects_truncation() -> None:
    losses = [x for year in simulate_losses(PGP_TRUNC, nyears=2000, seed=0) for x in year]
    assert len(losses) > 0
    assert max(losses) <= 5000.0


# ---------------------------------------------------------------------------
# Binomial case (dispersion < 1) matches R's dPanjer / rPanjer
# ---------------------------------------------------------------------------


def test_binomial_parameters_match_r() -> None:
    # R: n = ceiling(mean / (1 - dispersion)) = ceiling(2.5 / 0.4) = 7, p = mean / n = 2.5 / 7
    from scipy import stats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(8):
            assert d_panjer(k, mean=2.5, dispersion=0.6) == pytest.approx(
                stats.binom.pmf(k, 7, 2.5 / 7), rel=1e-12
            )


def test_d_panjer_binomial_mean_is_exact() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = np.array([d_panjer(k, mean=2.5, dispersion=0.6) for k in range(20)])
    assert probs.sum() == pytest.approx(1.0)
    assert (np.arange(20) * probs).sum() == pytest.approx(2.5, rel=1e-12)


def test_r_panjer_binomial_mean_is_exact() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = r_panjer(200_000, mean=2.5, dispersion=0.6, rng=np.random.default_rng(0))
    assert samples.max() <= 7
    assert samples.mean() == pytest.approx(2.5, rel=0.01)


def test_binomial_dispersion_adjustment_warns() -> None:
    # mean / (1 - dispersion) = 6.25 -> 7 trials, dispersion becomes 1 - 2.5/7 = 0.64
    with pytest.warns(UserWarning, match="Dispersion has been adjusted"):
        d_panjer(1, mean=2.5, dispersion=0.6)
    # An exactly representable binomial needs no adjustment.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d_panjer(1, mean=2.0, dispersion=0.5)


def test_simulate_losses_counts_follow_r_panjer() -> None:
    model = PPPModel(fq=2.0, t=np.array([1000.0]), alpha=np.array([2.0]), dispersion=0.5)
    counts = [len(year) for year in simulate_losses(model, nyears=100, seed=7)]
    expected = r_panjer(100, mean=2.0, dispersion=0.5, rng=np.random.default_rng(7))
    assert counts == expected.tolist()
