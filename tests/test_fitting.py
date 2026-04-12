"""Tests for fitting utilities (chunk 7, task 14)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyreto.fitting import d_panjer, local_pareto_alpha, r_panjer


class TestLocalParetoAlpha:
    def test_norm(self) -> None:
        x = np.arange(1, 11) * 1e6
        result = local_pareto_alpha(x, "norm", mean=5e6, sd=2e6)
        np.testing.assert_allclose(
            result,
            [
                0.027623931339495,
                0.138789750458851,
                0.431399956408768,
                1.01832086767407,
                1.99471140200716,
                3.42323331110419,
                5.33797346656343,
                7.75470866649017,
                10.6794698977028,
                14.1137239883195,
            ],
            rtol=1e-8,
        )

    def test_lnorm(self) -> None:
        x = np.arange(1, 11) * 1e6
        result = local_pareto_alpha(x, "lnorm", meanlog=0, sdlog=4)
        np.testing.assert_allclose(result[0], 0.92698000118132, rtol=1e-8)

    def test_pareto(self) -> None:
        x = np.arange(1, 11) * 1e6
        result = local_pareto_alpha(x, "Pareto", t=1e6, alpha=1, truncation=20e6)
        np.testing.assert_allclose(result[0], 1.05263157894737, rtol=1e-8)

    def test_pareto_no_truncation(self) -> None:
        # Pure Pareto(t, alpha): local alpha = alpha everywhere
        x = np.array([1e6, 2e6, 5e6])
        result = local_pareto_alpha(x, "Pareto", t=1e6, alpha=2.0)
        np.testing.assert_allclose(result, 2.0, rtol=1e-10)

    def test_returns_ndarray(self) -> None:
        result = local_pareto_alpha(np.array([1e6]), "norm", mean=5e6, sd=2e6)
        assert isinstance(result, np.ndarray)


class TestPanjer:
    def test_d_panjer_poisson(self) -> None:
        mean = 2.0
        for k in range(5):
            expected = math.exp(-mean) * mean**k / math.factorial(k)
            assert d_panjer(k, mean=mean, dispersion=1.0) == pytest.approx(expected, rel=1e-10)

    def test_d_panjer_negbin(self) -> None:
        probs = [d_panjer(k, mean=3.0, dispersion=2.0) for k in range(50)]
        assert sum(probs) == pytest.approx(1.0, rel=1e-6)

    def test_d_panjer_zero_k(self) -> None:
        assert d_panjer(0, mean=2.0, dispersion=1.0) == pytest.approx(math.exp(-2.0))

    def test_d_panjer_negative_k(self) -> None:
        assert d_panjer(-1, mean=2.0, dispersion=1.0) == 0.0

    def test_r_panjer_poisson_mean(self) -> None:
        rng = np.random.default_rng(42)
        samples = r_panjer(100_000, mean=3.0, dispersion=1.0, rng=rng)
        assert samples.mean() == pytest.approx(3.0, rel=0.02)

    def test_r_panjer_negbin_mean(self) -> None:
        rng = np.random.default_rng(42)
        samples = r_panjer(100_000, mean=5.0, dispersion=3.0, rng=rng)
        assert samples.mean() == pytest.approx(5.0, rel=0.02)

    def test_r_panjer_returns_ndarray(self) -> None:
        rng = np.random.default_rng(0)
        result = r_panjer(10, mean=2.0, dispersion=1.0, rng=rng)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
