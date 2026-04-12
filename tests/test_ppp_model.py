"""Tests for PPPModel dataclass (chunk 6, task 12)."""

from __future__ import annotations

import numpy as np
import pytest

from pyreto.ppp_model import PPPModel


class TestPPPModelCreation:
    def test_valid_model(self) -> None:
        model = PPPModel(fq=1.0, t=np.array([1000.0]), alpha=np.array([2.0]))
        assert model.is_valid()

    def test_invalid_model_negative_fq(self) -> None:
        model = PPPModel(fq=-1.0, t=np.array([1000.0]), alpha=np.array([2.0]))
        assert not model.is_valid()

    def test_invalid_model_zero_alpha_last(self) -> None:
        # Last alpha must be strictly positive
        model = PPPModel(fq=1.0, t=np.array([1000.0]), alpha=np.array([0.0]))
        assert not model.is_valid()

    def test_invalid_negative_dispersion(self) -> None:
        model = PPPModel(fq=1.0, t=np.array([1000.0]), alpha=np.array([2.0]), dispersion=-1.0)
        assert not model.is_valid()

    def test_repr(self) -> None:
        model = PPPModel(fq=1.0, t=np.array([1000.0]), alpha=np.array([2.0]))
        assert "PPPModel" in repr(model)

    def test_repr_with_truncation(self) -> None:
        model = PPPModel(fq=1.0, t=np.array([1000.0]), alpha=np.array([2.0]), truncation=5000.0)
        assert "truncation=5000" in repr(model)

    def test_t_converted_to_ndarray(self) -> None:
        model = PPPModel(fq=1.0, t=[1000.0, 2000.0], alpha=[2.0, 3.0])
        assert isinstance(model.t, np.ndarray)
        assert isinstance(model.alpha, np.ndarray)


class TestPPPModelLayerMean:
    def setup_method(self) -> None:
        ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
        el = np.array([1000, 900, 800, 600, 500], dtype=float)
        from pyreto.matching import piecewise_pareto_match_layer_losses

        self.model = piecewise_pareto_match_layer_losses(ap, el)
        self.cover = np.append(np.diff(ap), np.inf)
        self.ap = ap
        self.el = el

    def test_layer_means_match_targets(self) -> None:
        from pyreto.collective_model import layer_mean

        np.testing.assert_allclose(layer_mean(self.model, self.cover, self.ap), self.el, rtol=1e-6)

    def test_layer_var_nonnegative(self) -> None:
        from pyreto.collective_model import layer_var

        result = layer_var(self.model, self.cover, self.ap)
        assert np.all(result >= 0)

    def test_layer_sd_equals_sqrt_var(self) -> None:
        from pyreto.collective_model import layer_sd, layer_var

        v = layer_var(self.model, self.cover, self.ap)
        sd = layer_sd(self.model, self.cover, self.ap)
        np.testing.assert_allclose(sd, np.sqrt(v))

    def test_excess_frequency_monotone(self) -> None:
        from pyreto.collective_model import excess_frequency

        freqs = excess_frequency(self.model, np.array([1000, 2000, 3000, 4000, 5000], dtype=float))
        assert np.all(np.diff(freqs) <= 0)

    def test_simulate_losses(self) -> None:
        from pyreto.collective_model import simulate_losses

        losses = simulate_losses(self.model, nyears=100, seed=0)
        assert isinstance(losses, list)
        assert len(losses) == 100
        # Each element is a list (possibly empty)
        assert all(isinstance(yr, list) for yr in losses)


class TestPanjerDistribution:
    def test_poisson_mean_and_var(self) -> None:
        from pyreto.collective_model import layer_mean, layer_var

        model = PPPModel(fq=10.0, t=np.array([1000.0]), alpha=np.array([2.0]), dispersion=1.0)
        assert layer_mean(model, 1.0, 0.0) == pytest.approx(10.0)
        assert layer_var(model, 1.0, 0.0) == pytest.approx(10.0)

    def test_underdispersed_var(self) -> None:
        from pyreto.collective_model import layer_mean, layer_var

        model = PPPModel(fq=10.0, t=np.array([1000.0]), alpha=np.array([2.0]), dispersion=0.5)
        assert layer_mean(model, 1.0, 0.0) == pytest.approx(10.0)
        assert layer_var(model, 1.0, 0.0) == pytest.approx(5.0)

    def test_overdispersed_var(self) -> None:
        from pyreto.collective_model import layer_mean, layer_var

        model = PPPModel(fq=10.0, t=np.array([1000.0]), alpha=np.array([2.0]), dispersion=2.0)
        assert layer_mean(model, 1.0, 0.0) == pytest.approx(10.0)
        assert layer_var(model, 1.0, 0.0) == pytest.approx(20.0)


class TestPPPModelLayerSdVar:
    """Layer Sd/Var values from R test_functions_PPP_Model.R."""

    def setup_method(self) -> None:
        from pyreto.matching import piecewise_pareto_match_layer_losses

        ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
        el = np.array([1000, 900, 800, 600, 500], dtype=float)
        fqs = np.array([1.1, 0.95, np.nan, np.nan, 0.5])
        self._ap = ap
        self._el = el
        self._fqs = fqs
        self._match = piecewise_pareto_match_layer_losses

    def test_layer_sd_dispersion_1(self) -> None:
        from pyreto.collective_model import layer_sd

        model = self._match(
            self._ap, self._el, frequencies=self._fqs, truncation=10000, truncation_type="wd"
        )
        assert round(float(layer_sd(model, 1000.0, 2000.0)), 3) == pytest.approx(939.264)

    def test_layer_var_dispersion_1(self) -> None:
        from pyreto.collective_model import layer_var

        model = self._match(
            self._ap, self._el, frequencies=self._fqs, truncation=10000, truncation_type="wd"
        )
        assert round(float(layer_var(model, 1000.0, 2000.0)), 3) == pytest.approx(882217.474)

    def test_layer_sd_dispersion_063(self) -> None:
        from pyreto.collective_model import layer_sd

        model = self._match(
            self._ap,
            self._el,
            frequencies=self._fqs,
            truncation=10000,
            truncation_type="wd",
            dispersion=0.63,
        )
        assert round(float(layer_sd(model, 1000.0, 2000.0)), 3) == pytest.approx(780.873)

    def test_layer_var_dispersion_063(self) -> None:
        from pyreto.collective_model import layer_var

        model = self._match(
            self._ap,
            self._el,
            frequencies=self._fqs,
            truncation=10000,
            truncation_type="wd",
            dispersion=0.63,
        )
        assert round(float(layer_var(model, 1000.0, 2000.0)), 3) == pytest.approx(609762.928)

    def test_layer_sd_dispersion_2(self) -> None:
        from pyreto.collective_model import layer_sd

        model = self._match(
            self._ap,
            self._el,
            frequencies=self._fqs,
            truncation=10000,
            truncation_type="wd",
            dispersion=2.0,
        )
        assert round(float(layer_sd(model, 1000.0, 2000.0)), 3) == pytest.approx(1272.235)

    def test_layer_var_dispersion_2(self) -> None:
        from pyreto.collective_model import layer_var

        model = self._match(
            self._ap,
            self._el,
            frequencies=self._fqs,
            truncation=10000,
            truncation_type="wd",
            dispersion=2.0,
        )
        assert round(float(layer_var(model, 1000.0, 2000.0)), 3) == pytest.approx(1618581.110)
