"""Tests for PPPModel dataclass (chunk 6, task 12)."""

from __future__ import annotations

import numpy as np

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
