"""Tests for PGPModel dataclass and collective model dispatch (chunk 6, task 13)."""

from __future__ import annotations

import numpy as np
import pytest

from pyreto.pgp_model import PGPModel


class TestPGPModelCreation:
    def test_valid_model(self) -> None:
        model = PGPModel(fq=1.0, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)
        assert model.is_valid()

    def test_invalid_negative_fq(self) -> None:
        model = PGPModel(fq=-1.0, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)
        assert not model.is_valid()

    def test_invalid_alpha_ini(self) -> None:
        model = PGPModel(fq=1.0, t=1000.0, alpha_ini=0.0, alpha_tail=2.0)
        assert not model.is_valid()

    def test_invalid_alpha_tail(self) -> None:
        model = PGPModel(fq=1.0, t=1000.0, alpha_ini=1.5, alpha_tail=0.0)
        assert not model.is_valid()

    def test_repr(self) -> None:
        model = PGPModel(fq=1.0, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)
        assert "PGPModel" in repr(model)

    def test_repr_with_truncation(self) -> None:
        model = PGPModel(fq=1.0, t=1000.0, alpha_ini=1.5, alpha_tail=2.0, truncation=5000.0)
        assert "truncation=5000" in repr(model)


class TestPGPModelDispatch:
    def setup_method(self) -> None:
        self.model = PGPModel(fq=0.5, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)

    def test_layer_mean_positive(self) -> None:
        from pyreto.collective_model import layer_mean

        result = layer_mean(self.model, cover=9000.0, attachment_point=1000.0)
        assert float(result) > 0

    def test_layer_var_nonnegative(self) -> None:
        from pyreto.collective_model import layer_var

        result = layer_var(self.model, cover=9000.0, attachment_point=1000.0)
        assert float(result) >= 0

    def test_layer_sd_equals_sqrt_var(self) -> None:
        from pyreto.collective_model import layer_sd, layer_var

        v = layer_var(self.model, cover=9000.0, attachment_point=1000.0)
        sd = layer_sd(self.model, cover=9000.0, attachment_point=1000.0)
        assert float(sd) == pytest.approx(float(np.sqrt(v)))

    def test_excess_frequency_monotone(self) -> None:
        from pyreto.collective_model import excess_frequency

        freqs = excess_frequency(self.model, x=np.array([1000, 2000, 3000, 5000], dtype=float))
        assert np.all(np.diff(freqs) <= 0)

    def test_simulate_losses_returns_list(self) -> None:
        from pyreto.collective_model import simulate_losses

        result = simulate_losses(self.model, nyears=50, seed=0)
        assert isinstance(result, list)
        assert len(result) == 50
        assert all(isinstance(yr, list) for yr in result)
