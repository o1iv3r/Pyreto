"""Tests for pyreto.pareto — ported from test_functions_Pareto.R."""

import numpy as np
import pytest

from pyreto.pareto import (
    d_pareto,
    p_pareto,
    pareto_extrapolation,
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_fqs,
    pareto_find_alpha_btw_layers,
    pareto_layer_mean,
    pareto_layer_sm,
    pareto_layer_var,
    pareto_ml_estimator_alpha,
    q_pareto,
    r_pareto,
)


class TestPPareto:
    def test_basic(self) -> None:
        result = p_pareto(np.array([2000, 4000, 6000]), t=np.array([1000, 2000, 3000]), alpha=2)
        np.testing.assert_allclose(result, [0.75, 0.75, 0.75])

    def test_with_truncation(self) -> None:
        result = p_pareto(
            np.array([2000, 4000, 6000]),
            t=np.array([1000, 2000, 3000]),
            alpha=2,
            truncation=np.array([10000, 20000, 30000]),
        )
        np.testing.assert_allclose(result, [0.75757575757575757] * 3)


class TestDPareto:
    def test_basic(self) -> None:
        result = d_pareto(np.array([2000, 4000, 6000]), t=np.array([1000, 2000, 3000]), alpha=2)
        np.testing.assert_allclose(result, [2.5e-4, 1.25e-4, 8.333333333333333e-5], rtol=1e-10)


class TestQPareto:
    def test_basic(self) -> None:
        result = q_pareto(
            np.array([0.3, 0.6, 0.9]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
        )
        np.testing.assert_allclose(
            result, [1428.5714285714287, 3162.2776601683790, 6463.3040700956490], rtol=1e-10
        )

    def test_with_truncation_scalar(self) -> None:
        result = q_pareto(
            np.array([0.3, 0.6, 0.9]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=10000,
        )
        np.testing.assert_allclose(
            result, [1369.8630136986301, 3071.4755841697556, 6011.2419963076682], rtol=1e-10
        )

    def test_with_truncation_vector(self) -> None:
        result = q_pareto(
            np.array([0.3, 0.6, 0.9]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=np.array([10000, 20000, 30000]),
        )
        np.testing.assert_allclose(
            result, [1369.8630136986301, 3138.8241028717225, 6444.0296890428708], rtol=1e-10
        )


class TestRPareto:
    def test_shape(self) -> None:
        samples = r_pareto(100, t=1000, alpha=2)
        assert samples.shape == (100,)
        assert np.all(samples >= 1000)

    def test_simulation_parity_no_truncation(self) -> None:
        rng = np.random.default_rng(1972)
        n = 1_000_000
        losses = r_pareto(n, t=1000, alpha=1.5, rng=rng)
        xs = np.minimum(8000, np.maximum(0, losses - 2000))
        ratio = round(xs.mean() / 781.75803033941929, 2)
        assert ratio == 1.0

    def test_simulation_parity_with_truncation(self) -> None:
        rng = np.random.default_rng(1972)
        n = 1_000_000
        losses = r_pareto(n, t=1000, alpha=1.5, truncation=20000, rng=rng)
        xs = np.minimum(8000, np.maximum(0, losses - 2000))
        ratio = round(xs.mean() / 700.14314962210699, 2)
        assert ratio == 1.0


class TestParetoLayerMean:
    def test_basic(self) -> None:
        assert pareto_layer_mean(8000, 2000, alpha=2) == pytest.approx(1600)

    def test_with_t(self) -> None:
        assert pareto_layer_mean(8000, 2000, alpha=2, t=1000) == pytest.approx(400)

    def test_with_t_and_truncation(self) -> None:
        assert pareto_layer_mean(8000, 2000, alpha=2, t=5000, truncation=10000) == pytest.approx(
            4666.66666666667
        )

    def test_alpha_zero(self) -> None:
        assert pareto_layer_mean(2000, 1000, 0, truncation=5000, t=500) == pytest.approx(
            835.16520831955449
        )

    def test_vectorised(self) -> None:
        np.testing.assert_allclose(
            pareto_layer_mean(np.arange(1, 4) * 8000, np.arange(1, 4) * 2000, alpha=2),
            np.arange(1, 4) * 1600,
        )

    def test_vectorised_with_t(self) -> None:
        np.testing.assert_allclose(
            pareto_layer_mean(
                np.arange(1, 4) * 8000,
                np.arange(1, 4) * 2000,
                alpha=2,
                t=np.arange(1, 4) * 1000,
            ),
            np.arange(1, 4) * 400,
        )

    def test_vectorised_with_t_and_truncation(self) -> None:
        np.testing.assert_allclose(
            pareto_layer_mean(
                np.arange(1, 4) * 8000,
                np.arange(1, 4) * 2000,
                alpha=2,
                t=np.arange(1, 4) * 5000,
                truncation=np.arange(1, 4) * 10000,
            ),
            np.arange(1, 4) * 4666.66666666667,
        )


class TestParetoLayerSM:
    def test_basic(self) -> None:
        assert pareto_layer_sm(8000, 2000, alpha=2) == pytest.approx(6475503.2994728)

    def test_with_t(self) -> None:
        assert pareto_layer_sm(8000, 2000, alpha=2, t=1000) == pytest.approx(1618875.8248682)

    def test_with_truncation(self) -> None:
        assert pareto_layer_sm(8000, 2000, alpha=2, t=5000, truncation=10000) == pytest.approx(
            23543145.370663
        )

    def test_alpha_zero_with_t(self) -> None:
        assert pareto_layer_sm(2000, 1000, alpha=0, truncation=5000, t=1500) == pytest.approx(
            2584318.901925616
        )

    def test_vectorised(self) -> None:
        np.testing.assert_allclose(
            pareto_layer_sm(np.arange(1, 4) * 8000, np.arange(1, 4) * 2000, alpha=2),
            np.arange(1, 4) ** 2 * 6475503.2994728,
        )


class TestParetoLayerVar:
    def test_basic(self) -> None:
        assert pareto_layer_var(8000, 2000, alpha=2) == pytest.approx(3915503.2994728)

    def test_with_t(self) -> None:
        assert pareto_layer_var(8000, 2000, alpha=2, t=1000) == pytest.approx(1458875.8248682)

    def test_with_truncation(self) -> None:
        assert pareto_layer_var(8000, 2000, alpha=2, t=5000, truncation=10000) == pytest.approx(
            1765367.59288524
        )

    def test_alpha_zero_with_truncation(self) -> None:
        assert pareto_layer_var(2000, 1000, alpha=0, truncation=5000, t=500) == pytest.approx(
            667015.32799764269
        )


class TestParetoExtrapolation:
    def test_ratio(self) -> None:
        assert pareto_extrapolation(1000, 1000, 2000, 2000, alpha=2) == pytest.approx(0.5)

    def test_with_exp_loss(self) -> None:
        assert pareto_extrapolation(
            1000, 1000, 2000, 2000, alpha=2, exp_loss_1=1000
        ) == pytest.approx(500)

    def test_with_truncation(self) -> None:
        assert pareto_extrapolation(
            1000, 1000, 2000, 2000, alpha=2, truncation=3000, exp_loss_1=1000
        ) == pytest.approx(142.85714285714292)

    def test_vectorised(self) -> None:
        result = pareto_extrapolation(
            np.array([1000, 1000, 1000, np.inf]),
            1000,
            np.array([2000, 2000, 2000, np.inf]),
            2000,
            alpha=np.array([2, 2, 2, 0]),
            truncation=np.array([np.inf, np.inf, 3000, 3000]),
            exp_loss_1=np.array([1, 1000, 1000, 100]),
        )
        np.testing.assert_allclose(result, [0.5, 500, 142.85714285714292, 20.975411735345432])


class TestParetoFindAlpha:
    def test_btw_layers_equal(self) -> None:
        assert pareto_find_alpha_btw_layers(1000, 1000, 100, 2000, 2000, 100) == pytest.approx(1)

    def test_btw_layers_halved(self) -> None:
        assert pareto_find_alpha_btw_layers(1000, 1000, 100, 2000, 2000, 50) == pytest.approx(2)

    def test_btw_layers_with_truncation(self) -> None:
        assert pareto_find_alpha_btw_layers(
            1000, 1000, 100, 2000, 2000, 50, truncation=5000
        ) == pytest.approx(1.3871313763147217)

    def test_btw_fq_layer(self) -> None:
        assert pareto_find_alpha_btw_fq_layer(1000, 1, 2000, 2000, 100) == pytest.approx(
            2.9330042139247037
        )

    def test_btw_fq_layer_att_eq_threshold(self) -> None:
        assert pareto_find_alpha_btw_fq_layer(1000, 1, 1000, 1000, 500) == pytest.approx(2)

    def test_btw_fq_layer_threshold_above_att(self) -> None:
        assert pareto_find_alpha_btw_fq_layer(2000, 0.25, 1000, 1000, 500) == pytest.approx(2)

    def test_btw_fq_layer_with_truncation(self) -> None:
        assert pareto_find_alpha_btw_fq_layer(
            1000, 1, 1000, 1000, 500, truncation=5000
        ) == pytest.approx(1.8363401129702193)

    def test_btw_fqs_basic(self) -> None:
        assert pareto_find_alpha_btw_fqs(1000, 1, 2000, 0.5) == pytest.approx(1)
        assert pareto_find_alpha_btw_fqs(2000, 0.25, 1000, 1) == pytest.approx(2)

    def test_btw_fqs_with_truncation(self) -> None:
        assert pareto_find_alpha_btw_fqs(2000, 0.25, 1000, 1, truncation=4000) == pytest.approx(
            1.5849625007211574
        )


class TestParetoMLEstimator:
    def test_basic(self, pareto_losses: np.ndarray) -> None:
        assert round(pareto_ml_estimator_alpha(pareto_losses, 1000), 4) == 3.4060

    def test_with_truncation(self, pareto_losses: np.ndarray) -> None:
        assert round(pareto_ml_estimator_alpha(pareto_losses, 1000, truncation=3000), 4) == 2.9601

    def test_weights_equivalent_to_duplicates(self, pareto_losses: np.ndarray) -> None:
        losses2 = np.concatenate([pareto_losses, pareto_losses[:2]])
        w = np.ones(len(pareto_losses))
        w[:2] = 2
        assert pareto_ml_estimator_alpha(pareto_losses, 1000, weights=w) == pytest.approx(
            pareto_ml_estimator_alpha(losses2, 1000)
        )

    def test_with_reporting_thresholds(self, pareto_losses: np.ndarray) -> None:
        rt = np.array([1000, 1000, 1000, 1200, 1200, 1000, 1500, 1000, 1000, 1000], dtype=float)
        result = pareto_ml_estimator_alpha(pareto_losses, 1000, reporting_thresholds=rt)
        assert round(result, 5) == 4.61698

    def test_with_reporting_thresholds_weights_truncation(self, pareto_losses: np.ndarray) -> None:
        rt = np.array([1000, 1000, 1000, 1200, 1200, 1000, 1500, 1000, 1000, 1000], dtype=float)
        w = np.ones(len(pareto_losses))
        w[:2] = 2
        result = pareto_ml_estimator_alpha(
            pareto_losses, 1000, reporting_thresholds=rt, weights=w, truncation=3000
        )
        assert round(result, 5) == 4.20722

    def test_with_censoring_reporting_thresholds(self, pareto_losses: np.ndarray) -> None:
        rt = np.array([1000, 1000, 1000, 1200, 1200, 1000, 1500, 1000, 1000, 1000], dtype=float)
        censored = np.zeros(len(pareto_losses), dtype=bool)
        censored[:2] = True
        result = pareto_ml_estimator_alpha(
            pareto_losses, 1000, reporting_thresholds=rt, is_censored=censored
        )
        assert round(result, 5) == 3.69358

    def test_with_censoring_rt_weights_truncation(self, pareto_losses: np.ndarray) -> None:
        rt = np.array([1000, 1000, 1000, 1200, 1200, 1000, 1500, 1000, 1000, 1000], dtype=float)
        w = np.ones(len(pareto_losses))
        w[:2] = 2
        censored = np.zeros(len(pareto_losses), dtype=bool)
        censored[:2] = True
        result = pareto_ml_estimator_alpha(
            pareto_losses,
            1000,
            reporting_thresholds=rt,
            is_censored=censored,
            weights=w,
            truncation=3000,
        )
        assert round(result, 5) == 2.47743

    def test_weights_with_truncation_consistency(self, pareto_losses: np.ndarray) -> None:
        losses2 = np.concatenate([pareto_losses, pareto_losses[:2]])
        w = np.ones(len(pareto_losses))
        w[:2] = 2
        assert round(
            pareto_ml_estimator_alpha(pareto_losses, 1000, weights=w, truncation=3000), 5
        ) == round(pareto_ml_estimator_alpha(losses2, 1000, truncation=3000), 5)
