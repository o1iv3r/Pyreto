"""Tests for gen_pareto module -- ported from test_functions_GenPareto.R."""

from typing import ClassVar

import numpy as np
import pytest

from pyreto.gen_pareto import (
    d_gen_pareto,
    gen_pareto_layer_mean,
    gen_pareto_layer_sm,
    gen_pareto_layer_var,
    gen_pareto_ml_estimator_alpha,
    p_gen_pareto,
    r_gen_pareto,
)


class TestPGenPareto:
    def test_basic(self) -> None:
        result = p_gen_pareto(
            np.arange(1, 4) * 2000,
            t=np.arange(1, 4) * 1000,
            alpha_ini=np.array([1, 1.5, 2]),
            alpha_tail=np.array([1, 2, 3]),
        )
        np.testing.assert_allclose(result, [0.5, 0.67346938775510212, 0.78399999999999992])


class TestDGenPareto:
    def test_basic(self) -> None:
        result = d_gen_pareto(
            np.arange(1, 4) * 2000,
            t=np.arange(1, 4) * 1000,
            alpha_ini=np.array([1, 1.5, 2]),
            alpha_tail=np.array([1, 2, 3]),
        )
        np.testing.assert_allclose(
            result,
            [2.5e-04, 1.3994169096209912e-04, 8.6400000000000027e-05],
        )


class TestGenParetoLayerMean:
    def test_basic_same_alphas(self) -> None:
        result = gen_pareto_layer_mean(8000, 2000, t=2000, alpha_ini=2, alpha_tail=2)
        assert result == pytest.approx(1600)

    def test_different_t(self) -> None:
        result = gen_pareto_layer_mean(8000, 2000, t=1000, alpha_ini=2, alpha_tail=2)
        assert result == pytest.approx(400)

    def test_with_truncation(self) -> None:
        assert gen_pareto_layer_mean(
            8000, 2000, t=5000, alpha_ini=2, alpha_tail=1, truncation=10000
        ) == pytest.approx(4619.7960825054124)

    def test_small_alpha_ini_with_truncation(self) -> None:
        assert gen_pareto_layer_mean(
            2000, 1000, t=500, alpha_ini=0.01, alpha_tail=3, truncation=5000
        ) == pytest.approx(1308.8417333286236)

    def test_vectorised(self) -> None:
        result = gen_pareto_layer_mean(
            8000,
            2000,
            t=np.array([2000, 1000, 5000]),
            alpha_ini=2,
            alpha_tail=np.array([2, 2, 1]),
            truncation=np.array([np.inf, np.inf, 10000]),
        )
        np.testing.assert_allclose(result, [1600, 400, 4619.7960825054124])


class TestGenParetoLayerSM:
    def test_basic(self) -> None:
        result = gen_pareto_layer_sm(8000, 2000, t=2000, alpha_ini=2, alpha_tail=2)
        assert result == pytest.approx(6475503.2994728)

    def test_diff_alphas(self) -> None:
        result = gen_pareto_layer_sm(8000, 2000, t=1000, alpha_ini=2.5, alpha_tail=2.5)
        assert result == pytest.approx(705034.42336793197)

    def test_with_truncation(self) -> None:
        assert gen_pareto_layer_sm(
            8000, 2000, t=5000, alpha_ini=0.8, alpha_tail=2, truncation=10000
        ) == pytest.approx(27842963.469263397)

    def test_truncation_below_attachment(self) -> None:
        assert gen_pareto_layer_sm(
            2000, 1000, t=500, alpha_ini=0.1, alpha_tail=0.9, truncation=1000
        ) == pytest.approx(0.0)

    def test_vectorised(self) -> None:
        result = gen_pareto_layer_sm(
            np.array([8000, 2000]),
            np.array([2000, 1000]),
            t=np.array([2000, 500]),
            alpha_ini=np.array([2, 0.1]),
            alpha_tail=np.array([2, 0.9]),
            truncation=np.array([np.inf, 1000]),
        )
        np.testing.assert_allclose(result, [6475503.2994728, 0])


class TestGenParetoLayerVar:
    def test_basic(self) -> None:
        result = gen_pareto_layer_var(8000, 2000, t=1000, alpha_ini=2.1, alpha_tail=4)
        assert result == pytest.approx(398133.28976341535)

    def test_same_alphas(self) -> None:
        result = gen_pareto_layer_var(8000, 2000, t=1000, alpha_ini=2, alpha_tail=2)
        assert result == pytest.approx(1458875.8248682)

    def test_with_truncation_diff_alphas(self) -> None:
        assert gen_pareto_layer_var(
            8000, 2000, t=5000, alpha_ini=1.5, alpha_tail=2.7, truncation=10000
        ) == pytest.approx(1869703.2726483792)

    def test_small_alpha_tail_with_truncation(self) -> None:
        assert gen_pareto_layer_var(
            2000, 1000, t=500, alpha_ini=3, alpha_tail=1, truncation=5000
        ) == pytest.approx(196333.67104755351)

    def test_vectorised(self) -> None:
        result = gen_pareto_layer_var(
            np.array([8000, 2000]),
            np.array([2000, 1000]),
            t=np.array([1000, 500]),
            alpha_ini=np.array([2.1, 3]),
            alpha_tail=np.array([4, 1]),
            truncation=np.array([np.inf, 5000]),
        )
        np.testing.assert_allclose(result, [398133.28976341535, 196333.67104755351])


class TestRGenPareto:
    def test_with_truncation(self) -> None:
        """Simulated layer mean should match analytical value (ratio ~1)."""
        rng = np.random.default_rng(1972)
        n = 1_000_000
        t, alpha_ini, alpha_tail, truncation = 1000, 1.5, 1.0, 20000
        cover, att = 8000, 2000
        expected = gen_pareto_layer_mean(
            cover, att, t=t, alpha_ini=alpha_ini, alpha_tail=alpha_tail, truncation=truncation
        )
        assert expected == pytest.approx(932.32300743380154, rel=1e-6)
        losses = r_gen_pareto(n, t, alpha_ini, alpha_tail, truncation=truncation, rng=rng)
        xs = np.minimum(cover, np.maximum(0.0, losses - att))
        ratio = round(xs.mean() / expected, 2)
        assert ratio == 1.0

    def test_without_truncation(self) -> None:
        """Simulated layer mean should match analytical value (ratio ~1)."""
        rng = np.random.default_rng(1972)
        n = 1_000_000
        t, alpha_ini, alpha_tail = 1000, 1.5, 3.0
        cover, att = 8000, 2000
        expected = gen_pareto_layer_mean(
            cover, att, t=t, alpha_ini=alpha_ini, alpha_tail=alpha_tail
        )
        assert expected == pytest.approx(411.38659320477501, rel=1e-6)
        losses = r_gen_pareto(n, t, alpha_ini, alpha_tail, rng=rng)
        xs = np.minimum(cover, np.maximum(0.0, losses - att))
        ratio = round(xs.mean() / expected, 2)
        assert ratio == 1.0


class TestGenParetoMLEstimatorAlpha:
    LOSSES: ClassVar[list[float]] = [
        1622.49986584698,
        1025.1735923535,
        1142.67198754259,
        1598.2131674777,
        1369.79742768744,
        1006.5249344124,
        2019.3663238659,
        1007.2758879241,
        1377.79293040511,
        1605.21438984656,
        2579.4568112321,
        4500.45681,
    ]

    def test_basic(self) -> None:
        result = gen_pareto_ml_estimator_alpha(self.LOSSES, 1000)
        np.testing.assert_allclose(result, [2.1210190911501012, 2.5019159656778251], rtol=1e-5)

    def test_with_truncation(self) -> None:
        result = gen_pareto_ml_estimator_alpha(self.LOSSES, 1000, truncation=10000)
        np.testing.assert_allclose(result, [2.3410152490683958, 1.6923583375172637], rtol=1e-5)

    def test_with_reporting_thresholds(self) -> None:
        rt = [1000] * 12
        rt[0] = 1500
        result = gen_pareto_ml_estimator_alpha(self.LOSSES, 1000, reporting_thresholds=rt)
        np.testing.assert_allclose(result, [2.54341, 2.15354], atol=5e-5)

    def test_with_reporting_thresholds_and_truncation(self) -> None:
        rt = [1000] * 12
        rt[0] = 1500
        result = gen_pareto_ml_estimator_alpha(
            self.LOSSES, 1000, truncation=10000, reporting_thresholds=rt
        )
        np.testing.assert_allclose(result, [2.91339, 1.41459], atol=5e-5)

    def test_with_censored(self) -> None:
        rt = [1000] * 12
        rt[0] = 1500
        is_censored = [False] * 12
        is_censored[0] = True
        is_censored[1] = True
        result = gen_pareto_ml_estimator_alpha(
            self.LOSSES, 1000, reporting_thresholds=rt, is_censored=is_censored
        )
        np.testing.assert_allclose(result, [1.73377, 2.90203], atol=5e-5)

    def test_with_censored_no_rt(self) -> None:
        is_censored = [False] * 12
        is_censored[0] = True
        is_censored[1] = True
        result = gen_pareto_ml_estimator_alpha(self.LOSSES, 1000, is_censored=is_censored)
        np.testing.assert_allclose(result, [1.47613, 3.68474], atol=5e-5)

    def test_with_all_options(self) -> None:
        rt = [1000] * 12
        rt[0] = 1500
        is_censored = [False] * 12
        is_censored[0] = True
        is_censored[1] = True
        result = gen_pareto_ml_estimator_alpha(
            self.LOSSES,
            1000,
            truncation=10000,
            reporting_thresholds=rt,
            is_censored=is_censored,
        )
        np.testing.assert_allclose(result, [1.88626, 1.93255], atol=5e-5)

    def test_weights_equivalent_to_repeated_losses(self) -> None:
        losses = np.array(self.LOSSES)
        w = np.ones(len(losses))
        idx = [0, 3, 5]
        w[idx] = 2
        losses2 = np.concatenate([losses, losses[idx]])
        r1 = gen_pareto_ml_estimator_alpha(losses, 1000, weights=w)
        r2 = gen_pareto_ml_estimator_alpha(losses2, 1000)
        np.testing.assert_allclose(r1, r2, rtol=1e-5)

    def test_weights_with_truncation(self) -> None:
        losses = np.array(self.LOSSES)
        w = np.ones(len(losses))
        idx = [0, 3, 5]
        w[idx] = 2
        losses2 = np.concatenate([losses, losses[idx]])
        r1 = gen_pareto_ml_estimator_alpha(losses, 1000, weights=w, truncation=10000)
        r2 = gen_pareto_ml_estimator_alpha(losses2, 1000, truncation=10000)
        np.testing.assert_allclose(r1, r2, rtol=1e-5)
