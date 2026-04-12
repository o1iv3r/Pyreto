"""Tests for pyreto.piecewise_pareto — ported from test_functions_PiecewisePareto.R."""

import numpy as np
import pytest

from pyreto.piecewise_pareto import (
    d_piecewise_pareto,
    p_piecewise_pareto,
    piecewise_pareto_layer_mean,
    piecewise_pareto_layer_sm,
    piecewise_pareto_layer_var,
    piecewise_pareto_ml_estimator_alpha,
    q_piecewise_pareto,
    r_piecewise_pareto,
)

# Shared losses for ML estimator tests (ported from R test fixture)
_ML_LOSSES = np.array(
    [
        2021.37427270114,
        1144.4154953111,
        1263.31435215882,
        2159.17623012884,
        1597.37248664761,
        1272.28453257,
        2914.10585100956,
        1304.18266162408,
        1120.50113120055,
        2127.70698437685,
        2060.81683374026,
        2039.7843971136,
        1888.54098901941,
        2124.60113577822,
        1861.90023645773,
        1061.42587835879,
        1096.87195504782,
        2741.00486150546,
        3599.51085744179,
        1944.00791988792,
        4801.21672397748,
        2011.17712322692,
        1614.38016727125,
        1366.00934974079,
        2740.17455761229,
        3696.80509791037,
        1519.32575139179,
        2673.72771069715,
        1377.97530890359,
        2747.57267315656,
        2962.51959921929,
        2970.06262726659,
        3849.91766409146,
        1011.07955688529,
        2142.35280378138,
        2363.88639730512,
        1585.05201649176,
        1519.87964078085,
        4152.85247579578,
        1145.85174532018,
        1045.70673772088,
        2035.21218839728,
        2265.69458584306,
        1259.17343777875,
        1276.43540240796,
        1156.59174646013,
        2213.19952527109,
        2090.8925597549,
        1599.32662288134,
        1748.10607173994,
        1110.83766088527,
        4216.9888129572,
        1987.10180900716,
        2848.75495497778,
        2459.33207487438,
        1057.30210620244,
        2613.23427141216,
        4267.61547888255,
        2110.3988745262,
        1608.79852597768,
        3299.23731214781,
        1598.18841764305,
        1867.71915473237,
        1025.8298666397,
        1714.23234124254,
        2344.71364141762,
        1415.68319510445,
        1529.97435270215,
        1236.57055448752,
        2466.0269323041,
        1091.02769697244,
        2408.02293167307,
        3106.53753930817,
        2461.21276690245,
        3262.03972168424,
        4127.01858195009,
        1181.45647693256,
        2915.05440240728,
        2819.40842718265,
        1079.96118940952,
        3652.61964913811,
        3731.17802733447,
        4689.68469988139,
        3430.45722285605,
        1700.45433736218,
        2269.02625009652,
        3545.91823730921,
        3112.72960104325,
        2757.85209289505,
        1073.92628882837,
        3697.84111140716,
        2567.88718258095,
        2624.17812649749,
        2430.48340413282,
        2399.39130252127,
        1194.87342560877,
        1733.25071770226,
        2177.58393770677,
        2313.84456911248,
        2561.30080556934,
    ]
)


class TestPPiecewisePareto:
    def test_basic(self) -> None:
        result = p_piecewise_pareto(
            np.array([1000, 2000, 3000, 4000]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=4000,
        )
        np.testing.assert_allclose(result, [0, 0.5, 0.77777777777777779, 1])


class TestDPiecewisePareto:
    def test_basic(self) -> None:
        result = d_piecewise_pareto(
            np.array([1000, 2000, 3000, 4000]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=4000,
        )
        np.testing.assert_allclose(result, [0.001, 0.0005, 0.00022222222222222221, 0])


class TestQPiecewisePareto:
    def test_basic(self) -> None:
        result = q_piecewise_pareto(
            np.arange(1, 10) * 0.1,
            t=np.array([1000, 2000]),
            alpha=np.array([1, 2]),
        )
        np.testing.assert_allclose(
            result,
            [
                1111.1111111111111,
                1250.0,
                1428.5714285714287,
                1666.6666666666667,
                2000.0,
                2236.0679774997902,
                2581.9888974716114,
                3162.2776601683800,
                4472.1359549995805,
            ],
            rtol=1e-10,
        )

    def test_with_truncation(self) -> None:
        result = q_piecewise_pareto(
            np.arange(1, 5) * 0.2,
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=4000,
        )
        np.testing.assert_allclose(
            result,
            [1250.0, 1666.6666666666667, 2236.0679774997902, 3060.1459631738917],
            rtol=1e-10,
        )


class TestRPiecewisePareto:
    def test_simulation_parity_lp(self) -> None:
        rng = np.random.default_rng(1972)
        t = np.array([1000, 3000, 5000])
        alpha = np.array([0.7, 1.5, 2.0])
        losses = r_piecewise_pareto(
            1_000_000, t, alpha, truncation=6000, truncation_type="lp", rng=rng
        )
        xs = np.minimum(8000, np.maximum(0, losses - 2000))
        ref = piecewise_pareto_layer_mean(
            8000, 2000, t, alpha, truncation=6000, truncation_type="lp"
        )
        assert round(float(xs.mean()) / float(ref), 2) == 1.0

    def test_simulation_parity_wd(self) -> None:
        rng = np.random.default_rng(1972)
        t = np.array([1000, 3000, 5000])
        alpha = np.array([0.7, 1.5, 2.0])
        losses = r_piecewise_pareto(
            1_000_000, t, alpha, truncation=6000, truncation_type="wd", rng=rng
        )
        xs = np.minimum(8000, np.maximum(0, losses - 2000))
        ref = piecewise_pareto_layer_mean(
            8000, 2000, t, alpha, truncation=6000, truncation_type="wd"
        )
        assert round(float(xs.mean()) / float(ref), 2) == 1.0

    def test_simulation_parity_no_truncation(self) -> None:
        rng = np.random.default_rng(1972)
        t = np.array([1000, 3000, 5000])
        alpha = np.array([0.7, 1.5, 2.0])
        losses = r_piecewise_pareto(1_000_000, t, alpha, rng=rng)
        xs = np.minimum(8000, np.maximum(0, losses - 2000))
        ref = piecewise_pareto_layer_mean(8000, 2000, t, alpha)
        assert round(float(xs.mean()) / float(ref), 2) == 1.0


class TestPiecewisePareto_LayerMean:  # noqa: N801
    def test_basic(self) -> None:
        assert piecewise_pareto_layer_mean(8000, 2000, t=1000, alpha=2) == pytest.approx(400)

    def test_with_truncation(self) -> None:
        assert piecewise_pareto_layer_mean(
            8000, 2000, t=5000, alpha=2, truncation=10000
        ) == pytest.approx(4666.66666666667)

    def test_vectorised_covers_attachments(self) -> None:
        result = piecewise_pareto_layer_mean(
            np.array([8000, 2000]),
            np.array([2000, 1000]),
            t=5000,
            alpha=2,
            truncation=10000,
        )
        np.testing.assert_allclose(result, [4666.66666666667, 2000])

    def test_vectorised_piecewise(self) -> None:
        result = piecewise_pareto_layer_mean(
            np.array([8000, 2000]),
            np.array([2000, 1000]),
            t=np.array([1000, 3000, 5000]),
            alpha=np.array([1, 2, 3]),
            truncation=10000,
        )
        np.testing.assert_allclose(result, [976.89367953673514, 1098.61228866810916], rtol=1e-10)


class TestPiecewisePareto_LayerSM:  # noqa: N801
    def test_basic(self) -> None:
        assert piecewise_pareto_layer_sm(8000, 2000, t=1000, alpha=2) == pytest.approx(
            1618875.8248682
        )

    def test_with_truncation(self) -> None:
        assert piecewise_pareto_layer_sm(
            8000, 2000, t=5000, alpha=2, truncation=10000
        ) == pytest.approx(23543145.370663)

    def test_vectorised_piecewise(self) -> None:
        result = piecewise_pareto_layer_sm(
            np.array([1000, 8000]),
            np.array([1000, 2000]),
            t=np.array([1000, 3000, 5000]),
            alpha=np.array([1, 2, 3]),
            truncation=10000,
        )
        np.testing.assert_allclose(result, [613705.63888010941, 3300236.16730614332], rtol=1e-7)


class TestPiecewisePareto_LayerVar:  # noqa: N801
    def test_basic(self) -> None:
        assert piecewise_pareto_layer_var(8000, 2000, 2, t=1000) == pytest.approx(1458875.8248682)

    def test_with_truncation(self) -> None:
        assert piecewise_pareto_layer_var(8000, 2000, 2, t=5000, truncation=10000) == pytest.approx(
            1765367.59288524
        )

    def test_vectorised_piecewise(self) -> None:
        result = piecewise_pareto_layer_var(
            np.array([1000, 8000]),
            np.array([1000, 2000]),
            t=np.array([1000, 3000, 5000]),
            alpha=np.array([1, 2, 3]),
            truncation=10000,
        )
        np.testing.assert_allclose(result, [133252.62496190792, 2345914.90618732199], rtol=1e-7)


class TestPiecewisePareto_MLEstimator:  # noqa: N801
    def test_basic(self) -> None:
        result = piecewise_pareto_ml_estimator_alpha(_ML_LOSSES, np.array([1000, 2000, 3000]))
        np.testing.assert_allclose(np.round(result, 5), [0.81259, 2.65715, 4.42204])

    def test_with_lp_truncation(self) -> None:
        result = piecewise_pareto_ml_estimator_alpha(
            _ML_LOSSES,
            np.array([1000, 2000, 3000]),
            truncation=5000,
            truncation_type="lp",
        )
        np.testing.assert_allclose(np.round(result, 5), [0.81259, 2.65715, 1.35691])

    def test_with_wd_truncation(self) -> None:
        result = piecewise_pareto_ml_estimator_alpha(
            _ML_LOSSES,
            np.array([1000, 2000, 3000]),
            truncation=5000,
            truncation_type="wd",
        )
        # R reference: [0.59649, 1.50625, 0.98904]; atol covers optimizer implementation noise
        np.testing.assert_allclose(result, [0.59649, 1.50625, 0.98904], atol=1e-4)

    def test_with_reporting_thresholds(self) -> None:
        rt = np.full(len(_ML_LOSSES), 1000.0)
        rt[0] = 1500.0
        rt[3] = 1500.0
        result = piecewise_pareto_ml_estimator_alpha(
            _ML_LOSSES, np.array([1000, 2000, 3000]), reporting_thresholds=rt
        )
        np.testing.assert_allclose(np.round(result, 5), [0.82524, 2.65715, 4.42204])

    def test_with_censoring(self) -> None:
        rt = np.full(len(_ML_LOSSES), 1000.0)
        rt[0] = 1500.0
        rt[3] = 1500.0
        censored = np.zeros(len(_ML_LOSSES), dtype=bool)
        censored[:3] = True
        result = piecewise_pareto_ml_estimator_alpha(
            _ML_LOSSES,
            np.array([1000, 2000, 3000]),
            reporting_thresholds=rt,
            is_censored=censored,
        )
        np.testing.assert_allclose(np.round(result, 5), [0.78686, 2.58902, 4.42204])

    def test_with_censoring_lp_truncation(self) -> None:
        rt = np.full(len(_ML_LOSSES), 1000.0)
        rt[0] = 1500.0
        rt[3] = 1500.0
        censored = np.zeros(len(_ML_LOSSES), dtype=bool)
        censored[:3] = True
        result = piecewise_pareto_ml_estimator_alpha(
            _ML_LOSSES,
            np.array([1000, 2000, 3000]),
            reporting_thresholds=rt,
            is_censored=censored,
            truncation=5000,
            truncation_type="lp",
        )
        np.testing.assert_allclose(np.round(result, 5), [0.78686, 2.58902, 1.35691])

    def test_with_censoring_wd_truncation(self) -> None:
        rt = np.full(len(_ML_LOSSES), 1000.0)
        rt[0] = 1500.0
        rt[3] = 1500.0
        censored = np.zeros(len(_ML_LOSSES), dtype=bool)
        censored[:3] = True
        result = piecewise_pareto_ml_estimator_alpha(
            _ML_LOSSES,
            np.array([1000, 2000, 3000]),
            reporting_thresholds=rt,
            is_censored=censored,
            truncation=5000,
            truncation_type="wd",
        )
        # R reference: [0.51603, 1.24380, 0.74810]; atol covers optimizer implementation noise
        np.testing.assert_allclose(result, [0.51603, 1.24380, 0.74810], atol=1e-4)

    def test_weights_equivalent_to_duplicates(self) -> None:
        w = np.ones(len(_ML_LOSSES))
        w[0] = 2.0
        w[1] = 2.0
        w[2] = 5.0
        losses2 = np.concatenate([_ML_LOSSES, _ML_LOSSES[:2], np.repeat(_ML_LOSSES[2], 4)])
        result_w = piecewise_pareto_ml_estimator_alpha(
            _ML_LOSSES, np.array([1000, 2000, 3000]), weights=w
        )
        result_dup = piecewise_pareto_ml_estimator_alpha(losses2, np.array([1000, 2000, 3000]))
        np.testing.assert_allclose(result_w, result_dup)
