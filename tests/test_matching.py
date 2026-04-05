"""Tests for matching module — tasks 11a and 11b: LP engine and alpha-fitting."""

import numpy as np
import pytest

from pyreto.matching import _solve_lp


class TestSolveLp:
    def test_non_overlapping_consistent(self) -> None:
        # Three adjacent non-overlapping layers with strictly decreasing RoLs.
        # Each input layer maps 1-to-1 onto one sub-layer, so the LP solution
        # is trivially the input ELs.
        att = np.array([1000.0, 2000.0, 3000.0])
        limits = np.array([1000.0, 1000.0, np.inf])
        el = np.array([500.0, 400.0, 300.0])
        result = _solve_lp(att, limits, el)
        assert result.status == 0
        np.testing.assert_allclose(result.exp_loss[:3], [500.0, 400.0, 300.0], rtol=1e-6)

    def test_overlapping_two_layers(self) -> None:
        # Layer A: [1000, 3000), EL=600
        # Layer B: [2000, 4000), EL=350
        # Sub-layers split at 1000, 2000, 3000, 4000.
        # Equality constraints: x0+x1=600, x1+x2=350
        # RoL monotone: x0>=x1>=x2
        # Feasible x1 range: [175, 300]
        #
        # Average of per-sub-layer LP max:
        #   max x0 → x1=175 → (425, 175, 175)
        #   max x1 → x1=300 → (300, 300,  50)
        #   max x2 → x1=175 → (425, 175, 175)
        # Average: (1150/3, 650/3, 400/3)
        att = np.array([1000.0, 2000.0])
        limits = np.array([2000.0, 2000.0])
        el = np.array([600.0, 350.0])
        result = _solve_lp(att, limits, el)
        assert result.status == 0
        np.testing.assert_allclose(
            result.exp_loss[:3],
            [1150 / 3, 650 / 3, 400 / 3],
            rtol=1e-5,
        )

    def test_infeasible_returns_status_1(self) -> None:
        # Layer 1 EL/limit = 400/1000 = 0.4, Layer 2 EL/limit = 500/1000 = 0.5
        # RoL increases → infeasible.
        att = np.array([1000.0, 2000.0])
        limits = np.array([1000.0, 1000.0])
        el = np.array([400.0, 500.0])
        result = _solve_lp(att, limits, el)
        assert result.status == 1

    def test_with_frequency_threshold(self) -> None:
        # Same overlapping setup as test_overlapping_two_layers but with a
        # threshold at 2000 constraining x1/1000 <= 0.25 (i.e. x1 <= 250).
        # New feasible x1 range: [175, 250].
        #
        # max x0 → x1=175 → (425, 175, 175)
        # max x1 → x1=250 → (350, 250, 100)
        # max x2 → x1=175 → (425, 175, 175)
        # Average: (400, 200, 150)
        att = np.array([1000.0, 2000.0])
        limits = np.array([2000.0, 2000.0])
        el = np.array([600.0, 350.0])
        result = _solve_lp(
            att,
            limits,
            el,
            thresholds=np.array([2000.0]),
            threshold_freqs=np.array([0.25]),
        )
        assert result.status == 0
        np.testing.assert_allclose(result.exp_loss[:3], [400.0, 200.0, 150.0], rtol=1e-5)

    def test_tower_attachment_points(self) -> None:
        # Verify that the tower is correctly split by union of all entry/exit points.
        # Layer A: [1000, 2000), Layer B: [1500, 3000)
        # Sub-layer attachment points should be [1000, 1500, 2000, 3000].
        att = np.array([1000.0, 1500.0])
        limits = np.array([1000.0, 1500.0])
        el = np.array([200.0, 250.0])
        result = _solve_lp(att, limits, el)
        np.testing.assert_array_equal(result.att, [1000.0, 1500.0, 2000.0, 3000.0])
        assert len(result.limits) == 4
        assert np.isinf(result.limits[-1])

    def test_frequency_stored_in_result(self) -> None:
        # Frequencies from thresholds are stored in the result tower.
        att = np.array([1000.0, 2000.0])
        limits = np.array([1000.0, np.inf])
        el = np.array([500.0, 300.0])
        result = _solve_lp(
            att,
            limits,
            el,
            thresholds=np.array([2000.0]),
            threshold_freqs=np.array([0.3]),
        )
        assert result.status == 0
        # The sub-layer at att=2000 should have frequency=0.3 stored.
        idx = np.searchsorted(result.att, 2000.0)
        assert result.frequency[idx] == pytest.approx(0.3)


class TestFitPP:
    def test_single_segment(self) -> None:
        from pyreto.matching import _fit_pp

        # 2 attachment points -> 1 finite layer -> closed-form alpha from frequencies
        a = np.array([1000.0, 2000.0])
        s = np.array([100.0, 50.0])  # frequencies (strictly descending)
        el = np.array([100.0])  # n-1 elements: finite layers only
        result = _fit_pp(a, s, el, truncation=None)
        assert result is not None
        assert result.status == "OK"
        assert len(result.t) >= 1
        assert len(result.alpha) == len(result.t)

    def test_result_has_correct_structure(self) -> None:
        from pyreto.matching import _fit_pp

        a = np.array([1000.0, 2000.0, 3000.0])
        s = np.array([1.0, 0.5, 0.25])
        el = np.array([300.0, 150.0])
        result = _fit_pp(a, s, el, truncation=None)
        assert result.status == "OK"
        # t and alpha must be same length
        assert len(result.t) == len(result.alpha)
        # All attachment points in t must be positive
        assert np.all(np.array(result.t) > 0)
        # All alphas must be non-negative
        assert np.all(np.array(result.alpha) >= 0)

    def test_alphas_match_frequencies(self) -> None:
        from pyreto.matching import _fit_pp

        # With s descending (consistent Pareto), the recovered alphas should be
        # a reasonable positive value.
        a = np.array([1000.0, 2000.0, 4000.0])
        s = np.array([2.0, 1.0, 0.5])
        el = np.array([500.0, 600.0])
        result = _fit_pp(a, s, el, truncation=None)
        assert result.status == "OK"
        assert all(al >= 0 for al in result.alpha)

    def test_calculate_taus_returns_ordered_pair(self) -> None:
        from pyreto.matching import _calculate_taus

        s_0, s_1 = 1.0, 0.5
        a_0, a_1 = 1000.0, 2000.0
        exp_loss = 300.0
        tau_l, tau_u = _calculate_taus(s_0, s_1, a_0, a_1, exp_loss)
        assert a_0 <= tau_l <= a_1
        assert a_0 <= tau_u <= a_1
        assert tau_l <= tau_u + 1e-9  # tau_l <= tau_u (or approximately equal)

    def test_calculate_alphas_matches_layer_loss(self) -> None:
        from pyreto.matching import _calculate_alphas

        s_0, s_1 = 1.0, 0.5
        a_0, a_1 = 1000.0, 2000.0
        exp_loss = 300.0
        t = 1500.0
        alpha_0, alpha_1 = _calculate_alphas(s_0, s_1, a_0, a_1, exp_loss, t)
        assert alpha_0 >= 0
        assert alpha_1 >= 0

    def test_with_truncation(self) -> None:
        from pyreto.matching import _fit_pp

        a = np.array([1000.0, 2000.0, 3000.0])
        s = np.array([1.0, 0.5, 0.25])
        el = np.array([300.0, 150.0])
        result = _fit_pp(a, s, el, truncation=10000.0)
        assert result.status == "OK"
        assert len(result.t) == len(result.alpha)

    def test_pure_pareto_alpha2_recovers_single_segment(self) -> None:
        """A pure Pareto(alpha=2) input should collapse to 1 segment after merging."""
        from pyreto.matching import _fit_pp

        # Pareto(alpha=2) with frequency 1.0 at 1000:
        # s[1] = 1.0 * (1000/2000)^2 = 0.25
        # el[0] = LL(1000, 2000, 2) * 1.0 = 500
        # el[1] = LL(2000, inf,  2) * 0.25 = 0.25 * 2000 = 500
        a = np.array([1000.0, 2000.0])
        s = np.array([1.0, 0.25])
        el = np.array([500.0, 500.0])  # n elements (full R convention with infinite tail)
        result = _fit_pp(a, s, el, truncation=None)
        assert result.status == "OK"
        # Should recover alpha ≈ 2 in every segment
        assert all(abs(al - 2.0) < 0.01 for al in result.alpha)
