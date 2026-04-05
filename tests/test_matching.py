"""Tests for matching module — task 11a: LP engine."""

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
