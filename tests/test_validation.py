"""Tests for internal validation helpers."""

import numpy as np

from pyreto._validation import (
    is_nonnegative_finite_number,
    is_nonnegative_finite_vector,
    is_positive_finite_number,
    is_positive_vector,
    valid_parameters_pareto,
    valid_parameters_piecewise_pareto,
)


def test_is_positive_vector() -> None:
    assert is_positive_vector(np.array([1.0, 2.0]))
    assert not is_positive_vector(np.array([0.0, 1.0]))
    assert not is_positive_vector(np.array([-1.0, 2.0]))
    assert not is_positive_vector(np.array([np.nan, 1.0]))


def test_is_nonnegative_finite_vector() -> None:
    assert is_nonnegative_finite_vector(np.array([0.0, 1.0]))
    assert not is_nonnegative_finite_vector(np.array([-1.0, 1.0]))
    assert not is_nonnegative_finite_vector(np.array([np.inf, 1.0]))


def test_is_positive_finite_number() -> None:
    assert is_positive_finite_number(1.0)
    assert not is_positive_finite_number(0.0)
    assert not is_positive_finite_number(np.inf)
    assert not is_positive_finite_number(-1.0)


def test_is_nonnegative_finite_number() -> None:
    assert is_nonnegative_finite_number(0.0)
    assert is_nonnegative_finite_number(1.0)
    assert not is_nonnegative_finite_number(-1.0)
    assert not is_nonnegative_finite_number(np.inf)


def test_valid_parameters_pareto() -> None:
    assert valid_parameters_pareto(t=1000.0, alpha=2.0, truncation=None)
    assert not valid_parameters_pareto(t=0.0, alpha=2.0, truncation=None)
    assert not valid_parameters_pareto(t=1000.0, alpha=-1.0, truncation=None)
    assert valid_parameters_pareto(t=1000.0, alpha=2.0, truncation=5000.0)
    assert not valid_parameters_pareto(t=1000.0, alpha=2.0, truncation=500.0)  # truncation <= t


def test_valid_parameters_piecewise_pareto() -> None:
    # interior alpha=0 is allowed; last alpha must be > 0
    assert valid_parameters_piecewise_pareto(
        t=[1000, 2000], alpha=[0, 1.5], truncation=None, truncation_type="lp"
    )
    assert not valid_parameters_piecewise_pareto(
        t=[1000, 2000], alpha=[1, 0], truncation=None, truncation_type="lp"
    )  # last alpha is zero
    # truncation must exceed max(t), not just t[0]
    assert not valid_parameters_piecewise_pareto(
        t=[1000, 5000], alpha=[1, 2], truncation=3000, truncation_type="lp"
    )  # 3000 < max(t)=5000
    assert valid_parameters_piecewise_pareto(
        t=[1000, 5000], alpha=[1, 2], truncation=10000, truncation_type="lp"
    )
