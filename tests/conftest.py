"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def pareto_losses() -> np.ndarray:
    """Reference loss sample from R test suite."""
    return np.array(
        [
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
        ]
    )
