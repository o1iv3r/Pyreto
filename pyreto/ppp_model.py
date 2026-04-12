"""PPP_Model (Collective Panjer & Piecewise Pareto Model).

Port of PPPModel.R from the R Pareto package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyreto._validation import (
    is_nonnegative_finite_number,
    is_positive_finite_number,
    valid_parameters_piecewise_pareto,
)


@dataclass
class PPPModel:
    """Collective Panjer & Piecewise Pareto Model.

    Parameters
    ----------
    fq:
        Expected claim count (non-negative finite number).
    t:
        Thresholds of the piecewise-Pareto severity distribution (strictly
        ascending positive vector).
    alpha:
        Pareto alpha for each segment (same length as ``t``, all non-negative;
        last entry must be strictly positive).
    truncation:
        Upper truncation point, or ``None`` for no truncation.
    truncation_type:
        ``"lp"`` (truncated last piece) or ``"wd"`` (whole distribution).
    dispersion:
        Variance-to-mean ratio of the claim-count distribution (positive
        finite number; 1.0 = Poisson).
    status:
        0 = success, 1 = some information was ignored, 2 = no solution.
    comment:
        Human-readable comment returned by fitting functions.
    """

    fq: float
    t: np.ndarray
    alpha: np.ndarray
    truncation: float | None = None
    truncation_type: str = "lp"
    dispersion: float = 1.0
    status: int = 0
    comment: str = "OK"

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.alpha = np.asarray(self.alpha, dtype=float)

    def is_valid(self) -> bool:
        """Return True if the model parameters are consistent and usable."""
        if not is_nonnegative_finite_number(self.fq):
            return False
        if not is_positive_finite_number(self.dispersion):
            return False
        return valid_parameters_piecewise_pareto(
            self.t,
            self.alpha,
            self.truncation,
            self.truncation_type,
        )

    def __repr__(self) -> str:
        trunc_str = f", truncation={self.truncation}" if self.truncation is not None else ""
        return (
            f"PPPModel(fq={self.fq}, t={self.t.tolist()}, alpha={self.alpha.tolist()}{trunc_str})"
        )
