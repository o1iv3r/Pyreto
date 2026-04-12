"""PGP_Model (Collective Panjer & Generalized Pareto Model).

Port of PGPModel.R from the R Pareto package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyreto._validation import (
    is_nonnegative_finite_number,
    is_positive_finite_number,
    valid_parameters_gen_pareto,
)


@dataclass
class PGPModel:
    """Collective Panjer & Generalized Pareto Model.

    Parameters
    ----------
    fq:
        Expected claim count (non-negative finite number).
    t:
        Threshold of the generalized Pareto severity distribution (positive
        finite number).
    alpha_ini:
        Initial Pareto alpha (positive number).
    alpha_tail:
        Tail Pareto alpha (positive number).
    truncation:
        Upper truncation point, or ``None`` for no truncation.
    dispersion:
        Variance-to-mean ratio of the claim-count distribution (positive
        finite number; 1.0 = Poisson).
    status:
        0 = success, 1 = some information was ignored, 2 = no solution.
    comment:
        Human-readable comment returned by fitting functions.
    """

    fq: float
    t: float
    alpha_ini: float
    alpha_tail: float
    truncation: float | None = None
    dispersion: float = 1.0
    status: int = 0
    comment: str = "OK"

    def __post_init__(self) -> None:
        self.t = float(self.t)
        self.alpha_ini = float(self.alpha_ini)
        self.alpha_tail = float(self.alpha_tail)

    def is_valid(self) -> bool:
        """Return True if the model parameters are consistent and usable."""
        if not is_nonnegative_finite_number(self.fq):
            return False
        if not is_positive_finite_number(self.dispersion):
            return False
        return valid_parameters_gen_pareto(
            self.t,
            self.alpha_ini,
            self.alpha_tail,
            self.truncation,
        )

    def __repr__(self) -> str:
        trunc_str = f", truncation={self.truncation}" if self.truncation is not None else ""
        return (
            f"PGPModel(fq={self.fq}, t={self.t}, "
            f"alpha_ini={self.alpha_ini}, alpha_tail={self.alpha_tail}{trunc_str})"
        )

    # Expose t as a numpy array for compatibility with piecewise_pareto dispatch
    @property
    def t_arr(self) -> np.ndarray:
        return np.array([self.t])
