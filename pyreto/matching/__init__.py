"""LP matching engine and alpha-fitting for piecewise-Pareto layer-loss matching.

Port of lp_functions.R, FitPP.R, and Functions.R from the R Pareto package.

Modules:
  lp      -- _solve_lp: split overlapping reference layers into a tower (LP)
  tower   -- _calculate_layer_losses: fill gaps in the tower
  fit_pp  -- _calculate_taus, _calculate_alphas, _fit_pp (FitPP.R)
  api     -- piecewise_pareto_match_layer_losses (public entry point)
"""

from pyreto.matching.api import (
    piecewise_pareto_match_layer_losses,
)
from pyreto.matching.fit_pp import (
    _calculate_alphas as _calculate_alphas,
)
from pyreto.matching.fit_pp import (
    _calculate_taus as _calculate_taus,
)
from pyreto.matching.fit_pp import (
    _fit_pp as _fit_pp,
)
from pyreto.matching.tower import (
    _calculate_layer_losses as _calculate_layer_losses,
)
from pyreto.matching.tower import (
    _solve_lp as _solve_lp,
)

__all__ = [
    "piecewise_pareto_match_layer_losses",
]
