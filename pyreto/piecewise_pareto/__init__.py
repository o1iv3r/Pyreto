"""PiecewisePareto distribution functions for reinsurance pricing.

Mirrors the PiecewisePareto section of the R Pareto package (Functions.R).
All layer-moment functions accept scalar or array inputs for cover/attachment.

R -> Python name mapping (see docs/function_mapping.md):
  pPiecewisePareto                   -> p_piecewise_pareto
  dPiecewisePareto                   -> d_piecewise_pareto
  qPiecewisePareto                   -> q_piecewise_pareto
  rPiecewisePareto                   -> r_piecewise_pareto
  PiecewisePareto_Layer_Mean         -> piecewise_pareto_layer_mean
  PiecewisePareto_Layer_SM           -> piecewise_pareto_layer_sm
  PiecewisePareto_Layer_Var          -> piecewise_pareto_layer_var
  PiecewisePareto_ML_Estimator_Alpha -> piecewise_pareto_ml_estimator_alpha
"""

from pyreto.piecewise_pareto.distribution import (
    d_piecewise_pareto,
    p_piecewise_pareto,
    q_piecewise_pareto,
    r_piecewise_pareto,
)
from pyreto.piecewise_pareto.estimation import (
    piecewise_pareto_ml_estimator_alpha,
)
from pyreto.piecewise_pareto.layer import (
    piecewise_pareto_layer_mean,
    piecewise_pareto_layer_sm,
    piecewise_pareto_layer_var,
)

__all__ = [
    "d_piecewise_pareto",
    "p_piecewise_pareto",
    "piecewise_pareto_layer_mean",
    "piecewise_pareto_layer_sm",
    "piecewise_pareto_layer_var",
    "piecewise_pareto_ml_estimator_alpha",
    "q_piecewise_pareto",
    "r_piecewise_pareto",
]
