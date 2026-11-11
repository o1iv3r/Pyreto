"""Pareto distribution functions for reinsurance pricing.

Mirrors the Pareto section of the R Pareto package (Functions.R).
All functions accept scalar or array inputs (numpy broadcasting).

R -> Python name mapping (see docs/function_mapping.md):
  pPareto                         -> p_pareto
  dPareto                         -> d_pareto
  qPareto                         -> q_pareto
  rPareto                         -> r_pareto
  Pareto_Layer_Mean               -> pareto_layer_mean
  Pareto_Layer_SM                 -> pareto_layer_sm
  Pareto_Layer_Var                -> pareto_layer_var
  Pareto_Extrapolation            -> pareto_extrapolation
  Pareto_Find_Alpha_btw_Layers    -> pareto_find_alpha_btw_layers
  Pareto_Find_Alpha_btw_FQ_Layer  -> pareto_find_alpha_btw_fq_layer
  Pareto_Find_Alpha_btw_FQs       -> pareto_find_alpha_btw_fqs
  Pareto_ML_Estimator_Alpha       -> pareto_ml_estimator_alpha
"""

from pyreto.pareto.distribution import (
    d_pareto,
    p_pareto,
    q_pareto,
    r_pareto,
)
from pyreto.pareto.estimation import (
    pareto_extrapolation,
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_fqs,
    pareto_find_alpha_btw_layers,
    pareto_ml_estimator_alpha,
)
from pyreto.pareto.layer import (
    pareto_layer_mean,
    pareto_layer_sm,
    pareto_layer_var,
)

__all__ = [
    "d_pareto",
    "p_pareto",
    "pareto_extrapolation",
    "pareto_find_alpha_btw_fq_layer",
    "pareto_find_alpha_btw_fqs",
    "pareto_find_alpha_btw_layers",
    "pareto_layer_mean",
    "pareto_layer_sm",
    "pareto_layer_var",
    "pareto_ml_estimator_alpha",
    "q_pareto",
    "r_pareto",
]
