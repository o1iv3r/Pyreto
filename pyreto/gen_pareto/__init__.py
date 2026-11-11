"""Generalized Pareto distribution functions for reinsurance pricing.

Mirrors the GenPareto section of the R Pareto package (Functions.R, lines 3381-4261).
All public functions accept scalar or array inputs via np.vectorize dispatch.

The generalized Pareto distribution is parameterised by (t, alpha_ini, alpha_tail)
and has survival function:

    S(x) = (1 + alpha_ini/alpha_tail * (x/t - 1))^(-alpha_tail)   for x > t

When alpha_ini == alpha_tail the distribution reduces to the standard Pareto(t, alpha).

R -> Python name mapping (see docs/function_mapping.md):
  pGenPareto                    -> p_gen_pareto
  dGenPareto                    -> d_gen_pareto
  qGenPareto                    -> q_gen_pareto
  rGenPareto                    -> r_gen_pareto
  GenPareto_Layer_Mean          -> gen_pareto_layer_mean
  GenPareto_Layer_SM            -> gen_pareto_layer_sm
  GenPareto_Layer_Var           -> gen_pareto_layer_var
  GenPareto_ML_Estimator_Alpha  -> gen_pareto_ml_estimator_alpha
"""

from pyreto.gen_pareto.distribution import (
    d_gen_pareto,
    p_gen_pareto,
    q_gen_pareto,
    r_gen_pareto,
)
from pyreto.gen_pareto.estimation import (
    gen_pareto_ml_estimator_alpha,
)
from pyreto.gen_pareto.layer import (
    gen_pareto_layer_mean,
    gen_pareto_layer_sm,
    gen_pareto_layer_var,
)

__all__ = [
    "d_gen_pareto",
    "gen_pareto_layer_mean",
    "gen_pareto_layer_sm",
    "gen_pareto_layer_var",
    "gen_pareto_ml_estimator_alpha",
    "p_gen_pareto",
    "q_gen_pareto",
    "r_gen_pareto",
]
