"""Pyreto: Pareto, piecewise Pareto and generalized Pareto tools for reinsurance pricing.

Port of the R Pareto package by Ulrich Riegel (GPL >= 2).
See https://github.com/ulrichriegel/Pareto for the original.
"""

__version__ = "0.1.0"

from pyreto.gen_pareto import (
    d_gen_pareto,
    gen_pareto_layer_mean,
    gen_pareto_layer_sm,
    gen_pareto_layer_var,
    gen_pareto_ml_estimator_alpha,
    p_gen_pareto,
    q_gen_pareto,
    r_gen_pareto,
)
from pyreto.pareto import (
    d_pareto,
    p_pareto,
    pareto_extrapolation,
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_fqs,
    pareto_find_alpha_btw_layers,
    pareto_layer_mean,
    pareto_layer_sm,
    pareto_layer_var,
    pareto_ml_estimator_alpha,
    q_pareto,
    r_pareto,
)
from pyreto.piecewise_pareto import (
    d_piecewise_pareto,
    p_piecewise_pareto,
    piecewise_pareto_layer_mean,
    piecewise_pareto_layer_sm,
    piecewise_pareto_layer_var,
    piecewise_pareto_ml_estimator_alpha,
    q_piecewise_pareto,
    r_piecewise_pareto,
)

__all__ = [
    "d_gen_pareto",
    "d_pareto",
    "d_piecewise_pareto",
    "gen_pareto_layer_mean",
    "gen_pareto_layer_sm",
    "gen_pareto_layer_var",
    "gen_pareto_ml_estimator_alpha",
    "p_gen_pareto",
    "p_pareto",
    "p_piecewise_pareto",
    "pareto_extrapolation",
    "pareto_find_alpha_btw_fq_layer",
    "pareto_find_alpha_btw_fqs",
    "pareto_find_alpha_btw_layers",
    "pareto_layer_mean",
    "pareto_layer_sm",
    "pareto_layer_var",
    "pareto_ml_estimator_alpha",
    "piecewise_pareto_layer_mean",
    "piecewise_pareto_layer_sm",
    "piecewise_pareto_layer_var",
    "piecewise_pareto_ml_estimator_alpha",
    "q_gen_pareto",
    "q_pareto",
    "q_piecewise_pareto",
    "r_gen_pareto",
    "r_pareto",
    "r_piecewise_pareto",
]
