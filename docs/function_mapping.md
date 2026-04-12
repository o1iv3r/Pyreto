# R to Python Function Mapping

| R function | Python function | Module | Notes |
|---|---|---|---|
| `pPareto` | `p_pareto` | `pyreto.pareto` | |
| `dPareto` | `d_pareto` | `pyreto.pareto` | |
| `qPareto` | `q_pareto` | `pyreto.pareto` | |
| `rPareto` | `r_pareto` | `pyreto.pareto` | |
| `Pareto_CDF` | — | — | Deprecated alias; omitted |
| `Pareto_PDF` | — | — | Deprecated alias; omitted |
| `Pareto_Layer_Mean` | `pareto_layer_mean` | `pyreto.pareto` | |
| `Pareto_Layer_SM` | `pareto_layer_sm` | `pyreto.pareto` | |
| `Pareto_Layer_Var` | `pareto_layer_var` | `pyreto.pareto` | |
| `Pareto_Extrapolation` | `pareto_extrapolation` | `pyreto.pareto` | |
| `Pareto_Find_Alpha_btw_Layers` | `pareto_find_alpha_btw_layers` | `pyreto.pareto` | |
| `Pareto_Find_Alpha_btw_FQ_Layer` | `pareto_find_alpha_btw_fq_layer` | `pyreto.pareto` | |
| `Pareto_Find_Alpha_btw_FQs` | `pareto_find_alpha_btw_fqs` | `pyreto.pareto` | |
| `Pareto_ML_Estimator_Alpha` | `pareto_ml_estimator_alpha` | `pyreto.pareto` | |
| `pPiecewisePareto` | `p_piecewise_pareto` | `pyreto.piecewise_pareto` | |
| `dPiecewisePareto` | `d_piecewise_pareto` | `pyreto.piecewise_pareto` | |
| `qPiecewisePareto` | `q_piecewise_pareto` | `pyreto.piecewise_pareto` | |
| `rPiecewisePareto` | `r_piecewise_pareto` | `pyreto.piecewise_pareto` | |
| `PiecewisePareto_CDF` | — | — | Deprecated alias; omitted |
| `PiecewisePareto_PDF` | — | — | Deprecated alias; omitted |
| `PiecewisePareto_Layer_Mean` | `piecewise_pareto_layer_mean` | `pyreto.piecewise_pareto` | |
| `PiecewisePareto_Layer_SM` | `piecewise_pareto_layer_sm` | `pyreto.piecewise_pareto` | |
| `PiecewisePareto_Layer_Var` | `piecewise_pareto_layer_var` | `pyreto.piecewise_pareto` | |
| `PiecewisePareto_ML_Estimator_Alpha` | `piecewise_pareto_ml_estimator_alpha` | `pyreto.piecewise_pareto` | |
| `PiecewisePareto_Match_Layer_Losses` | `piecewise_pareto_match_layer_losses` | `pyreto.matching` | |
| `pGenPareto` | `p_gen_pareto` | `pyreto.gen_pareto` | |
| `dGenPareto` | `d_gen_pareto` | `pyreto.gen_pareto` | |
| `qGenPareto` | `q_gen_pareto` | `pyreto.gen_pareto` | |
| `rGenPareto` | `r_gen_pareto` | `pyreto.gen_pareto` | |
| `GenPareto_Layer_Mean` | `gen_pareto_layer_mean` | `pyreto.gen_pareto` | |
| `GenPareto_Layer_SM` | `gen_pareto_layer_sm` | `pyreto.gen_pareto` | |
| `GenPareto_Layer_Var` | `gen_pareto_layer_var` | `pyreto.gen_pareto` | |
| `GenPareto_ML_Estimator_Alpha` | `gen_pareto_ml_estimator_alpha` | `pyreto.gen_pareto` | |
| `PPP_Model` | `PPPModel` | `pyreto.ppp_model` | Constructor becomes dataclass |
| `is.valid.PPP_Model` | `PPPModel.is_valid()` | `pyreto.ppp_model` | |
| `PPP_Model_Exp_Layer_Loss` | `layer_mean(model, ...)` | `pyreto.collective_model` | Dispatch function |
| `PPP_Model_Layer_Var` | `layer_var(model, ...)` | `pyreto.collective_model` | Dispatch function |
| `PPP_Model_Layer_Sd` | `layer_sd(model, ...)` | `pyreto.collective_model` | Dispatch function |
| `PPP_Model_Excess_Frequency` | `excess_frequency(model, ...)` | `pyreto.collective_model` | Dispatch function |
| `PPP_Model_Simulate` | `simulate_losses(model, ...)` | `pyreto.collective_model` | Dispatch function |
| `PGP_Model` | `PGPModel` | `pyreto.pgp_model` | Constructor becomes dataclass |
| `is.valid.PGP_Model` | `PGPModel.is_valid()` | `pyreto.pgp_model` | |
| `Layer_Mean` | `layer_mean` | `pyreto.collective_model` | Dispatches to PPPModel/PGPModel |
| `Layer_Var` | `layer_var` | `pyreto.collective_model` | |
| `Layer_Sd` | `layer_sd` | `pyreto.collective_model` | |
| `Excess_Frequency` | `excess_frequency` | `pyreto.collective_model` | |
| `Simulate_Losses` | `simulate_losses` | `pyreto.collective_model` | |
| `dPanjer` | `d_panjer` | `pyreto.fitting` | |
| `rPanjer` | `r_panjer` | `pyreto.fitting` | |
| `Local_Pareto_Alpha` | `local_pareto_alpha` | `pyreto.fitting` | |
| `Fit_References` | `fit_references` | `pyreto.fitting` | |
| `Fit_PML_Curve` | `fit_pml_curve` | `pyreto.fitting` | |
