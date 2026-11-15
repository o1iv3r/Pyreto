# Using Pyreto from a coding agent

This page is written for coding agents (and the people steering them). It covers the
reinsurance-specific part of Pyreto: fitting models to market references or loss data,
pricing XL layers, and working with collective models. The plain single-Pareto
distribution functions (`p_/d_/q_/r_pareto`) are also available, but `scipy.stats.pareto`
covers them equally well.

## Install

> **Do not `pip install pyreto`.** The PyPI package called `pyreto` is a different,
> unrelated library. Install from GitHub:

```bash
uv add "pyreto @ git+https://github.com/o1iv3r/Pyreto"
# or
pip install "pyreto @ git+https://github.com/o1iv3r/Pyreto"
```

Everything public is importable from the top level: `import pyreto`. The package ships
type hints (`py.typed`) and NumPy-style docstrings, so `help(pyreto.fit_references)` is
the authoritative reference for every argument.

## Vocabulary

| Term | Meaning in Pyreto |
|---|---|
| layer `C xs AP` | Pays the part of each loss between `AP` and `AP + C`. `cover=np.inf` means unlimited. |
| expected layer loss | Expected annual loss to the layer (frequency × severity layer mean). |
| excess frequency | Expected number of losses per year above a threshold. |
| `PPPModel` | Collective model: Panjer claim count (`fq`, `dispersion`) × piecewise Pareto severity (`t`, `alpha`). |
| `PGPModel` | Collective model with a generalized Pareto severity (`t`, `alpha_ini`, `alpha_tail`). |
| `dispersion` | Variance/mean of the claim count: 1 = Poisson, > 1 negative binomial, < 1 binomial. |

## Workflow 1: fit a model to references, then price layers

Use `fit_references` when you have expected losses for several (possibly overlapping)
layers and/or excess frequencies. It returns a `PPPModel` (or a `PGPModel` with
`severity_distribution="GenPareto"`).

```python
import numpy as np
import pyreto

model = pyreto.fit_references(
    covers=[1000, 2000, np.inf],
    attachment_points=[1000, 2000, 4000],
    expected_layer_losses=[500, 300, 150],
    thresholds=[1000],
    frequencies=[1.0],
)
assert model.status == 0, model.comment   # see "Failure handling" below

pyreto.layer_mean(model, cover=[1000, 3000], attachment_point=[1500, 2500])  # price new layers
pyreto.layer_sd(model, cover=1000, attachment_point=1500)
pyreto.excess_frequency(model, [1000, 5000])
pyreto.simulate_losses(model, nyears=10_000, seed=1)   # list of per-year loss lists
```

If the references already form a contiguous tower (layer *i* is
`AP[i+1] - AP[i] xs AP[i]`, the last one unlimited), call
`piecewise_pareto_match_layer_losses(attachment_points, expected_layer_losses)` directly.
For a PML curve (return periods and PML amounts) use `fit_pml_curve`; note that it
currently converts the curve with a trapezoidal approximation and warns about it.

## Workflow 2: single-alpha rules of thumb

```python
# Alpha implied by two layers with known expected losses
alpha = pyreto.pareto_find_alpha_btw_layers(1000, 1000, 500, 2000, 3000, 150)

# Extrapolate a burning cost from one layer to another with that alpha
pyreto.pareto_extrapolation(1000, 1000, 2000, 3000, alpha, exp_loss_1=500)

# Alpha implied by an excess frequency and a layer loss, or by two frequencies
pyreto.pareto_find_alpha_btw_fq_layer(1000, 2.0, 1000, 2000, 1000)
pyreto.pareto_find_alpha_btw_fqs(1000, 2.0, 3000, 0.5)
```

## Workflow 3: severity-only layer statistics

Per ground-up loss (no frequency): `pareto_layer_mean/var/sm`,
`piecewise_pareto_layer_mean/var/sm` and `gen_pareto_layer_mean/var/sm`, all taking
`cover, attachment_point` plus the distribution parameters. Use the collective-model
functions (`layer_mean`, `layer_var`, `layer_sd`) when you need annual figures.

## Workflow 4: estimate alphas from individual losses

When you have a loss list rather than layer figures, use the ML estimators. Unlike
`scipy.stats.pareto.fit`, they handle what reinsurance data usually looks like: an upper
`truncation`, per-loss `reporting_thresholds`, censored (still open) claims via
`is_censored`, and `weights`.

```python
alpha = pyreto.pareto_ml_estimator_alpha(
    losses, t=1000, reporting_thresholds=reporting_thresholds, is_censored=is_open,
)
alphas = pyreto.piecewise_pareto_ml_estimator_alpha(losses, t=[1000, 3000])   # one alpha per piece
alpha_ini, alpha_tail = pyreto.gen_pareto_ml_estimator_alpha(losses, t=1000)
```

To see how tail-heavy a parametric severity is across the loss range, use
`local_pareto_alpha(x, "lnorm", meanlog=7, sdlog=1)` (also `"norm"`, `"gamma"`, and
others; see its docstring).

## Workflow 5: build a model directly from parameters

When the parameters come from expert judgment or an earlier estimate, construct the model
yourself and use the same pricing functions as in workflow 1:

```python
model = pyreto.PPPModel(fq=2.0, t=[1000, 5000], alpha=[1.5, 2.5], dispersion=1.5)
gp = pyreto.PGPModel(fq=2.0, t=1000, alpha_ini=1.5, alpha_tail=2.5)
assert model.is_valid() and gp.is_valid()
pyreto.layer_mean(model, cover=4000, attachment_point=1000)
```

The severity distributions themselves are available as `p_/d_/q_/r_` functions for
`piecewise_pareto` and `gen_pareto` (scipy has neither), e.g.
`q_piecewise_pareto([0.5, 0.99], t=[1000, 5000], alpha=[1.5, 2.5])`. Claim counts:
`d_panjer(k, mean=..., dispersion=...)` and `r_panjer(n, mean=..., dispersion=...)`.

## Failure handling

Fitting functions do **not** raise when the references cannot be matched. They return a
model with `status == 2`, the reason in `model.comment`, and emit a `UserWarning`. Layer
functions on such a model return `nan`. Always check:

```python
if model.status == 2:
    raise RuntimeError(model.comment)
if model.status == 1:
    print("Some references were ignored:", model.comment)
```

Inconsistent references (for example a higher layer with a larger loss than a lower layer
of the same width) give status 2; pass `ignore_inconsistent_references=True` to drop them
and fit the rest (status 1).

## Conventions

- Arguments are vectorised: scalars or array-likes, results are `np.ndarray` (or `float`
  for scalar-only functions). `np.inf` is a valid cover.
- Functions mirror the R `Pareto` package with snake_case names; see
  [Function Mapping](function_mapping.md) if you know the R names.
- For reproducible results pass `seed=` to `simulate_losses` and
  `rng=np.random.default_rng(...)` to the `r_*` functions.
