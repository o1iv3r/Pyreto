# Pyreto Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the R `Pareto` package to a modern, well-tested Python package (`pyreto`) with feature parity, Pythonic naming, and full documentation.

**Architecture:** The package is split into focused modules mirroring the R source files — one module per distribution family plus separate modules for collective models, the LP matching algorithm, and fitting utilities. R's S3 dispatch (`UseMethod`) becomes a Python Protocol + concrete classes. R's `Vectorize()` becomes numpy broadcasting or `np.vectorize`. R's `lpSolve` is replaced by `scipy.optimize.linprog`; `uniroot`/`optimize` become `scipy.optimize.brentq`/`minimize_scalar`.

**Tech Stack:** Python ≥ 3.11 (tested on 3.11, 3.12, 3.13), uv, pyproject.toml, ruff, ty, pytest, pre-commit, numpy, scipy, numba (optional, for later hot-path optimisation), polars (where tabular data needed), mkdocs-material, GitHub Actions.

**R → Python name mapping:** All exported names are converted to snake_case. Examples: `Pareto_Layer_Mean` → `pareto_layer_mean`, `pPareto` → `p_pareto`, `rPareto` → `r_pareto`, `PPP_Model` (constructor) → `PPPModel` (class), `is.valid.PPP_Model` → `PPPModel.is_valid()`. A full mapping table is documented in `docs/function_mapping.md`.

---

## File Structure

```
pyreto/
    __init__.py                  # public API re-exports
    _validation.py               # internal input validators (not exported)
    pareto.py                    # Pareto distribution: CDF/PDF/quantile/random/layer moments/ML/extrapolation/alpha-finding
    piecewise_pareto.py          # PiecewisePareto: CDF/PDF/quantile/random/layer moments/ML/matching
    gen_pareto.py                # Generalized Pareto: CDF/PDF/quantile/random/layer moments/ML
    fitting.py                   # fit_references, fit_pml_curve, local_pareto_alpha, panjer helpers
    matching.py                  # LP matching engine (FitPP + lp_functions internals)
    ppp_model.py                 # PPPModel dataclass + layer loss/var/sd/freq/simulate
    pgp_model.py                 # PGPModel dataclass + layer loss/var/sd/freq/simulate
    collective_model.py          # CollectiveModel Protocol + layer_mean/var/sd/excess_frequency/simulate_losses dispatch
tests/
    conftest.py                  # shared fixtures (e.g. reference loss arrays)
    test_pareto.py               # ports of test_functions_Pareto.R
    test_piecewise_pareto.py     # ports of test_functions_PiecewisePareto.R
    test_gen_pareto.py           # ports of test_functions_GenPareto.R
    test_ppp_model.py            # ports of test_functions_PPP_Model.R
    test_pgp_model.py            # PGP model tests
    test_collective_model.py     # collective model dispatch tests
    test_fitting.py              # fit_references, fit_pml_curve, local_pareto_alpha
docs/
    index.md                     # project overview
    vignette.md                  # Python translation of Pareto.Rmd vignette
    function_mapping.md          # R → Python name mapping table
    api/
        pareto.md
        piecewise_pareto.md
        gen_pareto.md
        models.md
        fitting.md
.github/
    workflows/
        ci.yml                   # ruff + ty + pytest on push/PR
pyproject.toml
.pre-commit-config.yaml
mkdocs.yml
README.md
```

---

## Chunks

| # | File | Content | Tasks |
|---|---|---|---|
| 1 | [chunk_1_project_scaffolding.md](chunk_1_project_scaffolding.md) | uv init, pyproject.toml, ruff, ty, pre-commit, GitHub Actions CI | 1–2 ✅ |
| 2 | [chunk_2_validation_and_pareto.md](chunk_2_validation_and_pareto.md) | Internal validation helpers, Pareto CDF/PDF/q/r, layer moments, alpha-finding, ML estimator | 3–7 ✅ |
| 3 | [chunk_3_piecewise_pareto.md](chunk_3_piecewise_pareto.md) | PiecewisePareto CDF/PDF/q/r, layer moments, ML estimator | 8–9 ✅ |
| 4 | [chunk_4_gen_pareto.md](chunk_4_gen_pareto.md) | Generalized Pareto CDF/PDF/q/r, layer moments, ML estimator | 10 ✅ |
| 5 | [chunk_5_lp_matching.md](chunk_5_lp_matching.md) | LP engine, alpha-fitting engine, `piecewise_pareto_match_layer_losses` | 11a ✅, 11b–11c |
| 6 | [chunk_6_collective_models.md](chunk_6_collective_models.md) | PPPModel, CollectiveModel protocol, PGPModel | 12–13 |
| 7 | [chunk_7_fitting_and_api.md](chunk_7_fitting_and_api.md) | Panjer, `local_pareto_alpha`, `fit_references`, `fit_pml_curve`, public API, function mapping | 14–15 |
| 8 | [chunk_8_documentation.md](chunk_8_documentation.md) | README, mkdocs, vignette, API reference pages, docs CI | 16–17 |
| 9 | [chunk_9_final_integration.md](chunk_9_final_integration.md) | Full test suite, quality gates, first release (develop → main, v0.1.0 tag) | 18–19 |

---

## Appendix: Key R → Python API Decisions

| Concern | R | Python |
|---|---|---|
| Vectorisation | `Vectorize(f, c("arg1","arg2"))` | `np.vectorize(f)` or natural numpy broadcasting |
| Root finding | `uniroot(f, c(lo, hi))` | `scipy.optimize.brentq(f, lo, hi)` |
| Optimisation | `optimize(f, c(lo, hi))` | `scipy.optimize.minimize_scalar(f, bounds=(lo, hi))` |
| MLE optimisation | `optimize(nll, ...)` | `scipy.optimize.minimize_scalar` / `minimize` |
| Linear programming | `lpSolve::lp("max", obj, mat, dir, rhs)` | `scipy.optimize.linprog(c=-obj, A_ub=A, b_ub=b)` (negate to convert max→min) |
| S3 dispatch | `UseMethod("Layer_Mean")` | `typing.Protocol` + `isinstance` dispatch |
| Warnings + NaN return | `warning("msg"); return(NaN)` | `warnings.warn("msg"); return np.nan` |
| NULL optional args | `NULL` | `None` |
| Infinite cover | `Inf` | `np.inf` |
| Numerical tolerance | `expect_equal(x, y)` (default tol ~1e-7) | `pytest.approx(y)` / `np.testing.assert_allclose(rtol=1e-7)` |
