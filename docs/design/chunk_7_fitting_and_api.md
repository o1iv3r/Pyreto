## Chunk 7: Fitting Utilities and Public API

### Task 14: Fitting utilities — Panjer, local_pareto_alpha, fit_references, fit_pml_curve

**Files:**
- Create: `pyreto/fitting.py`
- Create: `tests/test_fitting.py`

- [ ] **Step 1: Write failing tests for `local_pareto_alpha`**

```python
# tests/test_fitting.py
import numpy as np
import pytest
from pyreto.fitting import local_pareto_alpha


class TestLocalParetoAlpha:
    def test_norm(self):
        x = np.arange(1, 11) * 1e6
        result = local_pareto_alpha(x, "norm", mean=5e6, sd=2e6)
        np.testing.assert_allclose(
            result,
            [0.027623931339495, 0.138789750458851, 0.431399956408768, 1.01832086767407,
             1.99471140200716, 3.42323331110419, 5.33797346656343, 7.75470866649017,
             10.6794698977028, 14.1137239883195],
            rtol=1e-8,
        )

    def test_lnorm(self):
        x = np.arange(1, 11) * 1e6
        result = local_pareto_alpha(x, "lnorm", meanlog=0, sdlog=4)
        np.testing.assert_allclose(result[0], 0.92698000118132, rtol=1e-8)

    def test_pareto(self):
        x = np.arange(1, 11) * 1e6
        result = local_pareto_alpha(x, "Pareto", t=1e6, alpha=1, truncation=20e6)
        np.testing.assert_allclose(result[0], 1.05263157894737, rtol=1e-8)
```

- [ ] **Step 2: Add Panjer tests to `tests/test_fitting.py`**

```python
from pyreto.fitting import d_panjer, r_panjer


class TestPanjer:
    def test_d_panjer_poisson(self):
        # dispersion=1 → Poisson; P(X=k) = e^(-mean) * mean^k / k!
        import math
        mean = 2.0
        for k in range(5):
            expected = math.exp(-mean) * mean**k / math.factorial(k)
            assert d_panjer(k, mean=mean, dispersion=1.0) == pytest.approx(expected, rel=1e-10)

    def test_d_panjer_negbin(self):
        # dispersion > 1 → NegBin; verify probabilities sum to ~1
        probs = [d_panjer(k, mean=3.0, dispersion=2.0) for k in range(50)]
        assert sum(probs) == pytest.approx(1.0, rel=1e-6)

    def test_r_panjer_poisson_mean(self):
        rng = np.random.default_rng(42)
        samples = r_panjer(100_000, mean=3.0, dispersion=1.0, rng=rng)
        assert samples.mean() == pytest.approx(3.0, rel=0.02)

    def test_r_panjer_negbin_mean(self):
        rng = np.random.default_rng(42)
        samples = r_panjer(100_000, mean=5.0, dispersion=3.0, rng=rng)
        assert samples.mean() == pytest.approx(5.0, rel=0.02)
```

- [ ] **Step 3: Run to verify failure**

```bash
uv run pytest tests/test_fitting.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `pyreto/fitting.py`**

Port from `Functions.R`:
- `local_pareto_alpha` (lines 3303–3380): computes local Pareto alpha at given points. Uses `scipy.stats` for named distributions (`norm`, `lnorm`, `gamma`, `weibull`, `exp`); custom for `Pareto`, `GenPareto`, `PiecewisePareto`.
- `d_panjer` / `r_panjer` (lines 4263–4378): Panjer/NegBin claim-count distribution. `dispersion=1` → Poisson, `dispersion>1` → NegBin, `dispersion<1` → Binomial.
- `fit_references` (lines 4379–4755): complex reference fitting function calling the LP engine.
- `fit_pml_curve` (lines 4756–end): fits PML curve to return-period data.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_fitting.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add pyreto/fitting.py tests/test_fitting.py
git commit -m "feat: add fitting utilities (d_panjer, r_panjer, local_pareto_alpha, fit_references, fit_pml_curve)"
```

---

### Task 15: Public API — `__init__.py` and function mapping table

**Files:**
- Modify: `pyreto/__init__.py`
- Create: `docs/function_mapping.md`

- [ ] **Step 1: Update `pyreto/__init__.py` to export all public functions**

```python
"""Pyreto: Pareto, piecewise Pareto and generalized Pareto tools for reinsurance pricing.

Port of the R Pareto package by Ulrich Riegel (GPL >= 2).
"""

__version__ = "0.1.0"

from pyreto.pareto import (
    p_pareto, d_pareto, q_pareto, r_pareto,
    pareto_layer_mean, pareto_layer_sm, pareto_layer_var,
    pareto_extrapolation,
    pareto_find_alpha_btw_layers,
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_fqs,
    pareto_ml_estimator_alpha,
)
from pyreto.piecewise_pareto import (
    p_piecewise_pareto, d_piecewise_pareto, q_piecewise_pareto, r_piecewise_pareto,
    piecewise_pareto_layer_mean, piecewise_pareto_layer_sm, piecewise_pareto_layer_var,
    piecewise_pareto_ml_estimator_alpha,
)
from pyreto.gen_pareto import (
    p_gen_pareto, d_gen_pareto, q_gen_pareto, r_gen_pareto,
    gen_pareto_layer_mean, gen_pareto_layer_sm, gen_pareto_layer_var,
    gen_pareto_ml_estimator_alpha,
)
from pyreto.matching import piecewise_pareto_match_layer_losses
from pyreto.ppp_model import PPPModel
from pyreto.pgp_model import PGPModel
from pyreto.collective_model import layer_mean, layer_var, layer_sd, excess_frequency, simulate_losses
from pyreto.fitting import local_pareto_alpha, fit_references, fit_pml_curve

__all__ = [
    # Pareto
    "p_pareto", "d_pareto", "q_pareto", "r_pareto",
    "pareto_layer_mean", "pareto_layer_sm", "pareto_layer_var",
    "pareto_extrapolation",
    "pareto_find_alpha_btw_layers", "pareto_find_alpha_btw_fq_layer", "pareto_find_alpha_btw_fqs",
    "pareto_ml_estimator_alpha",
    # PiecewisePareto
    "p_piecewise_pareto", "d_piecewise_pareto", "q_piecewise_pareto", "r_piecewise_pareto",
    "piecewise_pareto_layer_mean", "piecewise_pareto_layer_sm", "piecewise_pareto_layer_var",
    "piecewise_pareto_ml_estimator_alpha",
    # GenPareto
    "p_gen_pareto", "d_gen_pareto", "q_gen_pareto", "r_gen_pareto",
    "gen_pareto_layer_mean", "gen_pareto_layer_sm", "gen_pareto_layer_var",
    "gen_pareto_ml_estimator_alpha",
    # Matching
    "piecewise_pareto_match_layer_losses",
    # Models
    "PPPModel", "PGPModel",
    "layer_mean", "layer_var", "layer_sd", "excess_frequency", "simulate_losses",
    # Fitting
    "local_pareto_alpha", "fit_references", "fit_pml_curve",
]
```

- [ ] **Step 2: Verify full import**

```bash
uv run python -c "import pyreto; print(dir(pyreto))"
```

- [ ] **Step 3: Create `docs/function_mapping.md`**

Create a full R → Python mapping table. Example:

```markdown
# R to Python Function Mapping

| R function | Python function | Module | Notes |
|---|---|---|---|
| `pPareto` | `p_pareto` | `pyreto.pareto` | |
| `dPareto` | `d_pareto` | `pyreto.pareto` | |
| `qPareto` | `q_pareto` | `pyreto.pareto` | |
| `rPareto` | `r_pareto` | `pyreto.pareto` | |
| `Pareto_CDF` | — | — | Deprecated alias for `pPareto`; intentionally omitted |
| `Pareto_PDF` | — | — | Deprecated alias for `dPareto`; intentionally omitted |
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
| `PiecewisePareto_CDF` | — | — | Deprecated alias for `pPiecewisePareto`; intentionally omitted |
| `PiecewisePareto_PDF` | — | — | Deprecated alias for `dPiecewisePareto`; intentionally omitted |
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
| `PPP_Model` | `PPPModel` | `pyreto.ppp_model` | Constructor → dataclass |
| `is.valid.PPP_Model` | `PPPModel.is_valid()` | `pyreto.ppp_model` | |
| `PPP_Model_Exp_Layer_Loss` | — | — | Deprecated wrapper; use `layer_mean(model, ...)` |
| `PPP_Model_Layer_Var` | — | — | Deprecated wrapper; use `layer_var(model, ...)` |
| `PPP_Model_Layer_Sd` | — | — | Deprecated wrapper; use `layer_sd(model, ...)` |
| `PPP_Model_Excess_Frequency` | — | — | Deprecated wrapper; use `excess_frequency(model, ...)` |
| `PPP_Model_Simulate` | — | — | Deprecated wrapper; use `simulate_losses(model, ...)` |
| `PGP_Model` | `PGPModel` | `pyreto.pgp_model` | Constructor → dataclass |
| `is.valid.PGP_Model` | `PGPModel.is_valid()` | `pyreto.pgp_model` | |
| `Layer_Mean` (generic) | `layer_mean` | `pyreto.collective_model` | |
| `Layer_Var` (generic) | `layer_var` | `pyreto.collective_model` | |
| `Layer_Sd` (generic) | `layer_sd` | `pyreto.collective_model` | |
| `Excess_Frequency` (generic) | `excess_frequency` | `pyreto.collective_model` | |
| `Simulate_Losses` (generic) | `simulate_losses` | `pyreto.collective_model` | |
| `dPanjer` | `d_panjer` | `pyreto.fitting` | |
| `rPanjer` | `r_panjer` | `pyreto.fitting` | |
| `Local_Pareto_Alpha` | `local_pareto_alpha` | `pyreto.fitting` | |
| `Fit_References` | `fit_references` | `pyreto.fitting` | |
| `Fit_PML_Curve` | `fit_pml_curve` | `pyreto.fitting` | |
...
```

- [ ] **Step 4: Commit**

```bash
git add pyreto/__init__.py docs/function_mapping.md
git commit -m "feat: expose public API and add R→Python function mapping table"
```

---

