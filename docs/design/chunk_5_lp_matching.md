## Chunk 5: LP Matching Algorithm

### Task 11a: LP engine — `solve_lp` and `calculate_layer_losses`

**Files:**
- Create: `pyreto/matching.py` (partial — LP engine only)
- Create: `tests/test_matching.py` (partial)

This is the most algorithmically complex module. Port `lp_functions.R` (360 lines) first, then build `Fit_PP` on top.

- [ ] **Step 1: Write failing tests for LP engine**

```python
# tests/test_matching.py
import numpy as np
import pytest
from pyreto.matching import _solve_lp, _calculate_layer_losses


class TestSolveLp:
    def test_produces_feasible_allocation(self):
        # 3-layer tower: verify solution respects decreasing-RoL constraint
        ap = np.array([1000.0, 2000.0, 3000.0])
        el = np.array([500.0, 400.0, 300.0])
        cover = np.array([1000.0, 1000.0, np.inf])
        result = _solve_lp(ap, el, cover)
        assert result is not None
        # Expected losses in result should be ≤ input target
        assert np.all(result >= 0)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_matching.py::TestSolveLp -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `_solve_lp` and `_calculate_layer_losses` in `pyreto/matching.py`**

Port `lp_functions.R:19` (`calculate_layer_losses`) and `lp_functions.R:269` (`solve_lp`). Each call to `_solve_lp` maximises one layer's expected loss; the function is called once per layer and results are averaged. Use `scipy.optimize.linprog` with negated objective.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_matching.py::TestSolveLp -v
```

- [ ] **Step 5: Commit**

```bash
git add pyreto/matching.py tests/test_matching.py
git commit -m "feat: add LP engine for layer-loss matching (solve_lp, calculate_layer_losses)"
```

---

### Task 11b: Alpha-fitting engine — `Fit_PP`, `calculate_taus`, `calculate_alphas`

**Files:**
- Modify: `pyreto/matching.py`
- Modify: `tests/test_matching.py`

Port `FitPP.R` (200 lines). Given a feasible layer-loss allocation from the LP, recover the piecewise-Pareto alpha parameters via root-finding.

- [ ] **Step 1: Write failing tests for alpha engine**

```python
class TestFitPP:
    def test_single_segment(self):
        from pyreto.matching import _fit_pp
        # 2-layer → 1 segment → closed-form alpha
        a = np.array([1000.0, 2000.0])
        s = np.array([100.0, 50.0])   # rate-on-line (EL/cover)
        l = np.array([100.0])
        result = _fit_pp(a, s, l, truncation=None)
        assert result is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_matching.py::TestFitPP -v
```

- [ ] **Step 3: Implement `_fit_pp`, `_calculate_taus`, `_calculate_alphas`**

R's `uniroot` → `scipy.optimize.brentq`.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_matching.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pyreto/matching.py tests/test_matching.py
git commit -m "feat: add alpha-fitting engine for layer-loss matching (Fit_PP)"
```

---

### Task 11c: Top-level `piecewise_pareto_match_layer_losses`

**Files:**
- Modify: `pyreto/matching.py`
- Modify: `tests/test_matching.py`

- [ ] **Step 1: Write failing integration tests (ported from `test_functions_PPP_Model.R`)**

```python
# tests/test_matching.py
import numpy as np
import pytest
from pyreto.matching import piecewise_pareto_match_layer_losses
from pyreto.collective_model import layer_mean, excess_frequency


class TestPiecewisePareto_MatchLayerLosses:
    def setup_method(self):
        self.ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
        self.el = np.array([1000, 900, 800, 600, 500], dtype=float)
        self.cover = np.append(np.diff(self.ap), np.inf)

    def test_basic_fit(self):
        model = piecewise_pareto_match_layer_losses(self.ap, self.el)
        assert model.is_valid()
        np.testing.assert_allclose(
            layer_mean(model, self.cover, self.ap), self.el, rtol=1e-8
        )

    def test_truncated_lp(self):
        model = piecewise_pareto_match_layer_losses(self.ap, self.el, truncation=10000)
        assert model.is_valid()
        np.testing.assert_allclose(
            layer_mean(model, self.cover, self.ap), self.el, rtol=1e-8
        )

    def test_truncated_wd(self):
        model = piecewise_pareto_match_layer_losses(
            self.ap, self.el, truncation=10000, truncation_type="wd"
        )
        assert model.is_valid()
        np.testing.assert_allclose(
            layer_mean(model, self.cover, self.ap), self.el, rtol=1e-8
        )

    def test_with_frequencies(self):
        fqs = np.array([1.1, 0.95, np.nan, np.nan, 0.5])
        model = piecewise_pareto_match_layer_losses(
            self.ap, self.el, frequencies=fqs, truncation=10000
        )
        assert model.is_valid()
        np.testing.assert_allclose(
            layer_mean(model, self.cover, self.ap), self.el, rtol=1e-8
        )
        np.testing.assert_allclose(
            excess_frequency(model, np.array([1000, 2000, 5000])),
            [1.1, 0.95, 0.5],
        )

    def test_only_two_layers(self):
        model = piecewise_pareto_match_layer_losses(
            np.array([1000, 2000], dtype=float),
            np.array([100, 100], dtype=float),
        )
        assert model.alpha == pytest.approx(2)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_matching.py -v
```

- [ ] **Step 3: Implement `piecewise_pareto_match_layer_losses` in `pyreto/matching.py`**

Wire together the LP engine (Task 11a) and alpha-fitting engine (Task 11b). Port `Functions.R` lines 1559–2158. This function:
1. Validates inputs
2. Handles edge cases: zero expected losses, only two layers (closed-form alpha), unlimited layers
3. Calls `_calculate_layer_losses` → `_solve_lp` → `_fit_pp`
4. Applies frequency constraints if provided
5. Constructs and returns a `PPPModel`

- [ ] **Step 4: Run matching tests**

```bash
uv run pytest tests/test_matching.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add pyreto/matching.py tests/test_matching.py
git commit -m "feat: add piecewise Pareto LP matching algorithm"
```

---

