## Chunk 4: Generalized Pareto Distribution

### Task 10: GenPareto — full distribution

**Files:**
- Create: `pyreto/gen_pareto.py`
- Create: `tests/test_gen_pareto.py`

- [ ] **Step 1: Write failing tests (ported from `test_functions_GenPareto.R`)**

```python
# tests/test_gen_pareto.py
import numpy as np
import pytest
from pyreto.gen_pareto import (
    p_gen_pareto, d_gen_pareto, q_gen_pareto, r_gen_pareto,
    gen_pareto_layer_mean, gen_pareto_layer_sm, gen_pareto_layer_var,
    gen_pareto_ml_estimator_alpha,
)


class TestPGenPareto:
    def test_basic(self):
        result = p_gen_pareto(
            np.arange(1, 4) * 2000,
            t=np.arange(1, 4) * 1000,
            alpha_ini=np.array([1, 1.5, 2]),
            alpha_tail=np.array([1, 2, 3]),
        )
        np.testing.assert_allclose(result, [0.5, 0.67346938775510212, 0.78399999999999992])


class TestGenParetoLayerMean:
    def test_basic_same_alphas(self):
        assert gen_pareto_layer_mean(8000, 2000, t=2000, alpha_ini=2, alpha_tail=2) == pytest.approx(1600)

    def test_different_t(self):
        assert gen_pareto_layer_mean(8000, 2000, t=1000, alpha_ini=2, alpha_tail=2) == pytest.approx(400)

    def test_with_truncation(self):
        assert gen_pareto_layer_mean(
            8000, 2000, t=5000, alpha_ini=2, alpha_tail=1, truncation=10000
        ) == pytest.approx(4619.7960825054124)

    def test_vectorised(self):
        result = gen_pareto_layer_mean(
            8000, 2000,
            t=np.array([2000, 1000, 5000]),
            alpha_ini=2,
            alpha_tail=np.array([2, 2, 1]),
            truncation=np.array([np.inf, np.inf, 10000]),
        )
        np.testing.assert_allclose(result, [1600, 400, 4619.7960825054124])
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_gen_pareto.py -v
```

- [ ] **Step 3: Implement `pyreto/gen_pareto.py`**

Follow `Functions.R` lines 3381–4261. The GenPareto is parameterised by `(t, alpha_ini, alpha_tail)` and defined by its CDF:

`F(x) = 1 - (t/x)^alpha_ini` for `t < x <= t*alpha_ini/alpha_tail` ... with a piecewise formula matching at the junction. Follow the R scalar implementations exactly.

- [ ] **Step 4: Run all GenPareto tests**

```bash
uv run pytest tests/test_gen_pareto.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add pyreto/gen_pareto.py tests/test_gen_pareto.py
git commit -m "feat: add generalized Pareto distribution (CDF, PDF, quantile, random, layer moments, ML)"
```

---

