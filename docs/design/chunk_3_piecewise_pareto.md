## Chunk 3: PiecewisePareto Distribution

### Task 8: PiecewisePareto — CDF, PDF, quantile, random sampling

**Files:**
- Create: `pyreto/piecewise_pareto.py`
- Create: `tests/test_piecewise_pareto.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_piecewise_pareto.py
import numpy as np
import pytest
from pyreto.piecewise_pareto import (
    p_piecewise_pareto,
    d_piecewise_pareto,
    q_piecewise_pareto,
    r_piecewise_pareto,
)


class TestPPiecewisePareto:
    def test_basic(self):
        result = p_piecewise_pareto(
            np.array([1000, 2000, 3000, 4000]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=4000,
        )
        np.testing.assert_allclose(result, [0, 0.5, 0.77777777777777779, 1])


class TestDPiecewisePareto:
    def test_basic(self):
        result = d_piecewise_pareto(
            np.array([1000, 2000, 3000, 4000]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=4000,
        )
        np.testing.assert_allclose(
            result, [0.001, 0.0005, 0.00022222222222222221, 0]
        )


class TestQPiecewisePareto:
    def test_basic(self):
        result = q_piecewise_pareto(
            np.arange(1, 10) * 0.1,
            t=np.array([1000, 2000]),
            alpha=np.array([1, 2]),
        )
        np.testing.assert_allclose(
            result,
            [1111.1111111111111, 1250.0, 1428.5714285714287, 1666.6666666666667,
             2000.0, 2236.0679774997902, 2581.9888974716114, 3162.2776601683800, 4472.1359549995805],
            rtol=1e-10,
        )

    def test_with_truncation(self):
        result = q_piecewise_pareto(
            np.arange(1, 5) * 0.2,
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=4000,
        )
        np.testing.assert_allclose(
            result, [1250.0, 1666.6666666666667, 2236.0679774997902, 3060.1459631738917],
            rtol=1e-10,
        )


class TestRPiecewisePareto:
    def test_simulation_parity_lp(self):
        from pyreto.piecewise_pareto import piecewise_pareto_layer_mean
        rng = np.random.default_rng(1972)
        t = np.array([1000, 3000, 5000])
        alpha = np.array([0.7, 1.5, 2.0])
        losses = r_piecewise_pareto(1_000_000, t, alpha, truncation=6000, truncation_type="lp", rng=rng)
        xs = np.minimum(8000, np.maximum(0, losses - 2000))
        ref = piecewise_pareto_layer_mean(8000, 2000, t, alpha, truncation=6000, truncation_type="lp")
        assert round(xs.mean() / ref, 2) == 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_piecewise_pareto.py -v
```

- [ ] **Step 3: Implement `pyreto/piecewise_pareto.py`**

The piecewise Pareto is a concatenation of Pareto segments. Key insight: the normalisation constant and the piece boundaries determine the CDF/PDF. The `truncation_type` parameter controls whether truncation is "last piece" (`lp`) or "whole distribution" (`wd`).

Follow `Functions.R` lines 2370–2713 for CDF/PDF/quantile/random. Use `np.vectorize` for scalar implementations.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_piecewise_pareto.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add pyreto/piecewise_pareto.py tests/test_piecewise_pareto.py
git commit -m "feat: add PiecewisePareto CDF, PDF, quantile, and random sampling"
```

---

### Task 9: PiecewisePareto — layer moments and ML estimator

**Files:**
- Modify: `pyreto/piecewise_pareto.py`
- Modify: `tests/test_piecewise_pareto.py`

- [ ] **Step 1: Add layer moment tests**

```python
from pyreto.piecewise_pareto import (
    piecewise_pareto_layer_mean,
    piecewise_pareto_layer_sm,
    piecewise_pareto_layer_var,
)


class TestPiecewisePareto_LayerMean:
    def test_basic(self):
        assert piecewise_pareto_layer_mean(8000, 2000, t=1000, alpha=2) == pytest.approx(400)

    def test_with_truncation(self):
        assert piecewise_pareto_layer_mean(
            8000, 2000, t=5000, alpha=2, truncation=10000
        ) == pytest.approx(4666.66666666667)

    def test_vectorised_covers_attachments(self):
        result = piecewise_pareto_layer_mean(
            np.array([8000, 2000]), np.array([2000, 1000]),
            t=5000, alpha=2, truncation=10000,
        )
        np.testing.assert_allclose(result, [4666.66666666667, 2000])

    def test_vectorised_piecewise(self):
        result = piecewise_pareto_layer_mean(
            np.array([8000, 2000]), np.array([2000, 1000]),
            t=np.array([1000, 3000, 5000]),
            alpha=np.array([1, 2, 3]),
            truncation=10000,
        )
        np.testing.assert_allclose(
            result, [976.89367953673514, 1098.61228866810916], rtol=1e-10
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_piecewise_pareto.py::TestPiecewisePareto_LayerMean -v
```

- [ ] **Step 3: Implement layer moments and ML estimator**

Follow `Functions.R` lines 1079–1431 (layer moments) and 3027–3302 (ML estimator). ML uses `scipy.optimize.minimize_scalar` or `brentq`.

- [ ] **Step 4: Run all PiecewisePareto tests**

```bash
uv run pytest tests/test_piecewise_pareto.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pyreto/piecewise_pareto.py tests/test_piecewise_pareto.py
git commit -m "feat: add PiecewisePareto layer moments and ML estimator"
```

---

