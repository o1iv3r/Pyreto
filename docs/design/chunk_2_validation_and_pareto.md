## Chunk 2: Validation and Pareto Distribution

### Task 3: Internal validation helpers

**Files:**
- Create: `pyreto/_validation.py`
- Create: `tests/conftest.py`

These mirror `ValidationFunctions.R`. They are internal (`_` prefix) — not part of the public API.

- [ ] **Step 1: Create `tests/conftest.py`**

```python
"""Shared test fixtures."""
import numpy as np
import pytest


@pytest.fixture
def pareto_losses() -> np.ndarray:
    """Reference loss sample from R test suite."""
    return np.array([
        1622.49986584698, 1025.1735923535, 1142.67198754259,
        1598.2131674777, 1369.79742768744, 1006.5249344124,
        2019.3663238659, 1007.2758879241, 1377.79293040511,
        1605.21438984656,
    ])
```

- [ ] **Step 2: Write failing tests for validation helpers**

```python
# tests/test_validation.py  (internal, not user-facing)
import numpy as np
import pytest
from pyreto._validation import (
    is_positive_vector,
    is_nonnegative_finite_vector,
    is_positive_finite_number,
    is_nonnegative_finite_number,
    valid_parameters_pareto,
    valid_parameters_piecewise_pareto,
    valid_parameters_gen_pareto,
)


def test_is_positive_vector():
    assert is_positive_vector(np.array([1.0, 2.0]))
    assert not is_positive_vector(np.array([0.0, 1.0]))
    assert not is_positive_vector(np.array([-1.0, 2.0]))
    assert not is_positive_vector(np.array([np.nan, 1.0]))


def test_is_nonnegative_finite_vector():
    assert is_nonnegative_finite_vector(np.array([0.0, 1.0]))
    assert not is_nonnegative_finite_vector(np.array([-1.0, 1.0]))
    assert not is_nonnegative_finite_vector(np.array([np.inf, 1.0]))


def test_is_positive_finite_number():
    assert is_positive_finite_number(1.0)
    assert not is_positive_finite_number(0.0)
    assert not is_positive_finite_number(np.inf)
    assert not is_positive_finite_number(-1.0)


def test_valid_parameters_pareto():
    assert valid_parameters_pareto(t=1000.0, alpha=2.0, truncation=None)
    assert not valid_parameters_pareto(t=0.0, alpha=2.0, truncation=None)
    assert not valid_parameters_pareto(t=1000.0, alpha=-1.0, truncation=None)
    assert valid_parameters_pareto(t=1000.0, alpha=2.0, truncation=5000.0)
    assert not valid_parameters_pareto(t=1000.0, alpha=2.0, truncation=500.0)  # truncation <= t


def test_valid_parameters_piecewise_pareto():
    # interior alpha=0 is allowed; last alpha must be > 0
    assert valid_parameters_piecewise_pareto(
        t=[1000, 2000], alpha=[0, 1.5], truncation=None, truncation_type="lp"
    )
    assert not valid_parameters_piecewise_pareto(
        t=[1000, 2000], alpha=[1, 0], truncation=None, truncation_type="lp"
    )  # last alpha is zero
    # truncation must exceed max(t), not just t[0]
    assert not valid_parameters_piecewise_pareto(
        t=[1000, 5000], alpha=[1, 2], truncation=3000, truncation_type="lp"
    )  # 3000 < max(t)=5000
    assert valid_parameters_piecewise_pareto(
        t=[1000, 5000], alpha=[1, 2], truncation=10000, truncation_type="lp"
    )
```

- [ ] **Step 3: Run to verify failure**

```bash
uv run pytest tests/test_validation.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 4: Implement `pyreto/_validation.py`**

```python
"""Internal input validation helpers.

Mirrors R's ValidationFunctions.R. All functions are private (prefixed _).
These are not part of the public API.
"""
from __future__ import annotations

import warnings
import numpy as np
from numpy.typing import ArrayLike


def is_positive_vector(x: ArrayLike) -> bool:
    """Return True if x is an array of strictly positive finite numbers."""
    arr = np.asarray(x, dtype=float)
    return bool(np.all(arr > 0) and np.all(np.isfinite(arr)))


def is_nonnegative_finite_vector(x: ArrayLike) -> bool:
    """Return True if x is an array of non-negative finite numbers."""
    arr = np.asarray(x, dtype=float)
    return bool(np.all(arr >= 0) and np.all(np.isfinite(arr)))


def is_positive_or_na_finite_vector(x: ArrayLike) -> bool:
    """Return True if x contains only positive finite numbers or NaN."""
    arr = np.asarray(x, dtype=float)
    return bool(np.all((arr > 0) | np.isnan(arr)) and np.all(np.isfinite(arr) | np.isnan(arr)))


def is_nonnegative_or_na_finite_vector(x: ArrayLike) -> bool:
    arr = np.asarray(x, dtype=float)
    return bool(np.all((arr >= 0) | np.isnan(arr)) and np.all(np.isfinite(arr) | np.isnan(arr)))


def is_positive_finite_number(x: float) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x) and x > 0


def is_nonnegative_finite_number(x: float) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x) and x >= 0


def is_positive_number(x: float) -> bool:
    return isinstance(x, (int, float)) and not np.isnan(x) and x > 0


def is_nonnegative_number(x: float) -> bool:
    return isinstance(x, (int, float)) and not np.isnan(x) and x >= 0


def valid_parameters_pareto(
    t: float,
    alpha: float,
    truncation: float | None,
    allow_alpha_zero: bool = False,
    comment: bool = False,
) -> bool:
    """Validate Pareto parameters. Returns True if valid, False otherwise (with optional warning)."""
    ok = True
    msgs: list[str] = []

    if not is_positive_finite_number(t):
        msgs.append("t must be a positive finite number.")
        ok = False
    if allow_alpha_zero:
        if not is_nonnegative_number(alpha):
            msgs.append("alpha must be a non-negative number.")
            ok = False
    else:
        if not is_positive_number(alpha):
            msgs.append("alpha must be a positive number.")
            ok = False
    if truncation is not None:
        if not is_positive_finite_number(truncation):
            msgs.append("truncation must be a positive finite number.")
            ok = False
        elif t is not None and np.isfinite(t) and truncation <= t:
            msgs.append("truncation must be greater than t.")
            ok = False

    if comment and msgs:
        warnings.warn(" ".join(msgs))
    return ok


def valid_parameters_piecewise_pareto(
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: float | None,
    truncation_type: str,
    comment: bool = False,
) -> bool:
    """Validate PiecewisePareto parameters."""
    t_arr = np.asarray(t, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    ok = True
    msgs: list[str] = []

    if truncation_type not in ("lp", "wd"):
        msgs.append("truncation_type must be 'lp' or 'wd'.")
        ok = False
    if not is_positive_finite_number(float(t_arr[0])):
        msgs.append("t[0] must be a positive finite number.")
        ok = False
    if not np.all(np.diff(t_arr) > 0):
        msgs.append("t must be strictly increasing.")
        ok = False
    # Interior alphas may be zero (flat severity segment); last alpha must be strictly positive.
    if not is_nonnegative_finite_vector(alpha_arr) or float(alpha_arr[-1]) <= 0:
        msgs.append("alpha must be non-negative with the last value strictly positive.")
        ok = False
    if len(t_arr) != len(alpha_arr):
        msgs.append("t and alpha must have the same length.")
        ok = False
    if truncation is not None:
        if not is_positive_finite_number(truncation):
            msgs.append("truncation must be a positive finite number.")
            ok = False
        elif truncation <= float(t_arr[-1]):
            # Must be greater than max(t), i.e. the last (highest) threshold
            msgs.append("truncation must be greater than max(t).")
            ok = False

    if comment and msgs:
        warnings.warn(" ".join(msgs))
    return ok


def valid_parameters_gen_pareto(
    t: float,
    alpha_ini: float,
    alpha_tail: float,
    truncation: float | None,
    comment: bool = False,
) -> bool:
    """Validate generalized Pareto parameters."""
    ok = True
    msgs: list[str] = []

    if not is_positive_finite_number(t):
        msgs.append("t must be a positive finite number.")
        ok = False
    if not is_positive_number(alpha_ini):
        msgs.append("alpha_ini must be a positive number.")
        ok = False
    if not is_positive_number(alpha_tail):
        msgs.append("alpha_tail must be a positive number.")
        ok = False
    if truncation is not None:
        if not is_positive_finite_number(truncation):
            msgs.append("truncation must be a positive finite number.")
            ok = False
        elif np.isfinite(t) and truncation <= t:
            msgs.append("truncation must be greater than t.")
            ok = False

    if comment and msgs:
        warnings.warn(" ".join(msgs))
    return ok
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_validation.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add pyreto/_validation.py tests/conftest.py tests/test_validation.py
git commit -m "feat: add internal validation helpers"
```

---

### Task 4: Pareto distribution — CDF, PDF, quantile, random sampling

**Files:**
- Create: `pyreto/pareto.py`
- Create: `tests/test_pareto.py` (partial — this task covers distribution functions)

- [ ] **Step 1: Write failing tests for p/d/q/r Pareto**

```python
# tests/test_pareto.py
import numpy as np
import pytest
from pyreto.pareto import p_pareto, d_pareto, q_pareto, r_pareto


class TestPPareto:
    def test_basic(self):
        result = p_pareto(np.array([2000, 4000, 6000]), t=np.array([1000, 2000, 3000]), alpha=2)
        np.testing.assert_allclose(result, [0.75, 0.75, 0.75])

    def test_with_truncation(self):
        result = p_pareto(
            np.array([2000, 4000, 6000]),
            t=np.array([1000, 2000, 3000]),
            alpha=2,
            truncation=np.array([10000, 20000, 30000]),
        )
        np.testing.assert_allclose(result, [0.75757575757575757] * 3)


class TestDPareto:
    def test_basic(self):
        result = d_pareto(np.array([2000, 4000, 6000]), t=np.array([1000, 2000, 3000]), alpha=2)
        np.testing.assert_allclose(
            result, [2.5e-4, 1.25e-4, 8.333333333333333e-5], rtol=1e-10
        )


class TestQPareto:
    def test_basic(self):
        result = q_pareto(
            np.array([0.3, 0.6, 0.9]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
        )
        np.testing.assert_allclose(
            result, [1428.5714285714287, 3162.2776601683790, 6463.3040700956490], rtol=1e-10
        )

    def test_with_truncation_scalar(self):
        result = q_pareto(
            np.array([0.3, 0.6, 0.9]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=10000,
        )
        np.testing.assert_allclose(
            result, [1369.8630136986301, 3071.4755841697556, 6011.2419963076682], rtol=1e-10
        )

    def test_with_truncation_vector(self):
        result = q_pareto(
            np.array([0.3, 0.6, 0.9]),
            t=np.array([1000, 2000, 3000]),
            alpha=np.array([1, 2, 3]),
            truncation=np.array([10000, 20000, 30000]),
        )
        np.testing.assert_allclose(
            result, [1369.8630136986301, 3138.8241028717225, 6444.0296890428708], rtol=1e-10
        )


class TestRPareto:
    def test_shape(self):
        samples = r_pareto(100, t=1000, alpha=2)
        assert samples.shape == (100,)
        assert np.all(samples >= 1000)

    def test_simulation_parity_no_truncation(self):
        rng = np.random.default_rng(1972)
        n = 1_000_000
        losses = r_pareto(n, t=1000, alpha=1.5, rng=rng)
        xs = np.minimum(8000, np.maximum(0, losses - 2000))
        ratio = round(xs.mean() / 781.75803033941929, 2)
        assert ratio == 1.0

    def test_simulation_parity_with_truncation(self):
        rng = np.random.default_rng(1972)
        n = 1_000_000
        losses = r_pareto(n, t=1000, alpha=1.5, truncation=20000, rng=rng)
        xs = np.minimum(8000, np.maximum(0, losses - 2000))
        ratio = round(xs.mean() / 700.14314962210699, 2)
        assert ratio == 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_pareto.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement CDF, PDF, quantile, random in `pyreto/pareto.py`**

Start the file with the standard/CDF/PDF/quantile/random section. Note that many functions in the R source have a scalar (`_s`) and vectorised version — in Python we implement only the vectorised form using numpy, which handles both scalar and array inputs naturally.

```python
"""Pareto distribution functions for reinsurance pricing.

Mirrors the Pareto section of the R Pareto package (Functions.R).
All functions accept scalar or array inputs (numpy broadcasting).

R → Python name mapping (see docs/function_mapping.md):
  pPareto               → p_pareto
  dPareto               → d_pareto
  qPareto               → q_pareto
  rPareto               → r_pareto
  Pareto_Layer_Mean     → pareto_layer_mean
  Pareto_Layer_SM       → pareto_layer_sm
  Pareto_Layer_Var      → pareto_layer_var
  Pareto_Extrapolation  → pareto_extrapolation
  Pareto_Find_Alpha_btw_Layers    → pareto_find_alpha_btw_layers
  Pareto_Find_Alpha_btw_FQ_Layer  → pareto_find_alpha_btw_fq_layer
  Pareto_Find_Alpha_btw_FQs       → pareto_find_alpha_btw_fqs
  Pareto_ML_Estimator_Alpha       → pareto_ml_estimator_alpha
  Local_Pareto_Alpha              → local_pareto_alpha  (in fitting.py)
"""
from __future__ import annotations

import warnings
import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from pyreto._validation import valid_parameters_pareto


def p_pareto(
    x: ArrayLike,
    t: ArrayLike,
    alpha: float,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """CDF of the Pareto(t, alpha) distribution.

    Parameters
    ----------
    x : array-like
        Values at which to evaluate the CDF.
    t : array-like
        Threshold (scale) parameter. Must be positive.
    alpha : float
        Shape parameter. Must be positive.
    truncation : array-like or None
        Upper truncation point. If provided, the distribution is truncated at
        this value and the CDF is renormalised.

    Returns
    -------
    np.ndarray
        CDF values in [0, 1].

    Examples
    --------
    >>> p_pareto(np.array([2000.0]), t=1000.0, alpha=2.0)
    array([0.75])
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    alpha = float(alpha)
    result = np.where(x <= t, 0.0, 1.0 - (t / x) ** alpha)

    if truncation is not None:
        trunc = np.asarray(truncation, dtype=float)
        # P(X <= trunc) under untruncated distribution
        p_trunc = 1.0 - (t / trunc) ** alpha
        result = np.where(x >= trunc, 1.0, result / p_trunc)

    return result


def d_pareto(
    x: ArrayLike,
    t: ArrayLike,
    alpha: float,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """PDF of the Pareto(t, alpha) distribution.

    Parameters
    ----------
    x : array-like
        Values at which to evaluate the PDF.
    t : array-like
        Threshold parameter.
    alpha : float
        Shape parameter.
    truncation : array-like or None
        Upper truncation point.

    Returns
    -------
    np.ndarray
        PDF values (zero outside support).

    Examples
    --------
    >>> d_pareto(np.array([2000.0]), t=1000.0, alpha=2.0)
    array([0.00025])
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    alpha = float(alpha)
    pdf = np.where(x < t, 0.0, alpha * t**alpha / x ** (alpha + 1))

    if truncation is not None:
        trunc = np.asarray(truncation, dtype=float)
        p_trunc = 1.0 - (t / trunc) ** alpha
        pdf = np.where(x > trunc, 0.0, pdf / p_trunc)

    return pdf


def q_pareto(
    p: ArrayLike,
    t: ArrayLike,
    alpha: ArrayLike,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Quantile function of the Pareto(t, alpha) distribution.

    Parameters
    ----------
    p : array-like
        Probabilities in [0, 1].
    t : array-like
        Threshold parameter.
    alpha : array-like
        Shape parameter.
    truncation : array-like or None
        Upper truncation point.

    Returns
    -------
    np.ndarray
        Quantiles.

    Examples
    --------
    >>> q_pareto(np.array([0.75]), t=1000.0, alpha=2.0)
    array([2000.])
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    if truncation is None:
        return t / (1.0 - p) ** (1.0 / alpha)

    trunc = np.asarray(truncation, dtype=float)
    p_trunc = 1.0 - (t / trunc) ** alpha
    p_adj = p * p_trunc  # scale p back to untruncated range
    return np.minimum(trunc, t / (1.0 - p_adj) ** (1.0 / alpha))


def r_pareto(
    n: int,
    t: float,
    alpha: float,
    truncation: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw random samples from the Pareto(t, alpha) distribution.

    Parameters
    ----------
    n : int
        Number of samples.
    t : float
        Threshold parameter.
    alpha : float
        Shape parameter.
    truncation : float or None
        Upper truncation point.
    rng : np.random.Generator or None
        Random number generator (for reproducibility). If None, uses default_rng().

    Returns
    -------
    np.ndarray
        Array of n samples, all >= t.

    Examples
    --------
    >>> r_pareto(5, t=1000.0, alpha=2.0, rng=np.random.default_rng(42))
    array([...])
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(0, 1, n)
    return q_pareto(u, t=t, alpha=alpha, truncation=truncation)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pareto.py::TestPPareto tests/test_pareto.py::TestDPareto tests/test_pareto.py::TestQPareto tests/test_pareto.py::TestRPareto -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add pyreto/pareto.py tests/test_pareto.py
git commit -m "feat: add Pareto CDF, PDF, quantile, and random sampling"
```

---

### Task 5: Pareto distribution — layer moments

**Files:**
- Modify: `pyreto/pareto.py`
- Modify: `tests/test_pareto.py`

- [ ] **Step 1: Add layer moment tests**

Append to `tests/test_pareto.py`:

```python
from pyreto.pareto import pareto_layer_mean, pareto_layer_sm, pareto_layer_var


class TestParetoLayerMean:
    def test_basic(self):
        assert pareto_layer_mean(8000, 2000, alpha=2) == pytest.approx(1600)

    def test_with_t(self):
        assert pareto_layer_mean(8000, 2000, alpha=2, t=1000) == pytest.approx(400)

    def test_with_t_and_truncation(self):
        assert pareto_layer_mean(8000, 2000, alpha=2, t=5000, truncation=10000) == pytest.approx(4666.66666666667)

    def test_alpha_zero(self):
        assert pareto_layer_mean(2000, 1000, 0, truncation=5000, t=500) == pytest.approx(835.16520831955449)

    def test_vectorised(self):
        np.testing.assert_allclose(
            pareto_layer_mean(np.arange(1, 4) * 8000, np.arange(1, 4) * 2000, alpha=2),
            np.arange(1, 4) * 1600,
        )

    def test_vectorised_with_t(self):
        np.testing.assert_allclose(
            pareto_layer_mean(
                np.arange(1, 4) * 8000, np.arange(1, 4) * 2000,
                alpha=2, t=np.arange(1, 4) * 1000,
            ),
            np.arange(1, 4) * 400,
        )

    def test_vectorised_with_t_and_truncation(self):
        np.testing.assert_allclose(
            pareto_layer_mean(
                np.arange(1, 4) * 8000, np.arange(1, 4) * 2000,
                alpha=2,
                t=np.arange(1, 4) * 5000,
                truncation=np.arange(1, 4) * 10000,
            ),
            np.arange(1, 4) * 4666.66666666667,
        )


class TestParetoLayerSM:
    def test_basic(self):
        assert pareto_layer_sm(8000, 2000, alpha=2) == pytest.approx(6475503.2994728)

    def test_with_t(self):
        assert pareto_layer_sm(8000, 2000, alpha=2, t=1000) == pytest.approx(1618875.8248682)

    def test_with_truncation(self):
        assert pareto_layer_sm(8000, 2000, alpha=2, t=5000, truncation=10000) == pytest.approx(23543145.370663)

    def test_alpha_zero_with_t(self):
        assert pareto_layer_sm(2000, 1000, alpha=0, truncation=5000, t=1500) == pytest.approx(2584318.901925616)

    def test_vectorised(self):
        np.testing.assert_allclose(
            pareto_layer_sm(np.arange(1, 4) * 8000, np.arange(1, 4) * 2000, alpha=2),
            np.arange(1, 4) ** 2 * 6475503.2994728,
        )


class TestParetoLayerVar:
    def test_basic(self):
        assert pareto_layer_var(8000, 2000, alpha=2) == pytest.approx(3915503.2994728)

    def test_with_t(self):
        assert pareto_layer_var(8000, 2000, alpha=2, t=1000) == pytest.approx(1458875.8248682)

    def test_with_truncation(self):
        assert pareto_layer_var(8000, 2000, alpha=2, t=5000, truncation=10000) == pytest.approx(1765367.59288524)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_pareto.py::TestParetoLayerMean -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement layer moments in `pyreto/pareto.py`**

The layer mean of Pareto(t, alpha) in cover C xs attachment A is derived from the analytical formula in the R source. Key formula for `Pareto_Layer_Mean_s` (scalar):

- If `t >= A + C`: 0
- If `t <= A`: standard formula using `(t/A)^alpha`
- With truncation: subtract contribution above truncation point

Implement `pareto_layer_mean`, `pareto_layer_sm`, `pareto_layer_var` following the scalar implementations in `Functions.R` lines 47–219, 369–437, 242–345, wrapped with `np.vectorize` for broadcasting:

```python
def _pareto_layer_mean_scalar(
    cover: float, attachment: float, alpha: float, t: float | None, truncation: float | None
) -> float:
    """Scalar implementation. See R source Pareto_Layer_Mean_s."""
    # ... (full implementation following Functions.R:47-155)

def pareto_layer_mean(
    cover: ArrayLike,
    attachment_point: ArrayLike,
    alpha: ArrayLike,
    t: ArrayLike | None = None,
    truncation: ArrayLike | None = None,
) -> np.ndarray:
    """Expected loss of Pareto(t, alpha) in reinsurance layer cover xs attachment_point.
    ...
    """
    # vectorize over (cover, attachment_point, alpha) and optionally t, truncation
    _vfun = np.vectorize(_pareto_layer_mean_scalar)
    cover = np.asarray(cover, dtype=float)
    attachment_point = np.asarray(attachment_point, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    t_arr = np.asarray(t, dtype=float) if t is not None else attachment_point.copy()
    trunc_arr = np.asarray(truncation, dtype=float) if truncation is not None else np.full_like(cover, np.inf)
    return _vfun(cover, attachment_point, alpha, t_arr, trunc_arr)
```

See `Functions.R` for the exact scalar formulas. Implement `pareto_layer_sm` (second moment) and `pareto_layer_var` similarly.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pareto.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add pyreto/pareto.py tests/test_pareto.py
git commit -m "feat: add Pareto layer mean, second moment, and variance"
```

---

### Task 6: Pareto distribution — alpha-finding and extrapolation

**Files:**
- Modify: `pyreto/pareto.py`
- Modify: `tests/test_pareto.py`

- [ ] **Step 1: Add tests**

```python
from pyreto.pareto import (
    pareto_extrapolation,
    pareto_find_alpha_btw_layers,
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_fqs,
)


class TestParetoExtrapolation:
    def test_ratio(self):
        assert pareto_extrapolation(1000, 1000, 2000, 2000, alpha=2) == pytest.approx(0.5)

    def test_with_exp_loss(self):
        assert pareto_extrapolation(1000, 1000, 2000, 2000, alpha=2, exp_loss_1=1000) == pytest.approx(500)

    def test_with_truncation(self):
        assert pareto_extrapolation(
            1000, 1000, 2000, 2000, alpha=2, truncation=3000, exp_loss_1=1000
        ) == pytest.approx(142.85714285714292)

    def test_vectorised(self):
        result = pareto_extrapolation(
            np.array([1000, 1000, 1000, np.inf]),
            1000,
            np.array([2000, 2000, 2000, np.inf]),
            2000,
            alpha=np.array([2, 2, 2, 0]),
            truncation=np.array([np.inf, np.inf, 3000, 3000]),
            exp_loss_1=np.array([1, 1000, 1000, 100]),
        )
        np.testing.assert_allclose(result, [0.5, 500, 142.85714285714292, 20.975411735345432])


class TestParetoFindAlpha:
    def test_btw_layers_equal(self):
        assert pareto_find_alpha_btw_layers(1000, 1000, 100, 2000, 2000, 100) == pytest.approx(1)

    def test_btw_layers_halved(self):
        assert pareto_find_alpha_btw_layers(1000, 1000, 100, 2000, 2000, 50) == pytest.approx(2)

    def test_btw_fq_layer(self):
        assert pareto_find_alpha_btw_fq_layer(1000, 1, 2000, 2000, 100) == pytest.approx(2.9330042139247037)

    def test_btw_fqs_basic(self):
        assert pareto_find_alpha_btw_fqs(1000, 1, 2000, 0.5) == pytest.approx(1)
        assert pareto_find_alpha_btw_fqs(2000, 0.25, 1000, 1) == pytest.approx(2)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_pareto.py::TestParetoExtrapolation tests/test_pareto.py::TestParetoFindAlpha -v
```

- [ ] **Step 3: Implement using `scipy.optimize.brentq` for root-finding**

R's `uniroot` → `scipy.optimize.brentq`. The alpha-finding functions solve `f(alpha) = 0` where `f` is the difference between two layer means or frequencies expressed in terms of alpha.

- [ ] **Step 4: Run all Pareto tests**

```bash
uv run pytest tests/test_pareto.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add pyreto/pareto.py tests/test_pareto.py
git commit -m "feat: add Pareto extrapolation and alpha-finding functions"
```

---

### Task 7: Pareto ML estimator

**Files:**
- Modify: `pyreto/pareto.py`
- Modify: `tests/test_pareto.py`

- [ ] **Step 1: Add ML estimator tests (ported from R)**

```python
from pyreto.pareto import pareto_ml_estimator_alpha


class TestParetoMLEstimator:
    def test_basic(self, pareto_losses):
        assert round(pareto_ml_estimator_alpha(pareto_losses, 1000), 4) == 3.4060

    def test_with_truncation(self, pareto_losses):
        assert round(pareto_ml_estimator_alpha(pareto_losses, 1000, truncation=3000), 4) == 2.9601

    def test_weights_equivalent_to_duplicates(self, pareto_losses):
        losses2 = np.concatenate([pareto_losses, pareto_losses[:2]])
        w = np.ones(len(pareto_losses))
        w[:2] = 2
        assert pareto_ml_estimator_alpha(pareto_losses, 1000, weights=w) == pytest.approx(
            pareto_ml_estimator_alpha(losses2, 1000)
        )

    def test_with_reporting_thresholds(self, pareto_losses):
        rt = np.array([1000, 1000, 1000, 1200, 1200, 1000, 1500, 1000, 1000, 1000], dtype=float)
        assert round(pareto_ml_estimator_alpha(pareto_losses, 1000, reporting_thresholds=rt), 5) == 4.61698
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_pareto.py::TestParetoMLEstimator -v
```

- [ ] **Step 3: Implement `pareto_ml_estimator_alpha`**

R source is `Functions.R` lines 2814–3026. The MLE solves the log-likelihood using `scipy.optimize.minimize_scalar` or `brentq`. Support: truncated, censored, weighted, heterogeneous reporting thresholds.

- [ ] **Step 4: Run all Pareto tests**

```bash
uv run pytest tests/test_pareto.py -v
```

Expected: All PASS.

- [ ] **Step 5: Run ruff and ty**

```bash
uv run ruff check pyreto/pareto.py && uv run ty check pyreto
```

- [ ] **Step 6: Commit**

```bash
git add pyreto/pareto.py tests/test_pareto.py
git commit -m "feat: add Pareto ML alpha estimator"
```

---

