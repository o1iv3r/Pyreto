# Pyreto Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the R `Pareto` package to a modern, well-tested Python package (`pyreto`) with feature parity, Pythonic naming, and full documentation.

**Architecture:** The package is split into focused modules mirroring the R source files — one module per distribution family plus separate modules for collective models, the LP matching algorithm, and fitting utilities. R's S3 dispatch (`UseMethod`) becomes a Python Protocol + concrete classes. R's `Vectorize()` becomes numpy broadcasting or `np.vectorize`. R's `lpSolve` is replaced by `scipy.optimize.linprog`; `uniroot`/`optimize` become `scipy.optimize.brentq`/`minimize_scalar`.

**Tech Stack:** Python ≥ 3.11, uv, pyproject.toml, ruff, ty, pytest, pre-commit, numpy, scipy, numba (optional, for later hot-path optimisation), polars (where tabular data needed), mkdocs-material, GitHub Actions.

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

## Chunk 1: Project Scaffolding

### Task 1: Initialise the uv project and pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `pyreto/__init__.py`

- [ ] **Step 1: Initialise uv project**

```bash
cd /home/oliver/Code/Pyreto
uv init --name pyreto --python 3.11
```

Expected: `pyproject.toml` created, `.python-version` file created.

- [ ] **Step 2: Replace generated pyproject.toml with correct content**

```toml
[project]
name = "pyreto"
version = "0.1.0"
description = "Python methods and tools for the Pareto, piecewise Pareto and generalized Pareto distributions for reinsurance pricing"
readme = "README.md"
license = { text = "GPL-2.0-or-later" }
authors = [{ name = "Oliver Pfaffel", email = "opfaffel@gmail.com" }]
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "polars>=0.20",
    "numba>=0.59",   # JIT for hot paths; used only where profiling shows it's needed
]

[project.urls]
Homepage = "https://github.com/o1iv3r/Pyreto"
Documentation = "https://o1iv3r.github.io/Pyreto"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "ty>=0.0.1a0",
    "pre-commit>=3.7",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.25",
]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "ANN", "RUF"]
ignore = ["ANN101", "ANN102"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync --all-groups
```

Expected: `.venv/` created with all packages.

- [ ] **Step 4: Create minimal `pyreto/__init__.py`**

```python
"""Pyreto: Pareto, piecewise Pareto and generalized Pareto tools for reinsurance pricing.

Port of the R Pareto package by Ulrich Riegel (GPL >= 2).
See https://github.com/ulrichriegel/Pareto for the original.
"""

__version__ = "0.1.0"
```

- [ ] **Step 5: Verify import works**

```bash
uv run python -c "import pyreto; print(pyreto.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml pyreto/__init__.py .python-version
git commit -m "chore: initialise pyreto package with uv and pyproject.toml"
```

---

### Task 2: Set up ruff, ty, pre-commit, and GitHub Actions CI

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: ty
        name: ty type check
        entry: uv run ty check pyreto
        language: system
        pass_filenames: false
        types: [python]

      - id: pytest-fast
        name: pytest fast tests
        entry: uv run pytest tests/ -x -q --ignore=tests/test_fitting.py
        language: system
        pass_filenames: false
        stages: [pre-commit]
```

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [develop, main, "feature/**"]
  pull_request:
    branches: [develop, main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --all-groups

      - name: Lint with ruff
        run: uv run ruff check pyreto tests

      - name: Format check
        run: uv run ruff format --check pyreto tests

      - name: Type check with ty
        run: uv run ty check pyreto

      - name: Run tests
        run: uv run pytest tests/ --cov=pyreto --cov-report=term-missing
```

- [ ] **Step 3: Install pre-commit hooks**

```bash
uv run pre-commit install
```

- [ ] **Step 4: Commit**

```bash
mkdir -p .github/workflows
git add .pre-commit-config.yaml .github/workflows/ci.yml
git commit -m "chore: add ruff, ty, pre-commit, and GitHub Actions CI"
```

---

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

## Chunk 6: Collective Models

### Task 12: PPPModel

**Files:**
- Create: `pyreto/ppp_model.py`
- Create: `tests/test_ppp_model.py`

- [ ] **Step 1: Write failing tests (ported from `test_functions_PPP_Model.R`)**

```python
# tests/test_ppp_model.py
import numpy as np
import pytest
from pyreto.ppp_model import PPPModel
from pyreto.collective_model import (
    layer_mean, layer_var, layer_sd, excess_frequency, simulate_losses
)


class TestPPPModelCreation:
    def test_valid_model(self):
        model = PPPModel(
            fq=1.0,
            t=np.array([1000.0]),
            alpha=np.array([2.0]),
        )
        assert model.is_valid()

    def test_invalid_model_negative_fq(self):
        model = PPPModel(fq=-1.0, t=np.array([1000.0]), alpha=np.array([2.0]))
        assert not model.is_valid()

    def test_repr(self):
        model = PPPModel(fq=1.0, t=np.array([1000.0]), alpha=np.array([2.0]))
        assert "PPPModel" in repr(model)


class TestPPPModelLayerMean:
    def setup_method(self):
        ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
        el = np.array([1000, 900, 800, 600, 500], dtype=float)
        from pyreto.matching import piecewise_pareto_match_layer_losses
        self.model = piecewise_pareto_match_layer_losses(ap, el)
        self.cover = np.append(np.diff(ap), np.inf)
        self.ap = ap
        self.el = el

    def test_layer_means_match_targets(self):
        np.testing.assert_allclose(
            layer_mean(self.model, self.cover, self.ap), self.el, rtol=1e-8
        )

    def test_layer_var(self):
        result = layer_var(self.model, self.cover, self.ap)
        assert np.all(result >= 0)

    def test_excess_frequency_monotone(self):
        freqs = excess_frequency(self.model, np.array([1000, 2000, 3000, 4000, 5000]))
        assert np.all(np.diff(freqs) <= 0)

    def test_simulate_losses(self):
        losses = simulate_losses(self.model, nyears=100)
        assert isinstance(losses, list)
        assert len(losses) == 100
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_ppp_model.py -v
```

- [ ] **Step 3: Implement `pyreto/ppp_model.py`**

Port `PPPModel.R`. Use a Python dataclass:

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PPPModel:
    """Piecewise Pareto collective risk model.

    Parameters
    ----------
    fq : float or None
        Claim frequency (expected number of claims above t[0] per year).
    t : np.ndarray
        Threshold breakpoints of the piecewise Pareto.
    alpha : np.ndarray
        Shape parameters for each piece.
    truncation : float or None
        Upper truncation.
    truncation_type : str
        'lp' (last piece) or 'wd' (whole distribution).
    dispersion : float
        Overdispersion parameter for the claim count distribution (1 = Poisson).
    status : int
        0 = OK.
    comment : str
        Status message.
    """
    fq: float | None = None
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    truncation: float | None = None
    truncation_type: str = "lp"
    dispersion: float = 1.0
    status: int = 0
    comment: str = "OK"

    def is_valid(self, comment: bool = False) -> bool:
        """Return True if the model parameters are valid."""
        ...

    def __repr__(self) -> str:
        ...
```

Port layer loss/var/sd/frequency/simulate methods from `PPPModel.R` as standalone functions in `collective_model.py` (see Task 13).

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_ppp_model.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pyreto/ppp_model.py tests/test_ppp_model.py
git commit -m "feat: add PPPModel dataclass"
```

---

### Task 13: CollectiveModel protocol and dispatch functions

**Files:**
- Create: `pyreto/collective_model.py`
- Modify: `pyreto/pgp_model.py`
- Create: `tests/test_collective_model.py`

R uses S3 dispatch (`UseMethod`). Python equivalent: a `Protocol` defining the interface, with free functions that dispatch based on type.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_collective_model.py
import numpy as np
import pytest
from pyreto.collective_model import layer_mean, layer_var, layer_sd, excess_frequency, simulate_losses
from pyreto.ppp_model import PPPModel
from pyreto.pgp_model import PGPModel


class TestCollectiveModelDispatch:
    def test_layer_mean_ppp(self):
        from pyreto.matching import piecewise_pareto_match_layer_losses
        ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
        el = np.array([1000, 900, 800, 600, 500], dtype=float)
        model = piecewise_pareto_match_layer_losses(ap, el)
        cover = np.append(np.diff(ap), np.inf)
        np.testing.assert_allclose(layer_mean(model, cover, ap), el, rtol=1e-8)

    def test_layer_mean_ppp_scalar(self):
        from pyreto.matching import piecewise_pareto_match_layer_losses
        model = piecewise_pareto_match_layer_losses(
            np.array([1000, 2000], dtype=float),
            np.array([100, 100], dtype=float),
        )
        result = layer_mean(model, cover=np.inf, attachment_point=0)
        assert result > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_collective_model.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `pyreto/collective_model.py`**

```python
"""Collective model protocol and dispatch functions.

Mirrors CollectiveModelMethods.R. Provides free functions that dispatch
to PPPModel or PGPModel implementations.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike


@runtime_checkable
class CollectiveModel(Protocol):
    """Protocol for collective risk models (PPPModel, PGPModel)."""
    fq: float | None
    t: np.ndarray
    alpha: np.ndarray


def layer_mean(
    model: CollectiveModel,
    cover: ArrayLike = np.inf,
    attachment_point: ArrayLike = 0,
) -> np.ndarray:
    """Expected aggregate loss in a reinsurance layer.

    Dispatches to the appropriate implementation based on model type.
    """
    from pyreto.ppp_model import PPPModel
    from pyreto.pgp_model import PGPModel

    if isinstance(model, PPPModel):
        return _ppp_layer_mean(cover, attachment_point, model)
    elif isinstance(model, PGPModel):
        return _pgp_layer_mean(cover, attachment_point, model)
    raise TypeError(f"Unsupported model type: {type(model)}")

# ... layer_var, layer_sd, excess_frequency, simulate_losses follow same pattern
```

- [ ] **Step 3: Write failing PGP model tests**

```python
# tests/test_pgp_model.py
import numpy as np
import pytest
from pyreto.pgp_model import PGPModel
from pyreto.collective_model import layer_mean, layer_var, layer_sd, excess_frequency, simulate_losses


class TestPGPModelCreation:
    def test_valid_model(self):
        model = PGPModel(fq=1.0, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)
        assert model.is_valid()

    def test_invalid_negative_fq(self):
        model = PGPModel(fq=-1.0, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)
        assert not model.is_valid()

    def test_invalid_alpha_ini(self):
        model = PGPModel(fq=1.0, t=1000.0, alpha_ini=0.0, alpha_tail=2.0)
        assert not model.is_valid()

    def test_repr(self):
        model = PGPModel(fq=1.0, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)
        assert "PGPModel" in repr(model)


class TestPGPModelDispatch:
    def setup_method(self):
        self.model = PGPModel(fq=0.5, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)

    def test_layer_mean_positive(self):
        result = layer_mean(self.model, cover=9000.0, attachment_point=1000.0)
        assert float(result) > 0

    def test_layer_var_nonnegative(self):
        result = layer_var(self.model, cover=9000.0, attachment_point=1000.0)
        assert float(result) >= 0

    def test_layer_sd_equals_sqrt_var(self):
        v = layer_var(self.model, cover=9000.0, attachment_point=1000.0)
        sd = layer_sd(self.model, cover=9000.0, attachment_point=1000.0)
        assert float(sd) == pytest.approx(float(np.sqrt(v)))

    def test_excess_frequency_monotone(self):
        freqs = excess_frequency(self.model, x=np.array([1000, 2000, 3000, 5000], dtype=float))
        assert np.all(np.diff(freqs) <= 0)

    def test_simulate_losses_returns_list(self):
        result = simulate_losses(self.model, nyears=50)
        assert isinstance(result, list)
        assert len(result) == 50
```

- [ ] **Step 4: Run to verify failure**

```bash
uv run pytest tests/test_pgp_model.py -v
```

Expected: `ImportError`.

- [ ] **Step 5: Implement `pyreto/pgp_model.py`** (port `PGPModel.R`)

Similar to `PPPModel` but parameterised by `(t, alpha_ini, alpha_tail)` — see `PGPModel.R` for the dataclass fields and validation.

- [ ] **Step 6: Run all collective model tests**

```bash
uv run pytest tests/test_ppp_model.py tests/test_pgp_model.py tests/test_collective_model.py -v
```

Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add pyreto/collective_model.py pyreto/pgp_model.py tests/test_collective_model.py tests/test_pgp_model.py
git commit -m "feat: add PGPModel and CollectiveModel dispatch protocol"
```

---

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

## Chunk 8: Documentation

### Task 16: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`**

Cover: project overview, attribution to R package, installation (`pip install pyreto` / `uv add pyreto`), quick-start (p/d/q/r functions, layer mean, matching), link to full docs.

```bash
git add README.md
git commit -m "docs: add README with overview, installation, and quick-start"
```

---

### Task 17: mkdocs configuration and vignette

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/vignette.md`
- Create: `docs/api/pareto.md`, `docs/api/piecewise_pareto.md`, etc.

- [ ] **Step 1: Create `mkdocs.yml`**

```yaml
site_name: Pyreto
site_url: https://o1iv3r.github.io/Pyreto
repo_url: https://github.com/o1iv3r/Pyreto
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: true

nav:
  - Home: index.md
  - Vignette: vignette.md
  - Function Mapping: function_mapping.md
  - API Reference:
      - Pareto: api/pareto.md
      - PiecewisePareto: api/piecewise_pareto.md
      - GenPareto: api/gen_pareto.md
      - Models: api/models.md
      - Fitting: api/fitting.md
```

- [ ] **Step 2: Create `docs/vignette.md`**

Port the R vignette `Pareto.Rmd` to Python. Replace R code blocks with equivalent Python/pyreto code. Cover:
- Pareto distribution basics (CDF, PDF, quantile, random)
- Layer means and moments
- Alpha estimation
- PiecewisePareto
- Layer-loss matching (`piecewise_pareto_match_layer_losses`)
- GenPareto
- Collective models (PPPModel, PGPModel)

- [ ] **Step 3: Add GitHub Actions step for docs deployment**

Append to `.github/workflows/ci.yml`:

```yaml
  docs:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-groups
      - run: uv run mkdocs gh-deploy --force
```

- [ ] **Step 4: Build docs locally to verify**

```bash
uv run mkdocs build --strict
```

Expected: `site/` created without errors.

- [ ] **Step 5: Commit**

```bash
git add mkdocs.yml docs/ .github/workflows/ci.yml
git commit -m "docs: add mkdocs configuration, vignette, and API reference pages"
```

---

## Chunk 9: Final Integration

### Task 18: Full test suite run and quality gates

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All tests PASS.

- [ ] **Step 2: Check coverage**

```bash
uv run pytest tests/ --cov=pyreto --cov-report=term-missing
```

Target: ≥ 90% line coverage.

- [ ] **Step 3: Run ruff**

```bash
uv run ruff check pyreto tests
uv run ruff format --check pyreto tests
```

Expected: No issues.

- [ ] **Step 4: Run ty**

```bash
uv run ty check pyreto
```

Expected: No errors.

- [ ] **Step 5: Run pre-commit on all files**

```bash
uv run pre-commit run --all-files
```

- [ ] **Step 6: Commit any fixes**

```bash
git add -u
git commit -m "chore: fix linting and type check issues"
```

---

### Task 19: First release — merge develop → main

- [ ] **Step 1: Push develop to remote**

```bash
git push origin develop
```

- [ ] **Step 2: Create PR via GitHub CLI**

```bash
gh pr create \
  --base main \
  --head develop \
  --title "feat: initial Pyreto package — full port of R Pareto package" \
  --body "Complete Python port of the R Pareto package. See docs/design/implementation_plan.md for details."
```

- [ ] **Step 3: Wait for CI to pass, then merge**

```bash
gh pr merge --merge
```

- [ ] **Step 4: Tag v0.1.0**

```bash
git checkout main && git pull
git tag v0.1.0 -m "Initial release"
git push origin v0.1.0
```

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
