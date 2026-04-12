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

