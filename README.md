# Pyreto

Python port of the R [Pareto](https://github.com/ulrichriegel/Pareto) package by Ulrich Riegel. Provides Pareto, piecewise Pareto, and generalized Pareto distributions for reinsurance pricing.

## Installation

```bash
pip install pyreto
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pyreto
```

## Quick start

```python
import numpy as np
import pyreto

# --- Pareto distribution ---
# CDF, PDF, quantile, random samples
pyreto.p_pareto(2000, t=1000, alpha=2)      # 0.75
pyreto.d_pareto(2000, t=1000, alpha=2)      # 5e-7
pyreto.q_pareto(0.75, t=1000, alpha=2)      # 2000.0
pyreto.r_pareto(5, t=1000, alpha=2)

# Layer moments
pyreto.pareto_layer_mean(1000, 1000, alpha=2)   # expected loss in 1000 xs 1000

# --- Piecewise Pareto ---
t = np.array([1000.0, 2000.0, 5000.0])
alpha = np.array([2.0, 1.5, 3.0])

pyreto.p_piecewise_pareto(3000, t=t, alpha=alpha)
pyreto.piecewise_pareto_layer_mean(5000, 1000, t=t, alpha=alpha)

# --- Generalized Pareto ---
pyreto.p_gen_pareto(3000, t=1000, alpha_ini=1.5, alpha_tail=2.0)

# --- LP layer-loss matching ---
ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
el = np.array([1000,  900,  800,  600,  500], dtype=float)
model = pyreto.piecewise_pareto_match_layer_losses(ap, el)
print(model)

# Verify: layer means recover the input
cover = np.append(np.diff(ap), np.inf)
pyreto.layer_mean(model, cover, ap)   # ≈ [1000, 900, 800, 600, 500]

# --- Collective model (PPPModel) ---
model = pyreto.PPPModel(fq=1.0, t=np.array([1000.0]), alpha=np.array([2.0]))
pyreto.layer_mean(model, cover=np.inf, attachment_point=1000)
pyreto.layer_var(model, cover=np.inf, attachment_point=1000)
pyreto.excess_frequency(model, np.array([1000.0, 2000.0]))
losses = pyreto.simulate_losses(model, nyears=100, seed=42)

# --- Fitting utilities ---
x = np.arange(1, 11) * 1e6
pyreto.local_pareto_alpha(x, "norm", mean=5e6, sd=2e6)
```

## Attribution

This package is a Python port of the R `Pareto` package (GPL >= 2) by Ulrich Riegel. The original R package is available at <https://github.com/ulrichriegel/Pareto> and on CRAN.

## License

GPL >= 2 (same as the original R package).
