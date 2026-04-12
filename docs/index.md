# Pyreto

Python port of the R [Pareto](https://github.com/ulrichriegel/Pareto) package by Ulrich Riegel. Provides Pareto, piecewise Pareto, and generalized Pareto distributions for reinsurance pricing.

## Overview

Pyreto provides:

- **Pareto distribution**: CDF, PDF, quantile, random sampling, layer moments, ML estimation, alpha-finding utilities
- **Piecewise Pareto distribution**: multi-segment Pareto with breakpoints; same suite of functions
- **Generalized Pareto distribution**: parameterized by `(t, alpha_ini, alpha_tail)`
- **LP layer-loss matching**: `piecewise_pareto_match_layer_losses` recovers a piecewise-Pareto model from observed layer expected losses
- **Collective models**: `PPPModel` (Piecewise Pareto) and `PGPModel` (Generalized Pareto) with layer mean/variance/simulation
- **Fitting utilities**: `local_pareto_alpha`, Panjer distribution (`d_panjer`, `r_panjer`), `fit_references`, `fit_pml_curve`

## Installation

```bash
pip install pyreto
# or
uv add pyreto
```

## Quick example

```python
import numpy as np
import pyreto

# Fit a piecewise-Pareto model to layer expected losses
ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
el = np.array([1000,  900,  800,  600,  500], dtype=float)
model = pyreto.piecewise_pareto_match_layer_losses(ap, el)

cover = np.append(np.diff(ap), np.inf)
pyreto.layer_mean(model, cover, ap)  # recovers el
```

See the [Vignette](vignette.md) for a complete tutorial.
