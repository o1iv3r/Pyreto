# Pyreto

> **Warning:** This package is in alpha. It is intended for learning and experimentation only — do not use it in production.

Pyreto is a Python library for modelling large insurance losses, built around the Pareto family of heavy-tailed distributions. It is a port of the R [Pareto](https://github.com/ulrichriegel/Pareto) package by Ulrich Riegel.

In reinsurance pricing, the central question is: given that a loss has already exceeded some threshold, how large is it likely to be — and what does that mean for a specific layer of cover? The Pareto distribution is the natural answer. It is heavy-tailed (large losses are far more probable than a normal distribution would suggest), and it has a simple scale-invariance property: above any attachment point, the excess looks Pareto again. This makes it the standard severity model for excess-of-loss (XL) reinsurance.

## Installation

```bash
pip install pyreto
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pyreto
```

## Core concepts

### Excess-of-loss layers

A reinsurance layer is described by two numbers: the **attachment point** (where the reinsurer's liability begins) and the **cover** (how much the reinsurer pays above that point). A *1000 xs 2000* layer pays losses between 2000 and 3000. The expected loss in a layer is the primary pricing quantity.

### The Pareto distribution

The Pareto distribution has two parameters: a threshold `t` (the minimum possible loss) and a shape `alpha` (the tail index — smaller alpha means heavier tail and more catastrophic losses). It is parameterised so that `t` is exactly the attachment point of your ground-up model.

```python
import numpy as np
import pyreto

# What fraction of losses from a Pareto(t=1000, alpha=2) are below 2000?
pyreto.p_pareto(2000, t=1000, alpha=2)      # 0.75

# Loss density at 2000
pyreto.d_pareto(2000, t=1000, alpha=2)      # 0.00025

# What loss corresponds to the 75th percentile?
pyreto.q_pareto(0.75, t=1000, alpha=2)      # 2000.0

# Simulate five losses
pyreto.r_pareto(5, t=1000, alpha=2)
```

### Layer pricing: expected loss in a layer

Once you have a severity model, the key pricing output is the **layer mean** — the expected amount paid by the layer per ground-up loss.

```python
# Unlimited cover above 1000: E[loss in unlimited xs 1000]
pyreto.pareto_layer_mean(cover=np.inf, attachment_point=1000, alpha=2)  # 1000.0   (unlimited xs 1000)

# Finite layer: E[loss in 1000 xs 5000]
pyreto.pareto_layer_mean(cover=1000,   attachment_point=5000, alpha=2)  # 833.33   (1000 xs 5000)
```

### Piecewise Pareto: when one alpha isn't enough

A single Pareto fits well in the body but may not match observed loss behaviour across the full range. The **piecewise Pareto** uses a different `alpha` in each segment, defined by a vector of breakpoints `t`. This is the workhorse model for multi-layer programmes where you need a consistent but flexible severity curve.

```python
t     = np.array([1000.0, 2000.0, 5000.0])
alpha = np.array([2.0,    1.5,    3.0])

# CDF at 3000 under the piecewise model
pyreto.p_piecewise_pareto(3000, t=t, alpha=alpha)

# Expected loss in the 5000 xs 1000 layer
pyreto.piecewise_pareto_layer_mean(5000, 1000, t=t, alpha=alpha)
```

### Generalized Pareto: smoothly varying tail

The **generalized Pareto** interpolates continuously between an initial shape `alpha_ini` (near the threshold) and an asymptotic tail shape `alpha_tail`. It is useful when you believe the tail gradually becomes heavier (or lighter) rather than jumping abruptly between segments.

```python
pyreto.p_gen_pareto(3000, t=1000, alpha_ini=1.5, alpha_tail=2.0)  # 0.84
```

### Fitting a model to observed layer losses

In practice you often have benchmark expected losses for several layers (from market data or historical experience), and you need a severity model that reproduces them all consistently. `piecewise_pareto_match_layer_losses` solves this as a linear programme and returns the piecewise Pareto that best matches the data.

```python
ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)   # attachment points
el = np.array([1000,  900,  800,  600,  500], dtype=float)    # expected losses per layer

model = pyreto.piecewise_pareto_match_layer_losses(ap, el)

# Verify: layer means recover the input
cover = np.append(np.diff(ap), np.inf)
pyreto.layer_mean(model, cover, ap)   # ≈ [1000, 900, 800, 600, 500]
```

### Collective model: from severity to aggregate losses

The **PPPModel** (Panjer & Piecewise Pareto) combines a Poisson frequency with a piecewise Pareto severity to model the full aggregate loss distribution of a portfolio. This is the natural next step once you have a calibrated severity model: you can price any layer on an aggregate basis, compute variance, estimate exceedance frequencies, or simulate years of losses.

```python
model = pyreto.PPPModel(fq=1.0, t=np.array([1000.0]), alpha=np.array([2.0]))

pyreto.layer_mean(model, cover=np.inf, attachment_point=1000)   # expected aggregate loss
pyreto.layer_var(model,  cover=np.inf, attachment_point=1000)   # variance
pyreto.excess_frequency(model, np.array([1000.0, 2000.0]))      # exceedance frequencies

losses = pyreto.simulate_losses(model, nyears=100, seed=42)     # year-by-year simulation
```

### Fitting alpha locally

When you have an empirical loss distribution (or a parametric severity assumption), `local_pareto_alpha` estimates the effective Pareto alpha at each point in a grid. This is useful for diagnostics and for building an intuition of how tail-heaviness varies across the loss range.

```python
x = np.arange(1, 11) * 1e6
pyreto.local_pareto_alpha(x, "norm", mean=5e6, sd=2e6)
```

## Attribution

This package is a Python port of the R `Pareto` package (GPL >= 2) by Ulrich Riegel. The original R package is available at <https://github.com/ulrichriegel/Pareto> and on CRAN.

## License

GPL >= 2 (same as the original R package).
