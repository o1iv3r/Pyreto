# Pyreto Vignette

This vignette is a Python translation of the R `Pareto` package vignette by Ulrich Riegel. It covers the main distribution families and the LP layer-loss matching algorithm.

## Setup

```python
import numpy as np
import pyreto
```

---

## 1. Pareto distribution

The Pareto distribution is parameterized by threshold `t` (minimum value) and shape `alpha` (tail index). The survival function is:

```
S(x) = (t/x)^alpha    for x >= t
```

```python
# CDF at x=2000 with t=1000, alpha=2
pyreto.p_pareto(2000, t=1000, alpha=2)     # 0.75

# PDF
pyreto.d_pareto(2000, t=1000, alpha=2)

# Quantile (inverse CDF)
pyreto.q_pareto(0.75, t=1000, alpha=2)    # 2000.0

# Random samples
rng = np.random.default_rng(42)
pyreto.r_pareto(10, t=1000, alpha=2)
```

### Layer moments

```python
# Expected loss in layer: 1000 xs 1000 (cover=1000, attachment=1000)
pyreto.pareto_layer_mean(1000, 1000, alpha=2, t=1000)

# Second moment
pyreto.pareto_layer_sm(1000, 1000, alpha=2, t=1000)

# Variance
pyreto.pareto_layer_var(1000, 1000, alpha=2, t=1000)
```

### Alpha estimation

```python
# Find alpha such that layer mean = target
alpha = pyreto.pareto_find_alpha_btw_layers(
    cover1=1000, ap1=1000, ell1=500,
    cover2=2000, ap2=2000, ell2=300,
)
```

---

## 2. Piecewise Pareto distribution

A piecewise Pareto distribution is a multi-segment Pareto with breakpoints `t` and shape parameters `alpha` (one per segment).

```python
t = np.array([1000.0, 2000.0, 5000.0])
alpha = np.array([2.0, 1.5, 3.0])

# CDF
pyreto.p_piecewise_pareto(3000, t=t, alpha=alpha)

# Layer mean
pyreto.piecewise_pareto_layer_mean(5000, 1000, t=t, alpha=alpha)
```

### LP layer-loss matching

Given observed attachment points and expected layer losses, recover the piecewise-Pareto parameters:

```python
ap = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
el = np.array([1000,  900,  800,  600,  500], dtype=float)

model = pyreto.piecewise_pareto_match_layer_losses(ap, el)
print(model)

# Verify: layer means recover the targets
cover = np.append(np.diff(ap), np.inf)
pyreto.layer_mean(model, cover, ap)

# With truncation
model_lp = pyreto.piecewise_pareto_match_layer_losses(
    ap, el, truncation=10_000, truncation_type="lp"
)
model_wd = pyreto.piecewise_pareto_match_layer_losses(
    ap, el, truncation=10_000, truncation_type="wd"
)
```

---

## 3. Generalized Pareto distribution

The generalized Pareto distribution is parameterized by `(t, alpha_ini, alpha_tail)`. When `alpha_ini == alpha_tail` it reduces to the standard Pareto.

```python
pyreto.p_gen_pareto(3000, t=1000, alpha_ini=1.5, alpha_tail=2.0)
pyreto.gen_pareto_layer_mean(9000, 1000, t=1000, alpha_ini=1.5, alpha_tail=2.0)
```

---

## 4. Collective models

### PPPModel — Panjer & Piecewise Pareto

```python
model = pyreto.PPPModel(
    fq=1.0,
    t=np.array([1000.0, 2000.0]),
    alpha=np.array([2.0, 1.5]),
)
print(model.is_valid())

# Expected aggregate loss in a layer
pyreto.layer_mean(model, cover=1000, attachment_point=1000)

# Variance
pyreto.layer_var(model, cover=1000, attachment_point=1000)

# Excess frequency
pyreto.excess_frequency(model, x=np.array([1000.0, 2000.0, 5000.0]))

# Simulate annual loss realisations
losses = pyreto.simulate_losses(model, nyears=100, seed=42)
# losses is a list of 100 lists, each containing individual claim severities
```

### PGPModel — Panjer & Generalized Pareto

```python
model = pyreto.PGPModel(fq=0.5, t=1000.0, alpha_ini=1.5, alpha_tail=2.0)
pyreto.layer_mean(model, cover=9000, attachment_point=1000)
```

---

## 5. Fitting utilities

### Local Pareto alpha

The local Pareto alpha measures the tail heaviness at a point:

```python
x = np.arange(1, 11) * 1e6
pyreto.local_pareto_alpha(x, "norm", mean=5e6, sd=2e6)
pyreto.local_pareto_alpha(x, "lnorm", meanlog=0, sdlog=4)
pyreto.local_pareto_alpha(x, "Pareto", t=1e6, alpha=1, truncation=20e6)
```

### Panjer distribution

```python
# PMF: Poisson(2)
pyreto.d_panjer(3, mean=2.0, dispersion=1.0)

# PMF: NegBin (dispersion > 1)
pyreto.d_panjer(3, mean=3.0, dispersion=2.0)

# Random samples
rng = np.random.default_rng(42)
pyreto.r_panjer(1000, mean=3.0, dispersion=1.5, rng=rng)
```
