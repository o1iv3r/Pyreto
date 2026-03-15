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

