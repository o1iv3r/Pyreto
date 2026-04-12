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
        python-version: ["3.11", "3.12", "3.13"]

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

