# Pyreto: Python methods and tools for the Pareto, piecewise Pareto and generalized Pareto distributions for reinsurance pricing

Port of the [Pareto R package](https://github.com/ulrichriegel/Pareto) by Ulrich Riegel (GPL ≥ 2) to Python.
Source R package: `/home/oliver/Code/Pareto`
GitHub: https://github.com/o1iv3r/Pyreto

## Functionality

- Feature parity with the R package
- Function names follow Python conventions (snake_case); e.g. `Pareto_Layer_Mean` → `pareto_layer_mean`. Provide a mapping table in the docs.
- Keep all functions with their corresponding arguments (adapted to Python conventions)
- Code improvements under the hood are encouraged (see Code Best Practices)

## Code Best Practices

- Proper Python package: SRP, DRY, KISS, SOLID
- Python ≥ 3.11
- Port all tests from the R package and verify numerical parity (use appropriate floating-point tolerances, e.g. `pytest.approx` or `np.testing.assert_allclose`)
- Ignore all R-specific artefacts (CRAN, `.Rbuildignore`, roxygen2 output, etc.)

### Dev toolstack

- `uv` (properly — use `uv sync`, `uv run`, workspaces; not legacy pip mode)
- `pyproject.toml`
- `ruff` (lint + format)
- `pytest`
- `ty` (type checking)
- `pre-commit` hooks covering: ruff (lint + format), ty (type check), pytest (fast tests only)

### Data science toolstack

- `numpy` — array operations and numerical computing
- `numba` — JIT compilation only where profiling identifies hot paths that need it
- `scipy` — statistical distributions, optimisation (replaces R's `lpSolve` dependency)
- `polars` — tabular data where needed

## Documentation

- `README.md` with project overview, installation, quick-start, and attribution to the original R package
- Full documentation via `mkdocs` (mkdocs-material theme) hosted on GitHub Pages
- The vignette lives as a dedicated mkdocs page (`docs/vignette.md`); the README links to it
- Docstrings in NumPy style throughout — functions should be self-describing for future MCP tool use
- GitHub Actions CI: run ruff, ty, and pytest on every push and pull request

## GitHub

- Repository: https://github.com/o1iv3r/Pyreto
- Use the github plugin and commit-commands skill for all version control operations
- Create a development and a main branch
- Package creation should happen in development
- Once the first version of the package is created, development should follow the git workflow, e.g., using feature branches of develop for development
- Once the work is done, develop is merged into main for the first release

## Suggested Claude Code plugins

- `context7` — up-to-date library documentation
- `github` + `commit-commands` — version control
- `claude-md-management` — keep this file and all documentation up to date
- `code-review` — review code before commits
- `superpowers` — advanced development skills (brainstorming, TDD, debugging, planning)
