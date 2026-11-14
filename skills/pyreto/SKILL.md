---
name: pyreto
description: Price excess-of-loss reinsurance layers with the Pyreto Python library - fit Pareto / piecewise Pareto collective models to layer losses, excess frequencies or PML curves, then compute layer means, variances, excess frequencies and simulations.
---

# Pyreto

Use this when a task involves pricing reinsurance XL layers, fitting a severity curve to
market references, or Pareto alpha calculations between layers.

1. Install from GitHub, never from PyPI (the PyPI `pyreto` is another package):
   `uv add "pyreto @ git+https://github.com/o1iv3r/Pyreto"`.
2. Write and run Python against `import pyreto` rather than doing the maths by hand.
3. Read the agent guide before the first call: `docs/agents.md` in the repo, or
   https://o1iv3r.github.io/Pyreto/agents/. Use `help(pyreto.<function>)` for arguments.
4. After every fit (`fit_references`, `piecewise_pareto_match_layer_losses`,
   `fit_pml_curve`) check `model.status`: 0 = OK, 1 = some references ignored,
   2 = failed (reason in `model.comment`; downstream results would be `nan`).
5. Report layers as `cover xs attachment_point` and say whether figures are per loss
   (severity-only functions) or annual (collective-model functions).
