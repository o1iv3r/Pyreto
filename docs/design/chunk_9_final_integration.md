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

