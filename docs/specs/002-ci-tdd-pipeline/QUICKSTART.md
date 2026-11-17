# QUICKSTART — Running Linter, Unit Tests, and Coverage Locally (FEAT-002)

---

## 1. Environment Setup (WSL2 Native Recommended)

Running inside the native WSL2 directory (`~/projects/LegislaML`) provides >20x faster filesystem I/O than `/mnt/c/`:

```bash
cd ~/projects/LegislaML

# Create or update the local virtual environment with CPU PyTorch wheels
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r pyproject.toml --group dev
```

---

## 2. Static Analysis (Ruff Linter)

```bash
# Check lint and import sorting rules across src/ and tests/
.venv/bin/ruff check --config pyproject.toml src/ tests/

# Auto-fix safe lint issues (imports, pyupgrade)
.venv/bin/ruff check --fix --config pyproject.toml src/ tests/
```

---

## 3. Running Unit Tests & Coverage Gate

```bash
# Execute the full unit test suite with the 85% coverage gate
PYTHONPATH=. .venv/bin/pytest tests/unit/ --cov=src --cov-fail-under=85
```

---

## 4. Optional Local Git Pre-Commit Hook

To enforce sub-second lint verification automatically before every `git commit`, create `.git/hooks/pre-commit`:

```bash
cat << 'EOF' > .git/hooks/pre-commit
#!/usr/bin/env bash
set -e
uvx ruff check --config pyproject.toml src/ tests/
EOF
chmod +x .git/hooks/pre-commit
```
