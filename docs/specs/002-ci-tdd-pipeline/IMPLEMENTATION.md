# IMPLEMENTATION — Continuous Integration & Automated TDD Suite (FEAT-002)

| Metadata | Detail |
| :--- | :--- |
| **Feature** | Continuous Integration, Automated TDD Suite, and Fast Quality Gates |
| **Identifier** | `FEAT-002` |
| **Status** | Phase 2 — Implemented & Verified |
| **Branch** | `feature/002-ci-tdd-pipeline` |

---

## 1. Architectural Overview

`FEAT-002` modernizes LegislaML's dependency management, static analysis, and unit testing infrastructure without introducing lockfile bloat or GPU hardware dependencies in CI:

1. **Canonical Manifest (`pyproject.toml`)**:
   - Consolidates runtime dependencies (`torch`, `transformers`, `scikit-learn`, `xgboost`, `pandas`, `numpy`, `nltk`, `spacy`, `requests`, `beautifulsoup4`, `lxml`, `mlflow`, `joblib`) and development tools (`pytest`, `pytest-cov`, `ruff`) under `[dependency-groups] dev`.
   - Uses `[tool.uv] package = false` and `exclude-newer = "2026-11-09T00:00:00Z"` to guarantee reproducible dependency resolution across environments while keeping `uv.lock` ignored in `.gitignore` to avoid 7,000+ lines of lockfile noise in version control.
   - Preserves `requirements.txt` untouched for legacy compatibility.

2. **Zero-GPU Unit Test Harness (`tests/`)**:
   - `tests/fixtures/synthetic_statutes.py`: Implements `DeterministicMockTokenizer` (sequential integer mapping without non-deterministic `hash()`) and `MockBertBackbone` (`nn.Module` operating on CPU tensors of shape `(B, L, 768)`).
   - `tests/conftest.py`: Exposes reusable pytest fixtures for short, long (>510 tokens), and noisy statutory strings, plus synthetic DataFrames with duplicate statute IDs.
   - `tests/unit/`: Contains 5 modular test suites mapped 1-to-1 to the public seams specified in `SPEC.md`.

3. **Parallel GitHub Actions Pipeline (`.github/workflows/ci.yml`)**:
   - `lint-and-format`: Executes `uvx ruff check --config pyproject.toml src/ tests/` in `<5s`.
   - `unit-tests`: Installs CPU-only PyTorch wheels (`--extra-index-url https://download.pytorch.org/whl/cpu`), runs `pytest tests/unit/ --cov=src --cov-fail-under=85`, and publishes the coverage table to `$GITHUB_STEP_SUMMARY`.

---

## 2. File Inventory

| Path | Role |
| :--- | :--- |
| `pyproject.toml` | Canonical PEP 621 manifest + `uv`, `ruff`, `pytest`, and `coverage` configuration |
| `.gitignore` | Ignores `uv.lock`, `.venv`, `.coverage`, `.pytest_cache/`, and `.ruff_cache/` |
| `.github/workflows/ci.yml` | Multi-job CI pipeline for PRs targeting `develop`/`main` and pushes to `develop` |
| `tests/fixtures/synthetic_statutes.py` | Deterministic CPU mocks for tokenizer, BERTimbau backbone, and statutory texts |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/unit/test_dataset.py` | Seam 1: Head-Tail window (`128 + 382`) and special token contracts |
| `tests/unit/test_model.py` | Seam 2: `LegislaBERTClassifier`, `train_epoch`, `evaluate`, and `run_fine_tuning` |
| `tests/unit/test_weights.py` | Seam 3: Inverse-frequency class weights (`calculate_class_weights`) and `set_seed` |
| `tests/unit/test_deduplication.py` | Seam 4: Duplicate `id` stripping across DataFrame, classic model, and scraper |
| `tests/unit/test_preprocessing.py` | Seam 5: Regex cleaning, Flesch readability, and `vides` metadata parsing |
