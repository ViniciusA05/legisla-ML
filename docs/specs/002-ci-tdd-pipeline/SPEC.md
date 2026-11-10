# SPEC — Continuous Integration & Automated TDD Suite

| Metadata | Detail |
| :--- | :--- |
| **Feature** | Continuous Integration, Automated TDD Suite, and Fast Quality Gates |
| **Identifier** | `FEAT-002` |
| **Role** | ML Engineer / Tech Lead |
| **Status** | Phase 2 — Implemented & Verified |
| **Date** | 2026-11-04 |
| **Branch** | `feature/002-ci-tdd-pipeline` ➔ `develop` |

---

## 1. Problem Statement & Motivation

Following the successful implementation of `FEAT-001` (LegislaBERT Fine-Tuning and MLflow Governance), LegislaML v2 transitioned from shallow tabular models to PyTorch-based Deep Learning and multivariable loss formulations. However, the repository currently lacks an automated continuous integration harness and standardized test framework:

1. **Absence of Regression Shields**: Critical domain invariants—such as the statute deduplication rule (`drop_duplicates(subset=["id"])`) that eliminated the v1 data leakage—are currently enforced only by developer vigilance.
2. **Heavyweight CI Inefficiency**: Running naive end-to-end tests in CI that download the 435 MB `neuralmind/bert-base-portuguese-cased` weights would bloat runner times to several minutes and introduce external network failure points.
3. **Fragmented Tooling**: The repository relies on an outdated v1 `requirements.txt` without a lockfile or centralized linter configuration, risking transitive dependency drift between Windows/WSL2 and CI runners.

**FEAT-002** establishes a modern, high-performance CI/CD foundation powered by `uv` (Astral), `ruff`, and `pytest`. It introduces isolated unit testing with CPU-safe synthetic fixtures, parallel GitHub Actions jobs, and an enforceable $\ge 85\%$ test coverage quality gate.

---

## 2. System Requirements

### 2.1. Functional Requirements (FR)

- **FR01 (Canonical Project Manifest)**: Standardize project dependencies, build configurations, and tool settings in `pyproject.toml`, paired with `exclude-newer` timestamp cutoff in `[tool.uv]` for deterministic resolution without bloating git history.
- **FR02 (Multi-Job Parallel CI)**: Construct a GitHub Actions workflow (`.github/workflows/ci.yml`) triggering on:
  - Pull requests targeting `develop` and `main`;
  - Pushes to `develop`.
  The workflow runs two concurrent jobs on `ubuntu-latest` (Python 3.12):
  - **Job 1 (`lint-and-format`)**: Executes `uv run ruff check .` and `uv run ruff format --check .`.
  - **Job 2 (`unit-tests`)**: Executes `uv run pytest tests/unit/ --cov=src --cov-fail-under=85`.
- **FR03 (Public Seam Unit Test Harness)**: Implement isolated unit tests under `tests/unit/` backed by synthetic fixtures in `tests/fixtures/`:
  - **Seam 1 (`test_dataset.py`)**: `LegislaBERTDataset.__getitem__` enforcing Head-Tail window boundary ($128 + 382$ tokens), `[CLS]`/`[SEP]` positioning, and attention masks.
  - **Seam 2 (`test_model.py`)**: `LegislaBERTClassifier` forward contract ensuring output logits shape $(B, 2)$, CLS embedding shape $(B, 768)$, and absence of `NaN`/`Inf` using dummy tensor mocks without network downloads.
  - **Seam 3 (`test_weights.py`)**: `calculate_class_weights` inverse-frequency equation correctness, minority class prioritization ($w_1 > w_0$ when $N_1 < N_0$), and division-by-zero guards.
  - **Seam 4 (`test_deduplication.py`)**: Statutory scraper and dataset deduplication by `id` ensuring zero train-test data leakage.
  - **Seam 5 (`test_preprocessing.py`)**: Statutory text cleaning, regex normalization, and token counter integrity.
- **FR04 (Local Developer Experience Hook)**: Provide a lightweight git pre-commit hook script executing `ruff check` and `ruff format --check` in $< 1\text{s}$ before local commits.
- **FR05 (GitHub Step Summary)**: Export a clean Markdown table summarizing test suite results and coverage metrics into `$GITHUB_STEP_SUMMARY`.

### 2.2. Non-Functional & Resource Constraints (NFR)

- **NFR01 (Zero-GPU / Offline CI)**: All CI test suites must run strictly on standard CPU runners without requiring GPU hardware or downloading Hugging Face model weights at test time.
- **NFR02 (Execution SLA)**:
  - `lint-and-format` job execution time: $< 10\text{s}$.
  - `unit-tests` job execution time: $< 35\text{s}$.
  - Total end-to-end CI pipeline latency: $< 60\text{s}$.
- **NFR03 (Coverage Gate)**: Fail the CI pipeline if unit test coverage across testable business logic modules falls below **85%**.
- **NFR04 (Clean Repository Governance)**: All workflow files, test docstrings, and commit messages must strictly comply with `AGENTS.md` (English language, relative paths, Conventional Commits).

---

## 3. Mathematical & Data Invariant Contracts

### 3.1. Head-Tail Tokenization Contract
For any raw statutory sequence $T$ with token length $|T| > 510$:
$$\text{Tokens} = [t_{\text{CLS}}] \mathbin{\Vert} T[0:128] \mathbin{\Vert} T[-(510-128):] \mathbin{\Vert} [t_{\text{SEP}}]$$
$$\text{shape}(\text{input\_ids}) = (512,), \quad \text{shape}(\text{attention\_mask}) = (512,)$$
The unit test asserts that both the *Ementa* tokens (head) and final sanction/revocation tokens (tail) are strictly preserved within the $512$ limit.

### 3.2. Inverse-Frequency Class Weight Contract
Given class counts $N_0$ and $N_1$ with $N_{\text{total}} = N_0 + N_1$:
$$w_c = \frac{N_{\text{total}}}{2 \cdot N_c}, \quad c \in \{0, 1\}$$
The unit test asserts:
$$N_1 < N_0 \iff w_1 > w_0$$
$$\sum_{c=0}^1 w_c \cdot N_c = N_{\text{total}}$$

### 3.3. Dataset Idempotency & Data Leakage Prevention
For any raw collection of statute records $D$:
$$|D_{\text{clean}}| = |\text{unique}(\{d.\text{id} \mid d \in D\})|$$
The unit test asserts that applying deduplication over overlapping API queries eliminates $100\%$ of duplicated IDs.

---

## 4. Architectural Trade-Off Matrix

| Decision Area | Selected Option | Discarded Alternative | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Package Manager** | **`uv` (Astral) + `exclude-newer`** | Standard `pip` + `requirements.txt` | 10x-100x faster resolution written in Rust; timestamp-pinned resolution preventing dependency drift without 7k-line lockfile bloat. |
| **Linter & Formatter** | **`ruff`** | `flake8` + `black` + `isort` | Single dependency and unified configuration in `pyproject.toml`; sub-second execution speed. |
| **Model Testing in CI** | **CPU Mocks & Synthetic Fixtures** | Real BERTimbau Downloads | Prevents downloading 435 MB on every run, eliminating network flakiness and slashing CI time to $<30\text{s}$. |
| **Workflow Topology** | **Parallel Multi-Job** | Monolithic Single-Job Script | Enables parallel execution; provides distinct status badges for linting vs unit test failures in GitHub PRs. |
| **Coverage Threshold** | **$\ge 85\%$ Hard Gate** | No gate or $<70\%$ | Enforces high software engineering rigor, ensuring all core preprocessing and modeling seams remain tested. |
| **Local Pre-commit** | **Lightweight Ruff Hook** | Heavyweight Full-Test Hook | Keeps local `git commit` latency $<1\text{s}$ while delegating full test suite to CI or explicit commands. |

---

## 5. Quality Gates (Definition of Done)

| Criterion | Minimum Threshold | Target Goal | Failure Action |
| :--- | :--- | :--- | :--- |
| **Unit Test Pass Rate** | $100\%$ | $100\%$ | Block Pull Request merge. |
| **Code Coverage (`pytest-cov`)** | $\ge 85\%$ | $\ge 90\%$ | Fail CI build via `--cov-fail-under=85`. |
| **Ruff Lint & Format Check** | 0 errors / 0 warnings | Clean baseline | Fail `lint-and-format` job. |
| **Total CI Execution Time** | $< 90\text{s}$ | $< 45\text{s}$ | Optimize caching or fixture initialization. |
| **Deterministic Install** | `uv sync` (`exclude-newer`) | Zero build compilation | Enforce reproducible cutoff in `pyproject.toml`. |
