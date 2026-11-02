# AGENTS.md — LegislaML Repository Governance

LegislaML is an AI and MLOps platform for predictive quality scoring, semantic diffing, and anomaly detection on Brazilian federal legislation.

## Universal Invariants

1. **Language**: Write all repository artifacts (code, docstrings, comments, SDD specs, ADRs, commit messages, and pull requests) in **English**. Preserve Portuguese exclusively for raw legal corpora, NLP tokenizers (`BERTimbau`/`spaCy`), and domain literals mapped in `GLOSSARY.md`.
2. **Privacy & Path Hygiene**: Use repository-relative paths (`src/...`, `data/...`, `docs/...`) and functional role titles (`Tech Lead`, `ML Engineer`) across all tracked files. Keep personal names, local OS paths, and machine-specific hardware models out of version control.
3. **Two-Phase SDD Gate**: Every feature (`FEAT-XXX`) lives in `docs/features/XXX-<slug>/` and executes in two gated phases:
   - **Phase 1 (Specification)**: Draft only `SPEC.md` and `PLAN.md`, commit atomically (`docs(feat-XXX): ...`), and **halt execution** until the Tech Lead explicitly approves the spec.
   - **Phase 2 (Execution)**: After approval, implement code via TDD and complete `IMPLEMENTATION.md`, `TESTS.md`, and `QUICKSTART.md`.

## Specialized Agent Playbooks (Context Pointers)

Load the target playbook on demand when your task enters its branch:

- **Specifying, planning, or implementing a feature (`FEAT-XXX`)**: Read `docs/agents/sdd-workflow.md` for the 5-artifact contracts (`SPEC`, `PLAN`, `IMPLEMENTATION`, `TESTS`, `QUICKSTART`) and quality gates.
- **Designing models, PyTorch training loops, NLP pipelines, or MLOps tracking**: Read `docs/agents/ml-engineering.md` for multivariable optimization standards, Head+Tail truncation, memory-constrained CUDA defaults (`fp16`, gradient accumulation), and MLflow instrumentation.
- **Branching, committing, opening PRs, or merging**: Read `docs/agents/git-and-pr.md` for the `develop` integration flow, Conventional Commits grammar, PR verification, and immediate post-merge branch pruning.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`GLOSSARY.md` at root + `docs/adr/`). See `docs/agents/domain.md`.
