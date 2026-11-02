# SDD Workflow Playbook (Spec & Architecture Agent)

Every feature in LegislaML follows a strict **Spec-Driven Development (SDD)** lifecycle inside `docs/features/XXX-<slug>/`.

## Directory Layout

```text
docs/features/XXX-<slug>/
├── SPEC.md            # Phase 1: Problem statement, requirements, math model, data contracts, trade-offs
├── PLAN.md            # Phase 1: Work Breakdown Structure (WBS), sequence, risks, contingencies
├── IMPLEMENTATION.md  # Phase 2: Module architecture, classes, hyperparameters, CLI interface
├── TESTS.md           # Phase 2: Tensor sanity checks, unit/integration test cases, Quality Gates (DoD)
└── QUICKSTART.md      # Phase 2: Reproducible CLI commands and verification steps
```

## Phase 1: Specification & Deliberation Gate

1. **Explore**: Inspect existing code, `GLOSSARY.md`, and `docs/adr/` to ground assumptions in repository facts.
2. **Draft**: Create only `SPEC.md` and `PLAN.md` in `docs/features/XXX-<slug>/`.
3. **Commit**: Stage and commit the specification atomically: `docs(feat-XXX): add phase 1 spec and plan for <slug>`.
4. **Completion Criterion (Hard Gate)**: Present the trade-offs, mathematical formulation, and quality gates to the Tech Lead, then **end the turn**. Zero production or test code files exist before explicit human sign-off.

## Phase 2: Execution & Verification Gate

Triggered only after the Tech Lead approves Phase 1:

1. **Test-First Implementation**: Implement the feature following TDD principles (`red` -> `green` -> `refactor`).
2. **Complete Documentation**: Write `IMPLEMENTATION.md`, `TESTS.md`, and `QUICKSTART.md` reflecting the verified implementation.
3. **Completion Criterion**: All automated tests pass, quality gates in `TESTS.md` are met, and documentation matches the exact CLI flags and tensor shapes in `src/`.
