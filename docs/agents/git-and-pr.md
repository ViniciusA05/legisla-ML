# Git, Conventional Commits & Pull Request Playbook

Version control and release discipline for LegislaML.

## 1. Branching Model

- **`main`**: Production-ready releases only. Direct commits are prohibited; update exclusively via reviewed Pull Requests from `develop`.
- **`develop`**: Active integration branch. All work branches off and merges back into `develop`.
- **Working Branches**: Create from `develop` using `<type>/<kebab-case-slug>`:
  - `feat/<slug>`: New capabilities or model architectures.
  - `fix/<slug>`: Bug fixes or pipeline corrections.
  - `docs/<slug>`: Specification or documentation updates.
  - `chore/<slug>`: Governance, dependencies, or tooling.
  - `ci/<slug>`: GitHub Actions and automated verification pipelines.

## 2. Atomic Conventional Commits

Every commit represents a single logical unit of change written in imperative English:

```text
<type>(<scope>): <imperative-summary>
```

- **Allowed `<type>`**: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`.
- **Allowed `<scope>`**: `feat-XXX`, `scraper`, `preprocessing`, `features`, `model`, `mlops`, `agents`, `ci`.
- **Separation of Concerns**: Commit Phase 1 SDD specifications (`docs(feat-XXX): ...`) separately from Phase 2 implementation (`feat(...)` / `test(...)`).

## 3. Pull Request Standard & Post-Merge Cleanup

1. **PR Creation**: Open Pull Requests targeting `develop` using `.github/pull_request_template.md`.
2. **Immediate Branch Pruning**: As soon as a branch is merged into `develop` (or `main`), delete both the remote and local branches and prune tracking references:
   ```bash
   git checkout develop
   git pull origin develop
   git branch -d <branch-name>
   git push origin --delete <branch-name>
   git fetch --prune
   ```
