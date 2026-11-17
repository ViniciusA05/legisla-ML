# TESTS — LegislaBERT Model Serving (FastAPI + Docker)

| Metadata | Detail |
| :--- | :--- |
| **Feature** | `FEAT-003` |
| **Status** | Phase 2 — Implemented & Verified (QG1–QG7) |
| **Command** | `PYTHONPATH=. .venv/bin/pytest tests/unit --cov=src --cov-fail-under=85` |

---

## 1. Result Summary

| Metric | Value |
| :--- | :--- |
| Tests | **70 passed** (13 from FEAT-002 unmodified + 57 new) |
| Total coverage | **92.15%** (gate ≥ 85%) |
| Coverage `src/serving/` | `app.py` 100%, `inference.py` 100%, `schemas.py` 100%, `settings.py` 100%, `checkpoint.py` 96%, `export.py` 90% |
| `src/legislabert.py` | 96% |
| Local wall-clock (WSL2, 12 vCPU) | ≈ 14–16 s for the full suite |
| Lint | `ruff check src tests scripts` — all checks passed |

Baseline before this feature (recorded first): 13 passed, 88.74% coverage.

---

## 2. Invariant Coverage (SPEC §3.5)

| Invariant | Test(s) |
| :--- | :--- |
| **I1** No skew | `test_encoding.py::test_dataset_delegates_to_shared_function_without_skew`, `test_head_tail_encode_*` (boundaries `0, 1, 509, 510, 511, 1000`; head/tail slices; custom window) |
| **I2** Parity with training forward | `test_inference.py::test_parity_with_training_forward` (`abs=1e-5`) |
| **I3** Normalisation, finiteness | `test_inference.py::test_scores_are_normalized_and_finite` |
| **I4** Batch ≡ single | `test_inference.py::test_batch_matches_single`, `test_batch_preserves_order_and_chunks_by_max_batch` |
| **I5** Fail fast | `test_checkpoint.py`: missing file, unreadable file, missing keys, wrong `max_length`, `state_dict` mismatch, target mismatch; `test_api.py::test_failed_model_load_keeps_liveness_but_never_ready` |
| **I6** Eval mode / determinism | `test_inference.py::test_deterministic_across_calls_and_eval_mode`, `test_predictor_forces_eval_mode` |
| **I7** FEAT-002 compatibility | All 13 original tests pass with **zero edits** to existing test files (`git diff --stat tests/` shows no modified files, only the new `tests/unit/serving/`) |

## 3. Test Inventory (new)

| File | Count | Focus |
| :--- | :---: | :--- |
| `tests/unit/serving/test_encoding.py` | 9 | Head-Tail function, window math, dataset delegation |
| `tests/unit/serving/test_checkpoint.py` | 12 | Export (slim schema, optimizer stripped, metrics sanitised, `weights_only=True` loadable, CLI), round-trip bit-equality, every failure mode |
| `tests/unit/serving/test_inference.py` | 11 | Parity, normalisation, batching, determinism, threshold boundary (`p1 == τ ⇒ 1`), custom τ, truncation flags |
| `tests/unit/serving/test_api.py` | 25 | Response contract, order, `/v1/model`, validation matrix (422), readiness semantics, 500 envelope without input echo, thread setting, offline `build_default_bundle`, `Settings` parsing and rejection |

All unit tests are network-free, GPU-free, and weight-free (mock backbone `d = 768`, deterministic mock tokenizer, fake predictors).

---

## 4. Quality Gates (Definition of Done)

| Gate | Criterion | Status | Evidence |
| :--- | :--- | :---: | :--- |
| QG1 | FEAT-002 tests pass unmodified | ✅ verified | 13/13 pass; no existing test file modified |
| QG2 | I1–I6 each covered | ✅ verified | §2 |
| QG3 | Coverage ≥ 85% total and for `src/serving/` | ✅ verified | 92.15% total; every serving module ≥ 90% |
| QG4 | CI `unit-tests` < 90 s | ✅ verified in CI | PR #7 run: `Unit Tests & Coverage (>=85%)` finished in **50 s** (baseline before this feature: 49 s) |
| QG5 | `docker build` succeeds; no-checkpoint container gives `healthz 200` / `readyz 503` | ✅ verified in CI | PR #7 `Docker Build & Smoke Test` job: `healthz=200 readyz=503`, `Smoke test passed (non-root uid=10001)`. Docker is unavailable in the local WSL distro, so the first real build happened in CI. |
| QG6 | Real slim checkpoint: readiness, parity, benchmark | ✅ verified | Trained `models/best_legislabl_bert_teve_veto.pt` (epoch 4) exported to `models/slim_teve_veto.pt` (`435,766,230 B`, `weights_only=True`, `strict=True`). Scored all `36` held-out validation statutes via `scripts/verify_parity.py` (`TestClient` -> `/readyz = 200`, `POST /v1/predict/batch`): Macro-F1 `0.733990` (`abs_diff_checkpoint = 0.0`, `abs_diff_expected = 9.85e-6 <= 0.002` per ADR-0003). Latency & RSS verified in §5. |
| QG7 | Lint clean, no personal data or machine paths | ✅ verified | `ruff` clean; tracked files use repo-relative paths |

---

## 5. Serving Validation with the Real Architecture (QG6, partial)

Setup: real `neuralmind/bert-base-portuguese-cased` tokenizer and config (offline, `HF_HUB_OFFLINE=1`), full 110 M-parameter `LegislaBERTClassifier` built with `from_config`, **random weights** exported through the slim CLI, `uvicorn` with `TORCH_NUM_THREADS=4` on a 12 vCPU WSL2 host. Latency and memory do not depend on weight values, so they are valid; accuracy is not measurable this way.

| Measurement | Target (SPEC) | Measured | Verdict |
| :--- | :--- | :--- | :--- |
| `/readyz` after launch | `200` ≤ 30 s | `200` on the first probe (taken ≈ 20 s after launch; exact cold-start time not instrumented) | consistent with target, not precisely measured |
| Single request p50 / p95 / p99 (900-word statute, truncated to 512 tokens, 30 requests, sequential) | p95 ≤ 3.0 s | **0.588 s / 0.705 s / 0.796 s** | ✅ met |
| Batch of 16 (one request) | — | **9.07 s** (≈ 0.57 s/item; no batch speed-up on CPU) | informational |
| Resident memory after warm-up | ≤ 1.5 GB | **≈ 1.0 GB** (1,017,480 KB) | ✅ met |
| Slim checkpoint size | — | 435,750,784 B (≈ 416 MiB) | informational |
| Image size | ≤ 2.0 GB | **pending readout** (`ci.yml` updated to `tee` output to job logs as well as `$GITHUB_STEP_SUMMARY`) | ⏳ inspect next `docker-build` CI log/summary or local Docker build |
| Strict loading of a full-architecture state dict via `from_config` | I5 | succeeded | ✅ |
| Slim file loads with `weights_only=True` | SPEC risk R1 | succeeded (NumPy-free after sanitisation) | ✅ |

Caveats: single host, single run, loopback client, `p99` from only 30 samples.

---

## 6. Trained Checkpoint Serving Parity (QG6, verified 2026-11-17)

Command (offline `TestClient`, `HF_HUB_OFFLINE=1`, `TORCH_NUM_THREADS=4`, `seed=42`, `threshold=0.5`, `batch_size=16`):

```bash
.venv/bin/python -m src.serving.export \
  --src models/best_legislabl_bert_teve_veto.pt \
  --dst models/slim_teve_veto.pt \
  --target teve_veto

.venv/bin/python scripts/verify_parity.py \
  --checkpoint models/slim_teve_veto.pt \
  --tokenizer-dir .local/tokenizer \
  --data-path data/processed/leis_clean.csv \
  --features-path data/processed/leis_features.csv \
  --target teve_veto \
  --threads 4
```

| Metric / Artifact | Training Checkpoint (Epoch 4) | Serving Stack (`POST /v1/predict/batch`) | Verdict |
| :--- | :---: | :---: | :---: |
| Validation split size (`N`) | `36` | `36` | ✅ exact |
| **Macro-F1** | `0.7339901477832512` | **`0.7339901477832512`** | ✅ `Δ = 0.0` (`≤ 0.002`) |
| Accuracy | `0.8333333333333334` | `0.8333333333333334` | ✅ exact (`30/36`) |
| Balanced Accuracy | `0.7142857142857143` | `0.7142857142857143` | ✅ exact |
| Precision (`teve_veto=1`) | `0.6666666666666666` | `0.6666666666666666` | ✅ exact (`4/6`) |
| Recall (`teve_veto=1`) | `0.5000` | `0.5000` | ✅ exact (`4/8`) |
| Wall-clock (36 statutes, CPU) | — | `25.757 s` | informational |
| Slim checkpoint size | `1,302,639,074 B` (raw `.pt`) | `435,766,230 B` (`~415.6 MiB`) | ✅ stripped optimizer |
| Source SHA-256 | `d4cfd6bf0adc429c0f9f3c715b4ee219fe8351e774a462af9d1224250103336b` | verified in slim metadata | ✅ |
| Slim SHA-256 | — | `a32477d77b9f0e682426f3448407cebb67b057b541c3d9b67655aa72ecbb8c5e` | ✅ |

---

## 7. Follow-up Operational Notes (Docker Host)

1. Read the uncompressed image size from the `docker-build` CI log / step summary (`.github/workflows/ci.yml` now pipes `docker image ls` through `tee -a "$GITHUB_STEP_SUMMARY"`).
2. On a host with Docker available, `scripts/verify_parity.py --url http://127.0.0.1:8000` and `scripts/bench_serving.py --url http://127.0.0.1:8000` (`docker stats`) can optionally be re-run against the mounted container.
