# SPEC — LegislaBERT Model Serving (FastAPI + Docker)

| Metadata | Detail |
| :--- | :--- |
| **Feature** | Containerized CPU inference microservice for the fine-tuned LegislaBERT classifier |
| **Identifier** | `FEAT-003` |
| **Role** | ML Engineer / Tech Lead |
| **Status** | Phase 2 — Implemented & Verified |
| **Date** | 2026-11-12 |
| **Branch** | `feat/003-model-serving` ➔ `develop` |
| **Depends on** | `FEAT-001` (fine-tuned checkpoint), `FEAT-002` (CI + TDD harness) |

---

## 1. Problem Statement & Motivation

`FEAT-001` produces a checkpoint (`models/best_legislabl_bert_<target>.pt`) that reaches Macro-F1 `0.7340` and veto Recall `0.5000` on the deduplicated corpus, but the model is only reachable through the training script. Three gaps block any consumer (the v3 semantic-diff and RAG modules, or an external client):

1. **No inference contract.** There is no stable, versioned interface that maps raw statutory text to a prediction. Callers would have to import `src/train_legislabl_bert.py`, which pulls `mlflow`, `pandas`, and `scikit-learn` into any inference process.
2. **Training/serving skew risk.** The Head-Tail Window (`128 + 382` tokens) is implemented only inside `LegislaBERTDataset.__getitem__`. A second, hand-copied implementation in a serving layer would silently drift and degrade predictions without raising an error.
3. **Non-reproducible runtime.** The model runs only on the developer machine (CUDA, Windows/WSL2). There is no container image, no health semantics, and no CI evidence that the model can be loaded and queried in a clean environment.

**FEAT-003** closes Sprint 0 of the v2 → v3 plan by delivering a **single-model, CPU-first FastAPI microservice** packaged as a Docker image, with a shared encoding path to eliminate skew, a deterministic checkpoint-loading contract, and CI gates that run without GPU or model weights.

### 1.1. Non-Goals (explicitly out of scope)

- GPU serving, TensorRT/ONNX export, and int8 quantization (recorded as future work in §7).
- MLflow Model Registry as the artifact source (the checkpoint is loaded from a mounted file in v1 of the service; registry integration is deferred, see §6 D2).
- Authentication, rate limiting, and public exposure. The service is an internal microservice behind a trusted network.
- Serving both targets (`teve_veto`, `instavel`) from one process. One container serves one checkpoint (see §6 D3).
- Semantic diffing, embeddings search, and RAG (Sprints 1–3 of v3). The `cls_embedding` endpoint is **not** included (see §6 D7).

---

## 2. System Requirements

### 2.1. Functional Requirements (FR)

- **FR01 (Shared Encoding Path)**: Extract the Head-Tail Window logic into a pure function `head_tail_encode(...)` in a lightweight module (`src/legislabert.py`) that depends only on `torch` and `transformers`. `LegislaBERTDataset` and the serving layer both call it. No second implementation may exist.
- **FR02 (Light Model Module)**: Move `LegislaBERTClassifier` and `LegislaBERTDataset` into `src/legislabert.py`. `src/train_legislabl_bert.py` re-exports both names so every existing import and every FEAT-002 test keeps working unchanged.
- **FR03 (Offline Model Construction)**: Add `LegislaBERTClassifier.from_config(model_name, num_classes)` building the BERT backbone from `AutoConfig` (random init, **no weight download**), then loading the fine-tuned `model_state_dict` over it. Backbone weights shipped by Hugging Face are never needed at serving time because the checkpoint overwrites every parameter.
- **FR04 (Checkpoint Contract)**: Load a checkpoint dict with keys `model_state_dict` and `config = {model_name, max_length}` (the exact schema written by `run_fine_tuning`). Validate before serving:
  - missing keys, `max_length != 512`, or a `state_dict` that fails `strict=True` loading ➔ raises a typed `CheckpointError` so the lifespan keeps the service in not-ready state (`/healthz = 200`, `/readyz = 503`, never serving traffic with a half-loaded model);
  - `optimizer_state_dict` is ignored and released immediately.
- **FR05 (Checkpoint Export CLI)**: Provide `python -m src.serving.export --src <ckpt.pt> --dst <slim.pt> --target <teve_veto|instavel>` that strips the optimizer state and writes `{model_state_dict, config, metrics, target, sha256_of_source}`. Slim artifacts are the only files mounted into the container.
- **FR06 (Inference Endpoints)**:
  - `POST /v1/predict` — body `{"text": str}` ➔ single prediction.
  - `POST /v1/predict/batch` — body `{"texts": [str, ...]}` with `1 <= len <= MAX_BATCH` (default `16`) ➔ ordered list of predictions.
  - Response per item (see §3.3): `label`, `score_positive`, `scores`, `threshold`, `truncated`, `n_tokens`.
- **FR07 (Operational Endpoints)**:
  - `GET /healthz` — liveness, always `200` while the process is up.
  - `GET /readyz` — readiness, `200` only after the model is loaded and a warm-up forward pass completed; otherwise `503`.
  - `GET /v1/model` — metadata: `target`, `model_name`, `max_length`, `head_tokens`, `tail_tokens`, `checkpoint_sha256`, `val_metrics` (from the checkpoint), `device`, `service_version`.
- **FR08 (Configuration via Environment)**: All runtime settings come from environment variables parsed by a single typed settings object (see §3.4). No setting is read from code constants at call sites.
- **FR09 (Input Validation)**: Reject with `422` empty/whitespace-only text and text longer than `MAX_CHARS` (default `500_000`). Reject batches over `MAX_BATCH` with `422`. Errors use a single JSON error schema `{"error": {"code": str, "message": str}}`.
- **FR10 (Container Image)**: A multi-stage `Dockerfile` producing a CPU-only image that runs as a **non-root** user, bakes the **tokenizer and config** (not the weights) at build time, and sets `HF_HUB_OFFLINE=1` at runtime so the container never reaches the network.
- **FR11 (CI Integration)**: Extend `.github/workflows/ci.yml` with a `docker-build` job (build + smoke test, no push) and keep existing jobs and the `≥ 85%` coverage gate green with the new modules included.

### 2.2. Non-Functional & Resource Constraints (NFR)

| ID | Constraint | Target | How it is verified |
| :--- | :--- | :--- | :--- |
| NFR01 | Offline tests | Unit tests download nothing and use no GPU | CI runs without `HF_TOKEN`; network-free fixtures |
| NFR02 | Single-request latency (512 tokens, 4 vCPU) | **p95 ≤ 3.0 s** (target; measured value is recorded in `TESTS.md`) | Benchmark script `scripts/bench_serving.py` (Phase 2) |
| NFR03 | Memory footprint | Resident set ≤ **1.5 GB** after warm-up (fp32 weights ≈ 440 MB + activations) | `docker stats` during benchmark |
| NFR04 | Image size | ≤ **2.0 GB** uncompressed | `docker image ls` recorded in `TESTS.md` |
| NFR05 | Cold start | `/readyz` returns `200` within **≤ 30 s** of container start with a mounted slim checkpoint | Smoke script |
| NFR06 | Determinism | Same text ➔ bit-identical logits across requests in one process (`model.eval()`, `torch.inference_mode()`) | Unit test |
| NFR07 | Security hygiene | Non-root user, no secrets or weights baked into the image, `.dockerignore` excluding `data/`, `models/`, `.git/` | Image inspection step in CI |
| NFR08 | Governance | English artifacts, relative paths, Conventional Commits, no personal data | `AGENTS.md` review |
| NFR09 | Coverage | Total coverage `≥ 85%` including `src/serving/` | `--cov-fail-under=85` |

> **Note on NFR02/NFR03/NFR04.** These are engineering targets, not measured facts. `TESTS.md` (Phase 2) must report the **actual** numbers; if a target is missed, the report states the gap and the follow-up (e.g., int8 quantization), rather than relaxing the target silently.

---

## 3. Contracts & Mathematical Model

### 3.1. Inference Computation Graph

For input text $T$ the service computes, in this exact order:

1. **Tokenize**: $\mathbf{t} = \text{tokenizer}(T, \text{add\_special\_tokens}=\text{False})$, with $|\mathbf{t}| = n$.
2. **Head-Tail Window** (identical to training): with budget $B = L_{\max} - 2 = 510$, head $H = 128$, tail $B - H = 382$:

$$
\mathbf{s} =
\begin{cases}
\mathbf{t}[0:H] \mathbin{\Vert} \mathbf{t}[-(B-H):] & n > B \\
\mathbf{t} & n \le B
\end{cases}
\qquad
\mathbf{x} = [\text{CLS}] \mathbin{\Vert} \mathbf{s} \mathbin{\Vert} [\text{SEP}] \mathbin{\Vert} [\text{PAD}]^{L_{\max}-|\mathbf{s}|-2}
$$

   with attention mask $m_i = \mathbb{1}[x_i \ne \text{PAD}]$ (positional, as in training). The response field `truncated` is $\mathbb{1}[n > B]$ and `n_tokens` is $n$.
3. **Forward** (dropout disabled by `model.eval()`): $\mathbf{h} = \text{BERT}(\mathbf{x}, m)_{[\text{CLS}]} \in \mathbb{R}^{768}$, logits $\mathbf{z} = W\mathbf{h} + \mathbf{b} \in \mathbb{R}^{2}$.
4. **Score**: $p_c = \text{softmax}(\mathbf{z})_c = \dfrac{e^{z_c}}{\sum_k e^{z_k}}$; computed in fp32 with the max-subtraction stabilization built into `torch.softmax`.
5. **Decision**: $\hat{y} = \mathbb{1}[p_1 \ge \tau]$ with default $\tau = 0.5$ (configurable via `DECISION_THRESHOLD`).

### 3.2. What `score_positive` Means (and What It Does Not)

`FEAT-001` trains with $\text{CE}_w = -\sum_c w_c\, \mathbb{1}[y=c]\log p_c$ and $w_c = \dfrac{N}{C\, N_c}$. Minimizing a class-weighted cross-entropy makes the network learn the posterior of a **re-weighted** distribution:

$$
p'(y \mid x) \;\propto\; w_y\, p(y \mid x)
\;\;\Longrightarrow\;\;
p(y \mid x) \;\propto\; \pi_y \, p'(y \mid x), \qquad \pi_y = \dfrac{N_y}{N}
$$

Consequences, which this spec makes contractual:

- `score_positive` $= p'_1$ is a **balanced-prior ranking score**, not a calibrated probability of veto in the real legislative population. The API field is therefore named `score_positive` (never `probability`), and `GET /v1/model` documents this.
- With $\pi_1 \approx 0.2$ (veto class is the minority), recovering the real-prior posterior would **lower** the score. A $\tau = 0.5$ threshold on $p'_1$ is consistent with the Macro-F1 / minority-recall objective the model was selected on, so it is the default.
- Threshold tuning on validation PR curves is a Sprint 2 concern; this feature exposes `DECISION_THRESHOLD` but ships `0.5`.

### 3.3. Data Contracts (API Schemas)

```text
POST /v1/predict
  request : { "text": string }                       # 1..MAX_CHARS chars after strip
  response: {
    "label": 0 | 1,
    "score_positive": float,                         # p'_1 in [0, 1]
    "scores": [float, float],                        # [p'_0, p'_1], sums to 1 within 1e-6
    "threshold": float,
    "truncated": bool,                               # n_tokens > 510
    "n_tokens": int                                  # tokenizer length before windowing
  }

POST /v1/predict/batch
  request : { "texts": [string, ...] }               # 1..MAX_BATCH items
  response: { "predictions": [<same object as above>, ...] }   # order preserved

GET /v1/model
  response: {
    "service_version": string,
    "target": "teve_veto" | "instavel",
    "model_name": string,
    "max_length": 512, "head_tokens": 128, "tail_tokens": 382,
    "checkpoint_sha256": string,
    "val_metrics": { "f1_macro": float, "recall_veto": float, ... },
    "device": "cpu",
    "score_semantics": "balanced-prior ranking score; not a calibrated probability"
  }

Error envelope (all non-2xx):
  { "error": { "code": "invalid_input" | "batch_too_large" | "model_not_ready" | "internal", "message": string } }
```

### 3.4. Configuration Contract (environment variables)

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `LEGISLA_CHECKPOINT_PATH` | `/models/model.pt` | Slim checkpoint produced by FR05 |
| `LEGISLA_TARGET` | *(read from checkpoint)* | If set, must equal the checkpoint's `target`, else startup aborts |
| `LEGISLA_TOKENIZER_DIR` | `/opt/legisla/tokenizer` | Tokenizer baked at image build |
| `DECISION_THRESHOLD` | `0.5` | $\tau$ in §3.1 |
| `MAX_BATCH` | `16` | Max items per batch request |
| `MAX_CHARS` | `500000` | Max characters per text |
| `TORCH_NUM_THREADS` | `0` (= `os.cpu_count()`) | Passed to `torch.set_num_threads` |
| `PORT` | `8000` | Uvicorn port |

### 3.5. Invariants (each backed by a test in Phase 2)

- **I1 (No skew)**: for any text, `head_tail_encode` called by the dataset and by the service returns identical `input_ids` and `attention_mask`.
- **I2 (Parity)**: logits from the service inference path equal logits from the training-time `LegislaBERTClassifier.forward` for the same weights and input, within `atol=1e-5`.
- **I3 (Normalization)**: $\sum_c p_c = 1 \pm 10^{-6}$ and $p_c \in [0,1]$ for all outputs; no `NaN`/`Inf`.
- **I4 (Batch/single equivalence)**: predicting texts individually or as a batch yields the same labels and scores within `atol=1e-5` (padding must not leak through the attention mask).
- **I5 (Fail fast)**: a checkpoint with a missing key, wrong `max_length`, or mismatched `state_dict` shape aborts startup; `/readyz` never returns `200` in that state.
- **I6 (Eval mode)**: repeated identical requests return identical scores (dropout inactive).
- **I7 (Backward compatibility)**: every FEAT-002 test passes unmodified after the extraction refactor (FR01–FR02).

---

## 4. Architecture Overview

```text
src/
├── legislabert.py            # NEW: head_tail_encode, LegislaBERTDataset, LegislaBERTClassifier (torch + transformers only)
├── train_legislabl_bert.py   # MODIFIED: imports + re-exports the names above; training logic untouched
└── serving/                  # NEW
    ├── __init__.py
    ├── settings.py           # typed env parsing (stdlib dataclass or pydantic-settings)
    ├── schemas.py            # Pydantic request/response/error models
    ├── checkpoint.py         # load_slim_checkpoint(): validation + from_config + state_dict (FR03, FR04)
    ├── inference.py          # Predictor: encode -> forward -> softmax -> decision (no FastAPI import)
    ├── app.py                # create_app(predictor_factory) + lifespan + routes + error handlers
    └── export.py             # CLI: strip optimizer, hash source, write slim checkpoint (FR05)

Dockerfile                    # NEW: multi-stage, CPU-only, non-root
.dockerignore                 # NEW
tests/unit/serving/           # NEW: tests per seam (see §5)
```

**Layering rule:** `inference.py` has zero web dependencies, so the model path is unit-testable without FastAPI; `app.py` receives a `Predictor` through a factory so tests inject a fake or a mock-backbone predictor.

**Concurrency model:** route handlers are plain `def` (FastAPI runs them in its threadpool), one Uvicorn worker, and a `threading.Lock` is **not** used because `torch.inference_mode()` forward passes on a shared eval-mode module are thread-safe for read-only inference; intra-op parallelism is bounded by `torch.set_num_threads`. Horizontal scale is by running more containers, not more workers (each worker would duplicate ≈ 440 MB of weights).

---

## 5. Quality Gates (Definition of Done)

| Gate | Criterion |
| :--- | :--- |
| QG1 | All FEAT-002 tests pass **unmodified** (I7) |
| QG2 | Invariants I1–I6 each covered by at least one test |
| QG3 | Total coverage `≥ 85%`, with `src/serving/` individually `≥ 85%` |
| QG4 | CI wall-clock for `unit-tests` stays `< 90 s` (up from 49 s; budget for the extra suite) |
| QG5 | `docker build` succeeds in CI; container with **no** checkpoint answers `/healthz = 200` and `/readyz = 503` |
| QG6 | Local run with a real slim checkpoint: `/readyz = 200` within NFR05, and benchmark numbers for NFR02–NFR04 recorded in `TESTS.md` |
| QG7 | `ruff check` clean; no personal names or machine-specific paths in tracked files |

---

## 6. Design Decisions & Trade-offs (require Tech Lead sign-off)

| # | Decision | Chosen | Alternatives rejected | Rationale / trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | Where shared model code lives | New `src/legislabert.py`; training module re-exports | (a) import from the training module in the service; (b) duplicate code | (a) drags `mlflow`/`pandas`/`sklearn` into the image (+hundreds of MB) and couples serving to training; (b) guarantees skew. Cost: a refactor touching a tested file, mitigated by re-exports and I7. |
| **D2** | Checkpoint source | Slim file mounted at runtime (`LEGISLA_CHECKPOINT_PATH`) | (a) bake weights in image; (b) pull from MLflow Registry at startup | (a) 440 MB layer per model version, slow pushes, couples release to retrain; (b) adds a network/tracking-server dependency and an auth surface before there is a registry-promotion workflow. Cost: operator must provision the volume. Registry loading is a documented follow-up. |
| **D3** | One target per container | Yes (`teve_veto` **or** `instavel`) | One process, two models | Halves memory and keeps readiness semantics simple. Cost: two deployments if both targets are needed. |
| **D4** | Model construction | `AutoConfig` + `from_config` + `load_state_dict(strict=True)` | `from_pretrained` then overwrite | Avoids a 435 MB download at startup and makes offline containers possible. Cost: relies on the checkpoint covering all parameters, enforced by `strict=True` (I5). |
| **D5** | Score semantics | Expose `score_positive`, document non-calibration, `τ = 0.5` | Return `probability`; or apply prior correction | Honest naming prevents downstream misuse (e.g., treating 0.5 as 50% veto chance). Calibration deferred to Sprint 2 with a gold-standard sample. |
| **D6** | CPU-only image | Yes, PyTorch CPU wheels (same index strategy as CI) | CUDA base image | Target environment is a plain container host; CUDA image is ≈ 4× larger and needs the NVIDIA runtime. Cost: slower inference; acceptable for an internal, low-QPS service. |
| **D7** | Embedding endpoint | **Not** in this feature | `POST /v1/embed` returning the 768-d CLS vector | Sprint 2 may prefer a different encoder (Ulysses-BERT / JurisBERT); locking an embedding API to this classifier now would create a contract we may break. |
| **D8** | Dependency declaration | New `serving` dependency group (`fastapi`, `uvicorn[standard]`, `pydantic`, plus runtime `torch`, `transformers`, `numpy`); Docker installs `--only-group serving` | Add FastAPI to the default dependencies | Keeps the training environment unchanged and the image minimal (no `spacy`, `xgboost`, `mlflow`, `nltk`). Cost: tests need `--group serving` in CI, added in PLAN. |
| **D9** | Branch naming | `feat/003-model-serving` | `feature/003-model-serving` (FEAT-002 precedent) | Follows `docs/agents/git-and-pr.md` (`feat/<slug>`). Approved by the Tech Lead; the FEAT-002 `feature/` prefix is a legacy deviation. |

---

## 7. Risks & Open Questions

| Risk / Question | Impact | Mitigation |
| :--- | :--- | :--- |
| `torch.load(weights_only=True)` may reject the checkpoint because it embeds a nested `metrics` dict (including a `classification_report` with NumPy scalar types) | Startup failure | Phase 2 first task: load the real checkpoint with `weights_only=True`; if it fails, `export.py` (which runs in a trusted context) sanitizes `metrics` to plain Python types, so the serving path keeps `weights_only=True` (no arbitrary-pickle execution in the container). |
| `models/` is not gitignored; a `.pt` file could be committed | 440 MB–1.3 GB repo bloat | Add `models/` and `*.pt` to `.gitignore` (Phase 2, Step 1). |
| The real checkpoint is not present in this clone (`data/processed/` is empty, `models/` absent) | QG6 cannot be verified in CI | CI uses a mock-backbone predictor; QG6 is a **local** gate executed with the real artifact, and its output is pasted into `TESTS.md`. |
| CPU latency may exceed NFR02 with a 512-token input | Poor UX for sync callers | Record the measurement; contingencies (dynamic int8 quantization, `torch.compile`, `MAX_BATCH` reduction) listed in `PLAN.md`. |
| Checkpoint stores `target` only in the filename, not in the dict | Wrong model served for a target | `export.py` requires an explicit `--target` and writes it into the slim file; `LEGISLA_TARGET` mismatch aborts startup. |
| `torch` CPU wheel size may break NFR04 (≤ 2.0 GB) | Image too large | Multi-stage build, `--no-cache`, strip `__pycache__`, drop unused extras; if still over, relax NFR04 explicitly with data. |

**Open questions for the Tech Lead:**

1. Confirm **D2** (mounted slim checkpoint, no MLflow Registry in this feature).
2. Confirm **D1** (extract `src/legislabert.py` and re-export), accepting a refactor of a tested module.
3. Confirm **D5** (`score_positive` naming and `τ = 0.5` default).
4. Are the NFR targets (p95 ≤ 3 s, RSS ≤ 1.5 GB, image ≤ 2 GB) acceptable as targets to be measured and reported?
