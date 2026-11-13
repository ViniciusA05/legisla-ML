# IMPLEMENTATION — LegislaBERT Model Serving (FastAPI + Docker)

| Metadata | Detail |
| :--- | :--- |
| **Feature** | Containerized CPU inference microservice for the fine-tuned LegislaBERT classifier |
| **Identifier** | `FEAT-003` |
| **Status** | Phase 2 — Implemented (see `TESTS.md` for the verification status of each gate) |
| **Branch** | `feat/003-model-serving` ➔ `develop` |

---

## 1. Module Map

```text
src/
├── legislabert.py            # head_tail_encode, LegislaBERTDataset, LegislaBERTClassifier (torch + transformers only)
├── train_legislabl_bert.py   # imports and re-exports the two classes; training logic untouched
└── serving/
    ├── __init__.py           # SERVICE_VERSION, VALID_TARGETS
    ├── settings.py           # Settings.from_env(): typed env parsing + validation
    ├── schemas.py            # Pydantic request/response/error models
    ├── checkpoint.py         # load_slim_checkpoint(), CheckpointError, sha256_of_file()
    ├── inference.py          # Predictor, Prediction (no web dependency)
    ├── export.py             # export_slim_checkpoint() + CLI
    └── app.py                # create_app(bundle_factory, settings), build_default_bundle(), module-level `app`
scripts/
├── fetch_tokenizer.py        # bakes tokenizer + config (never weights) at image build
├── smoke_container.sh        # CI smoke test (liveness, readiness, non-root)
└── bench_serving.py          # latency benchmark against a running service
Dockerfile, .dockerignore
tests/unit/serving/           # test_encoding, test_checkpoint, test_inference, test_api
```

---

## 2. Shared Core (`src/legislabert.py`)

### `head_tail_encode(tokenizer, text, max_length=512, head_tokens=128, tail_tokens=382)`
Returns `(input_ids, attention_mask, n_tokens, truncated)`; both lists have length `max_length`.

- `usable = max_length - 2`; if `n_tokens > usable` the window is `tokens[:head_tokens] + tokens[-(usable - head_tokens):]`.
- Layout: `[CLS] + window + [SEP] + [PAD]*k`; mask is `1` for real tokens (including `[CLS]`/`[SEP]`) and `0` for padding.
- The logic was moved **verbatim** from `LegislaBERTDataset.__getitem__`; the dataset now delegates to this function, so training and serving cannot drift (invariant I1).

### `LegislaBERTClassifier`
- New optional `backbone` argument: when provided, `AutoModel.from_pretrained` is skipped.
- New `from_config(config_source, num_classes=2, dropout_prob=0.3)`: builds a randomly initialised backbone with `AutoModel.from_config(AutoConfig.from_pretrained(config_source))`. `config_source` is a local directory (serving) or a model name; **no weights are downloaded**.
- `forward(input_ids, attention_mask) -> (logits, cls_embedding)` is unchanged.

### Backward compatibility
`src/train_legislabl_bert.py` imports both classes and declares `__all__` including `AutoModel` and `AutoTokenizer`. The FEAT-002 tests patch `bert_mod.AutoModel.from_pretrained` (an attribute of the shared `transformers.AutoModel` class), so they run **unmodified**.

---

## 3. Checkpoint Contract

### Slim checkpoint schema (written by `export.py`, read by `checkpoint.py`)
```text
{
  "model_state_dict": {name: CPU contiguous tensor},
  "config":           {"model_name": str, "max_length": 512},
  "metrics":          plain-Python dict (NumPy scalars converted)
  "target":           "teve_veto" | "instavel",
  "sha256_of_source": hex digest of the training checkpoint
}
```

### `export_slim_checkpoint(src, dst, target)`
- Reads the training checkpoint with `weights_only=False` (own, trusted artifact that embeds NumPy scalar metrics).
- Drops `optimizer_state_dict`; sanitises `metrics` and `config` to plain types so the slim file loads with `weights_only=True`.
- Raises `ValueError` for an unknown `target`, `CheckpointError` when `model_state_dict` or `config` is missing.

### `load_slim_checkpoint(path, model_factory, expected_target=None)`
Validation order (every failure raises `CheckpointError`):
1. file exists → 2. `torch.load(map_location="cpu", weights_only=True)` succeeds → 3. payload is a dict with keys `model_state_dict`, `config`, `target` → 4. `config.max_length == 512` → 5. `target` valid and equal to `expected_target` when set → 6. `load_state_dict(strict=True)` on the model built by `model_factory()`.

On success the model is switched to `eval()` and returned in `LoadedCheckpoint(model, target, model_name, max_length, sha256, metrics)`, where `sha256` is the digest of the served file.

---

## 4. Inference Path (`src/serving/inference.py`)

`Predictor(model, tokenizer, threshold=0.5, max_batch=16)`:

| Method | Behaviour |
| :--- | :--- |
| `predict_one(text)` | `predict_batch([text])[0]` |
| `predict_batch(texts)` | splits into chunks of `max_batch`, preserves order, returns `[]` for an empty list |
| `warmup()` | one forward pass on `"warmup"` |

Per chunk: `head_tail_encode` for each text → `torch.tensor` stack `(B, 512)` → `model(...)` under `torch.inference_mode()` → `softmax(logits.float(), dim=-1)` → `label = int(p1 >= threshold)` (the `>=` boundary is inclusive and tested).
`Prediction(label, score_positive, scores, threshold, truncated, n_tokens)` is a frozen dataclass; `score_positive == scores[1]`.

The constructor calls `model.eval()`, so a model handed over in training mode cannot serve with active dropout (I6).

---

## 5. HTTP Service (`src/serving/app.py`)

### Routes
| Method | Path | Success | Errors |
| :--- | :--- | :--- | :--- |
| GET | `/healthz` | `200 {"status":"ok"}` always | — |
| GET | `/readyz` | `200 {"status":"ready"}` | `503 model_not_ready` |
| GET | `/v1/model` | `200 ModelInfo` | `503 model_not_ready` |
| POST | `/v1/predict` | `200 PredictionOut` | `422 invalid_input`, `503` |
| POST | `/v1/predict/batch` | `200 {"predictions":[...]}` | `422 invalid_input` / `batch_too_large`, `503` |

Error envelope for every non-2xx: `{"error": {"code": str, "message": str}}`. Codes in use: `invalid_input`, `batch_too_large`, `model_not_ready`, `internal`.

### Lifecycle
`create_app(bundle_factory, settings)` loads the model inside the FastAPI lifespan: optional `torch.set_num_threads` → `bundle_factory(settings)` → `predictor.warmup()` → readiness flag set.

> **Deviation from SPEC FR04 wording ("aborts at startup").** A load failure is **logged and the process stays alive but never ready** (`/healthz = 200`, `/readyz = 503`). This is required by quality gate QG5 (the container started without a checkpoint must answer `healthz = 200` and `readyz = 503`) and is the correct Kubernetes-style semantics. The intent of I5 is preserved: a bad checkpoint can never serve traffic. `SPEC.md` was updated to say so.

### Handlers
- `ApiError` ➔ its own status/code; `RequestValidationError` ➔ `422 invalid_input` listing only field names; any other exception ➔ `500 internal` with a fixed message. Request text is **never** logged or echoed.
- Handlers are plain `def`, so FastAPI runs them in its threadpool; one Uvicorn worker; intra-op parallelism bounded by `TORCH_NUM_THREADS`.

### `build_default_bundle(settings)`
`AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)` → `load_slim_checkpoint(checkpoint_path, model_factory=LegislaBERTClassifier.from_config(tokenizer_dir), expected_target=settings.target)` → `Predictor` → `ModelInfo` (`device` is always `"cpu"`).

---

## 6. Configuration (`Settings.from_env`)

| Variable | Default | Validation |
| :--- | :--- | :--- |
| `LEGISLA_CHECKPOINT_PATH` | `/models/model.pt` | — |
| `LEGISLA_TARGET` | unset | one of `teve_veto`, `instavel`; must match the checkpoint |
| `LEGISLA_TOKENIZER_DIR` | `/opt/legisla/tokenizer` | — |
| `DECISION_THRESHOLD` | `0.5` | `0 < τ < 1` |
| `MAX_BATCH` | `16` | `>= 1` |
| `MAX_CHARS` | `500000` | `>= 1` |
| `TORCH_NUM_THREADS` | `0` (library default) | `>= 0`; applied only when `> 0` |
| `PORT` | `8000` | read by the container `CMD` |

Invalid values raise `SettingsError` at import time of `src.serving.app`, so a mis-set container fails immediately.

---

## 7. Container Image

- **Builder** (`python:3.12-slim` + pinned `uv 0.13.7`): `uv venv /opt/venv` then `uv pip install --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --group serving`. `--group` installs **only** the `serving` dependency group (no `spacy`, `xgboost`, `mlflow`, `nltk`). A dry run resolved `torch==2.15.0+cpu` with no CUDA wheels. `scripts/fetch_tokenizer.py` bakes tokenizer + config into `/opt/legisla/tokenizer`.
- **Runtime** (`python:3.12-slim`): copies the venv, tokenizer, and `src/`; runs as non-root `uid 10001`; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; `HEALTHCHECK` on `/healthz`; `CMD uvicorn src.serving.app:app --port ${PORT}` with a single worker.
- `.dockerignore` excludes `.git`, `data/`, `models/`, `*.pt`, `tests/`, `docs/`, notebooks, and caches.
- Model weights are **mounted** (`-v <host>/model.pt:/models/model.pt:ro`), never baked in.

## 8. CI (`.github/workflows/ci.yml`)

- `lint-and-format`: `ruff check` now also covers `scripts/`.
- `unit-tests`: installs `--group dev --group serving`; the existing `--cov-fail-under=85` gate now includes `src/serving/`.
- `docker-build` (new, parallel): Buildx build with GitHub Actions layer cache → `scripts/smoke_container.sh` → image size printed to `$GITHUB_STEP_SUMMARY`.

## 9. Repository Hygiene
- `.gitignore`: `models/` and `*.pt` are ignored.
- `pyproject.toml`: new `serving` dependency group; `httpx` added to `dev` (required by `TestClient`).
