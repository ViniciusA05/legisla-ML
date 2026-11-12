# QUICKSTART — LegislaBERT Model Serving

Reproducible commands for `FEAT-003`. Run everything from the repository root.

## 1. Run the tests

```bash
uv venv .venv
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r pyproject.toml --group dev --group serving

PYTHONPATH=. .venv/bin/pytest tests/unit --cov=src --cov-fail-under=85
.venv/bin/ruff check src tests scripts
```

Expected: `70 passed`, coverage above 85%.

## 2. Export a slim checkpoint

The training checkpoint is produced by `src/train_legislabl_bert.py` as `models/best_legislabl_bert_<target>.pt`. Strip the optimizer state and tag the target:

```bash
PYTHONPATH=. .venv/bin/python -m src.serving.export \
  --src models/best_legislabl_bert_teve_veto.pt \
  --dst models/slim_teve_veto.pt \
  --target teve_veto
```

`models/` and `*.pt` are git-ignored; never commit checkpoints.

## 3. Run the container

```bash
docker build -t legislaml-serving:local .

docker run --rm -p 8000:8000 \
  -v "$(pwd)/models/slim_teve_veto.pt:/models/model.pt:ro" \
  -e LEGISLA_TARGET=teve_veto \
  legislaml-serving:local
```

Smoke test without a checkpoint (expects `healthz=200 readyz=503`, non-root):

```bash
bash scripts/smoke_container.sh legislaml-serving:local
```

## 4. Run without Docker (development)

Fetch the tokenizer and config once, then start the server offline:

```bash
PYTHONPATH=. .venv/bin/python scripts/fetch_tokenizer.py --out .local/tokenizer

HF_HUB_OFFLINE=1 PYTHONPATH=. \
LEGISLA_CHECKPOINT_PATH=models/slim_teve_veto.pt \
LEGISLA_TOKENIZER_DIR=.local/tokenizer \
LEGISLA_TARGET=teve_veto \
  .venv/bin/uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
```

## 5. Call the API

```bash
curl -s localhost:8000/readyz
curl -s localhost:8000/v1/model

curl -s -X POST localhost:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"text": "Art. 1o Esta Lei institui a politica nacional de dados abertos."}'

curl -s -X POST localhost:8000/v1/predict/batch \
  -H 'content-type: application/json' \
  -d '{"texts": ["Art. 1o ...", "Art. 2o ..."]}'
```

Response of `/v1/predict`:

```json
{
  "label": 1,
  "score_positive": 0.55,
  "scores": [0.45, 0.55],
  "threshold": 0.5,
  "truncated": false,
  "n_tokens": 16
}
```

> **`score_positive` is a balanced-prior ranking score, not a calibrated probability.** The model was trained with inverse-frequency class weights, so `0.55` does **not** mean a 55% chance of veto in the real legislative population. Use it to rank and compare statutes; the default decision threshold is `0.5`.

Error shape (all non-2xx): `{"error": {"code": "invalid_input", "message": "..."}}`.

## 6. Benchmark

```bash
.venv/bin/python scripts/bench_serving.py --url http://127.0.0.1:8000 --requests 50
```

Prints p50/p95/p99 for a 512-token request and the total time of a 16-item batch.

## 7. Configuration

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `LEGISLA_CHECKPOINT_PATH` | `/models/model.pt` | slim checkpoint |
| `LEGISLA_TARGET` | unset | if set, must match the checkpoint target |
| `LEGISLA_TOKENIZER_DIR` | `/opt/legisla/tokenizer` | baked in the image |
| `DECISION_THRESHOLD` | `0.5` | in `(0, 1)` |
| `MAX_BATCH` | `16` | per-request item limit |
| `MAX_CHARS` | `500000` | per-text limit |
| `TORCH_NUM_THREADS` | `0` | `0` keeps the PyTorch default |
| `PORT` | `8000` | container only |

## 8. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `/readyz` stays `503` | Checkpoint missing, wrong target, or `state_dict` mismatch | Read the container log line `Model failed to load: ...` |
| `Could not read checkpoint ... Unsupported global` | Checkpoint not exported | Re-run step 2; the service only loads slim files with `weights_only=True` |
| `422 invalid_input` | Empty text, text over `MAX_CHARS`, wrong JSON types | Fix the request |
| `422 batch_too_large` | More than `MAX_BATCH` items | Split the batch |
