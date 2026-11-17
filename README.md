# LegislaML

Predictive quality modeling, semantic risk scoring, and containerized inference for Brazilian federal legislation (`LEI` and `LCP` Statutes enacted from 2020 to 2026).

LegislaML combines classical NLP baselines with a fine-tuned **BERTimbau** classifier (`LegislaBERT`) to predict two post-enactment risk targets extracted from Federal Senate `Vide` records:
- **Veto Risk (`teve_veto`)**: whether a Statute received a partial or total presidential veto message.
- **Normative Instability (`instavel`)**: whether a Statute required subsequent statutory amendments or revocations due to drafting ambiguity or conflict.

---

## Architecture Overview

```text
legisla-ML/
├── .github/workflows/ci.yml   # Ruff lint, CPU pytest (>=85% cov), Docker build & smoke test
├── docs/
│   ├── adr/                   # Architecture Decision Records (0001–0003)
│   ├── agents/                # SDD, ML engineering, domain, and Git/PR playbooks
│   └── specs/                 # SDD feature specs (001-bert-fine-tuning .. 003-model-serving)
├── scripts/
│   ├── fetch_tokenizer.py     # Bakes BERTimbau tokenizer + config for offline serving
│   ├── smoke_container.sh     # Verifies /healthz=200, /readyz=503 (no ckpt), non-root uid
│   ├── bench_serving.py       # Latency benchmark against a running inference service
│   └── verify_parity.py       # QG6 validation of Serving Parity on the validation split
├── src/
│   ├── scraper.py             # Federal Senate Open Data API + Sigen statute collector
│   ├── preprocessing.py       # Text cleaning and spaCy Portuguese normalization
│   ├── features.py            # Linguistic readability/complexity metrics & target extraction
│   ├── model.py               # Classical ML baselines (Random Forest, SVM, XGBoost)
│   ├── legislabert.py         # Shared pure head_tail_encode, Dataset, and LegislaBERTClassifier
│   ├── train_legislabl_bert.py# BERTimbau fine-tuning loop (fp16, grad accum, MLflow)
│   └── serving/               # FastAPI offline CPU microservice + slim checkpoint export CLI
├── tests/                     # Network-free, GPU-free unit test suite (70 tests, 92.15% cov)
├── AGENTS.md                  # Repository governance invariants
├── GLOSSARY.md                # Canonical domain terminology
├── Dockerfile                 # Multi-stage CPU-only runtime image (non-root, HF_HUB_OFFLINE=1)
└── pyproject.toml             # Canonical uv project & dependency groups (default, serving, dev)
```

---

## Benchmarks & Data Leakage Audit (`teve_veto`)

Auditing the Federal Senate API responses uncovered duplicate Statute `id` entries across `LEI` and `LCP` queries (360 raw rows representing **180 unique Statutes**: 140 negative, 40 positive). Deduplicating by `id` before `train_test_split` (`144` train / `36` stratified validation, `seed=42`) eliminated train-test leakage and established the clean benchmark below:

| Model / Scenario | Dataset Integrity | Accuracy | Macro-F1 | Minority Recall (`teve_veto=1`) |
| :--- | :--- | :---: | :---: | :---: |
| Random Forest (`v1`) | Leaked (360 rows / duplicate IDs) | `0.9444` | `0.9113` | `0.7500` |
| **Random Forest (`v1`)** | **Deduplicated (180 unique Statutes)** | `0.7778` | `0.6000` | `0.2500` |
| **LegislaBERT (`BERTimbau`, `v2`, epoch 4)** | **Deduplicated (180 unique Statutes)** | **`0.8333`** | **`0.7340` (+13.4 pp)** | **`0.5000` (2x)** |

### Key Modeling & Serving Decisions
- **Zero Training-Serving Skew (`src/legislabert.py`)**: Both training and FastAPI serving share `head_tail_encode` ($512$ tokens: `[CLS] + 128 head + 382 tail + [SEP]`), preserving the *Ementa* and closing sanction/enforcement clauses.
- **Imbalance Mitigation & `score_positive` (`ADR-0002`)**: Training minimizes inverse-frequency weighted `CrossEntropyLoss` ($w_c = \frac{N}{C \cdot N_c}$). Accordingly, the API exposes `score_positive` as a balanced-prior ranking score (`τ = 0.5`), never as a calibrated real-world probability.
- **Mounted Slim Checkpoint (`ADR-0001`)**: `src.serving.export` strips `optimizer_state_dict`, sanitizes NumPy scalars, hashes the source checkpoint, and writes a ~416 MiB artifact loaded offline with `weights_only=True` and `strict=True`.

---

## Quickstart

### 1. Environment Setup (`uv`)

For CPU development, testing, and serving verification:

```bash
uv venv .venv
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r pyproject.toml --group dev --group serving
```

For GPU training and data collection, install the default dependency set with `uv pip install --python .venv/bin/python -r pyproject.toml`.

### 2. Quality Gates (Lint & Unit Tests)

```bash
.venv/bin/ruff check --config pyproject.toml src/ tests/ scripts/
PYTHONPATH=. .venv/bin/pytest tests/unit --cov=src --cov-fail-under=85
```

- **Current baseline**: `70 passed`, **`92.15%` coverage** (`100%` on `app.py`, `inference.py`, `schemas.py`, `settings.py`).

### 3. Data Pipeline & Fine-Tuning (`FEAT-001`)

```bash
python src/scraper.py
python src/preprocessing.py
python src/features.py
python -m src.train_legislabl_bert --target teve_veto --epochs 5 --batch-size 4 --grad-accum 4
```

Training logs epoch metrics (`train_loss`, `val_loss`, `val_f1_macro`, `val_balanced_acc`, `val_precision_veto`, `val_recall_veto`) and checkpoints (`models/best_legislabl_bert_<target>.pt`) to MLflow experiment `legislabl-bert-v2`.

### 4. Export Slim Checkpoint & Serve (`FEAT-003`)

Export the trained checkpoint and mount it read-only into the CPU container (`models/` and `*.pt` are git-ignored):

```bash
PYTHONPATH=. .venv/bin/python -m src.serving.export \
  --src models/best_legislabl_bert_teve_veto.pt \
  --dst models/slim_teve_veto.pt \
  --target teve_veto

docker build -t legislaml-serving:local .

docker run --rm -p 8000:8000 \
  -v "$(pwd)/models/slim_teve_veto.pt:/models/model.pt:ro" \
  -e LEGISLA_TARGET=teve_veto \
  legislaml-serving:local
```

**Endpoints:**
- `GET /healthz` — liveness (`200` while process is alive)
- `GET /readyz` — readiness (`200` after checkpoint load and warm-up; `503` otherwise)
- `GET /v1/model` — active target, SHA-256, validation metrics, and score semantics
- `POST /v1/predict` (`{"text": "..."}`) and `POST /v1/predict/batch` (`{"texts": [...]}`)

### 5. Verify Serving Parity (`QG6`, `ADR-0003`)

```bash
.venv/bin/python scripts/fetch_tokenizer.py --out .local/tokenizer
.venv/bin/python scripts/verify_parity.py \
  --checkpoint models/slim_teve_veto.pt \
  --tokenizer-dir .local/tokenizer \
  --data-path data/processed/leis_clean.csv \
  --features-path data/processed/leis_features.csv \
  --target teve_veto \
  --threads 4
```

Pass `--url http://127.0.0.1:8000` to score against a running container instead of the in-process `TestClient`.

---

## Serving & CI Evidence

| Gate / Metric | Target | Verified Measurement |
| :--- | :---: | :--- |
| Unit tests & coverage (`CI`) | `≥ 85%`, `< 90 s` | `70 passed`, **`92.15%`** (`45 s` in CI) |
| Single-request CPU latency (512 tokens, 4 threads) | `p95 ≤ 3.0 s` | `p50 = 0.588 s` / **`p95 = 0.705 s`** / `p99 = 0.796 s` |
| Resident memory after warm-up (CPU) | `≤ 1.5 GB` | **`~1.0 GB`** (`1,017,480 KB`) |
| Uncompressed Docker image size (`NFR04`) | `≤ 2.0 GB` | **`1.29 GB`** disk usage (`263 MB` content size) |
| Trained checkpoint parity (`QG6`, `N=36`) | `\|Δ\| ≤ 0.002` | **Macro-F1 `0.733990`** (`Δ = 0.0` vs. epoch 4 checkpoint) |

---

## Documentation & Governance

- [AGENTS.md](AGENTS.md) — Repository invariants and SDD workflow pointers
- [GLOSSARY.md](GLOSSARY.md) — Canonical legislative and ML serving terminology
- [docs/adr/](docs/adr/) — Architecture Decision Records (`0001`–`0003`)
- [docs/specs/](docs/specs/) — Full SDD specs for `FEAT-001`, `FEAT-002`, and `FEAT-003`
- [LICENSE](LICENSE) — MIT License
