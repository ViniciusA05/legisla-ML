# syntax=docker/dockerfile:1
# LegislaBERT CPU inference image (FEAT-003). Weights are mounted at runtime, never baked in.

ARG PYTHON_VERSION=3.12

# ---------- builder: resolve CPU-only dependencies and bake tokenizer + config ----------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build
COPY pyproject.toml ./

RUN uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match \
      --group serving

ARG MODEL_NAME=neuralmind/bert-base-portuguese-cased
COPY scripts/fetch_tokenizer.py ./scripts/fetch_tokenizer.py
RUN /opt/venv/bin/python scripts/fetch_tokenizer.py \
      --model-name "${MODEL_NAME}" --out /opt/legisla/tokenizer \
 && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && find /opt/venv -type d \( -name "tests" -o -name "test" \) -prune -exec rm -rf {} +

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

RUN useradd --create-home --uid 10001 legisla

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/legisla/tokenizer /opt/legisla/tokenizer
WORKDIR /app
COPY src/ ./src/

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LEGISLA_CHECKPOINT_PATH=/models/model.pt \
    LEGISLA_TOKENIZER_DIR=/opt/legisla/tokenizer \
    PORT=8000

USER legisla
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'), timeout=3)"]

# Single worker: each extra worker would duplicate ~440 MB of weights. Scale with containers.
CMD ["sh", "-c", "exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}"]
