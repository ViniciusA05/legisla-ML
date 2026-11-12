"""FastAPI application factory for the LegislaBERT inference service (FEAT-003)."""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.legislabert import HEAD_TOKENS, MAX_LENGTH, TAIL_TOKENS, LegislaBERTClassifier
from src.serving import SERVICE_VERSION
from src.serving.checkpoint import load_slim_checkpoint
from src.serving.inference import Predictor
from src.serving.schemas import (
    BatchPredictionOut,
    BatchPredictRequest,
    ModelInfo,
    PredictionOut,
    PredictRequest,
)
from src.serving.settings import Settings

logger = logging.getLogger("legislaml.serving")


@dataclass(frozen=True)
class ModelBundle:
    predictor: Predictor
    info: ModelInfo


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def build_default_bundle(settings: Settings) -> ModelBundle:
    """Load the tokenizer and slim checkpoint from local paths only (offline)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(settings.tokenizer_dir), local_files_only=True)
    loaded = load_slim_checkpoint(
        settings.checkpoint_path,
        model_factory=lambda: LegislaBERTClassifier.from_config(str(settings.tokenizer_dir)),
        expected_target=settings.target,
    )
    predictor = Predictor(
        loaded.model,
        tokenizer,
        threshold=settings.decision_threshold,
        max_batch=settings.max_batch,
    )
    info = ModelInfo(
        service_version=SERVICE_VERSION,
        target=loaded.target,
        model_name=loaded.model_name,
        max_length=MAX_LENGTH,
        head_tokens=HEAD_TOKENS,
        tail_tokens=TAIL_TOKENS,
        checkpoint_sha256=loaded.sha256,
        val_metrics=loaded.metrics,
        device="cpu",
    )
    return ModelBundle(predictor=predictor, info=info)


def create_app(
    bundle_factory: Callable[[Settings], ModelBundle] = build_default_bundle,
    settings: Settings | None = None,
) -> FastAPI:
    """
    Build the application. The model loads in the lifespan; a load failure is logged
    and leaves the service alive but never ready (`/readyz` stays 503).
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = None
        try:
            if settings.torch_num_threads > 0:
                torch.set_num_threads(settings.torch_num_threads)
            bundle = bundle_factory(settings)
            bundle.predictor.warmup()
            app.state.bundle = bundle
            logger.info("Model ready (target=%s)", bundle.info.target)
        except Exception as exc:  # noqa: BLE001 - keep process alive, stay not-ready
            logger.error("Model failed to load: %s", exc)
        yield

    app = FastAPI(title="LegislaML Inference", version=SERVICE_VERSION, lifespan=lifespan)

    def get_bundle(request: Request) -> ModelBundle:
        bundle = getattr(request.app.state, "bundle", None)
        if bundle is None:
            raise ApiError(503, "model_not_ready", "Model is not loaded")
        return bundle

    def check_text(text: str) -> None:
        if not text.strip():
            raise ApiError(422, "invalid_input", "text must not be empty")
        if len(text) > settings.max_chars:
            raise ApiError(
                422, "invalid_input", f"text exceeds the {settings.max_chars} character limit"
            )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors())
        return _error_response(422, "invalid_input", f"Invalid request fields: {fields}")

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        logger.error("Unhandled error: %s", type(exc).__name__)  # never log request content
        return _error_response(500, "internal", "Internal server error")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request):
        get_bundle(request)
        return {"status": "ready"}

    @app.get("/v1/model", response_model=ModelInfo)
    def model_info(request: Request):
        return get_bundle(request).info

    @app.post("/v1/predict", response_model=PredictionOut)
    def predict(body: PredictRequest, request: Request):
        bundle = get_bundle(request)
        check_text(body.text)
        return PredictionOut.from_prediction(bundle.predictor.predict_one(body.text))

    @app.post("/v1/predict/batch", response_model=BatchPredictionOut)
    def predict_batch(body: BatchPredictRequest, request: Request):
        bundle = get_bundle(request)
        if not body.texts:
            raise ApiError(422, "invalid_input", "texts must contain at least one item")
        if len(body.texts) > settings.max_batch:
            raise ApiError(
                422, "batch_too_large", f"batch exceeds the {settings.max_batch} item limit"
            )
        for text in body.texts:
            check_text(text)
        preds = bundle.predictor.predict_batch(body.texts)
        return BatchPredictionOut(predictions=[PredictionOut.from_prediction(p) for p in preds])

    return app


app = create_app()
