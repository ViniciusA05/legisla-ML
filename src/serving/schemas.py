"""Pydantic request/response contracts for the inference API (FEAT-003 §3.3)."""

from pydantic import BaseModel

from src.serving.inference import Prediction

SCORE_SEMANTICS = "balanced-prior ranking score; not a calibrated probability"


class PredictRequest(BaseModel):
    text: str


class BatchPredictRequest(BaseModel):
    texts: list[str]


class PredictionOut(BaseModel):
    label: int
    score_positive: float
    scores: tuple[float, float]
    threshold: float
    truncated: bool
    n_tokens: int

    @classmethod
    def from_prediction(cls, pred: Prediction) -> "PredictionOut":
        return cls(
            label=pred.label,
            score_positive=pred.score_positive,
            scores=pred.scores,
            threshold=pred.threshold,
            truncated=pred.truncated,
            n_tokens=pred.n_tokens,
        )


class BatchPredictionOut(BaseModel):
    predictions: list[PredictionOut]


class ModelInfo(BaseModel):
    service_version: str
    target: str
    model_name: str
    max_length: int
    head_tokens: int
    tail_tokens: int
    checkpoint_sha256: str
    val_metrics: dict
    device: str
    score_semantics: str = SCORE_SEMANTICS


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody
