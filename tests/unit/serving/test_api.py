"""Seam: FastAPI routes, validation, readiness, error envelope, settings (FEAT-003 FR06-FR09)."""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.legislabert import LegislaBERTClassifier
from src.serving import app as app_module
from src.serving.app import ModelBundle, build_default_bundle, create_app
from src.serving.export import export_slim_checkpoint
from src.serving.inference import Predictor
from src.serving.schemas import ModelInfo
from src.serving.settings import Settings, SettingsError
from tests.fixtures.synthetic_statutes import MockBertBackbone


def _info(target: str = "teve_veto") -> ModelInfo:
    return ModelInfo(
        service_version="test",
        target=target,
        model_name="mock",
        max_length=512,
        head_tokens=128,
        tail_tokens=382,
        checkpoint_sha256="abc",
        val_metrics={"f1_macro": 0.7},
        device="cpu",
    )


def _bundle_factory(mock_tokenizer):
    def factory(settings: Settings) -> ModelBundle:
        torch.manual_seed(0)
        model = LegislaBERTClassifier(backbone=MockBertBackbone(hidden_size=768))
        predictor = Predictor(
            model, mock_tokenizer, settings.decision_threshold, settings.max_batch
        )
        return ModelBundle(predictor=predictor, info=_info())

    return factory


@pytest.fixture
def settings():
    return Settings(max_batch=3, max_chars=5000)


@pytest.fixture
def client(mock_tokenizer, settings):
    app = create_app(_bundle_factory(mock_tokenizer), settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _failing_factory(_settings):
    raise RuntimeError("boom")


def test_predict_returns_contract_shape(client):
    response = client.post("/v1/predict", json={"text": "Art. 1 Esta lei dispoe sobre dados."})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "score_positive", "scores", "threshold", "truncated", "n_tokens"}
    assert body["label"] in (0, 1)
    assert sum(body["scores"]) == pytest.approx(1.0, abs=1e-6)
    assert body["score_positive"] == pytest.approx(body["scores"][1])
    assert body["truncated"] is False


def test_batch_preserves_order(client):
    texts = ["a", " ".join(["w"] * 700), "b c"]
    response = client.post("/v1/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert [p["n_tokens"] for p in preds] == [1, 700, 2]
    assert [p["truncated"] for p in preds] == [False, True, False]


def test_model_endpoint_exposes_metadata_and_score_semantics(client):
    body = client.get("/v1/model").json()
    assert body["target"] == "teve_veto"
    assert body["head_tokens"] == 128 and body["tail_tokens"] == 382
    assert "not a calibrated probability" in body["score_semantics"]


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   \n\t"}, {"text": "x" * 5001}, {"text": 123}, {}, {"nope": "x"}],
)
def test_predict_rejects_invalid_input(client, payload):
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"texts": []}, "invalid_input"),
        ({"texts": ["a", "b", "c", "d"]}, "batch_too_large"),
        ({"texts": ["a", ""]}, "invalid_input"),
        ({"texts": "not-a-list"}, "invalid_input"),
    ],
)
def test_batch_rejects_invalid_input(client, payload, code):
    response = client.post("/v1/predict/batch", json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": code, "message": response.json()["error"]["message"]}}


def test_health_ok_and_ready_after_warmup(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200


def test_failed_model_load_keeps_liveness_but_never_ready(settings):
    with TestClient(create_app(_failing_factory, settings), raise_server_exceptions=False) as c:
        assert c.get("/healthz").status_code == 200
        for method, path, kwargs in [
            ("get", "/readyz", {}),
            ("get", "/v1/model", {}),
            ("post", "/v1/predict", {"json": {"text": "a"}}),
            ("post", "/v1/predict/batch", {"json": {"texts": ["a"]}}),
        ]:
            response = getattr(c, method)(path, **kwargs)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "model_not_ready"


def test_unexpected_exception_returns_internal_without_echoing_input(mock_tokenizer, settings):
    secret = "CONFIDENTIAL-STATUTE-TEXT"

    def factory(s):
        bundle = _bundle_factory(mock_tokenizer)(s)

        original = bundle.predictor.predict_one

        def explode(text):
            if text == "warmup":  # let the lifespan warm-up succeed
                return original(text)
            raise ValueError(f"leak {secret}")

        bundle.predictor.predict_one = explode
        return bundle

    with TestClient(create_app(factory, settings), raise_server_exceptions=False) as c:
        response = c.post("/v1/predict", json={"text": secret})
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal", "message": "Internal server error"}}
    assert secret not in response.text


def test_torch_threads_setting_is_applied(mock_tokenizer, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module.torch, "set_num_threads", calls.append)
    app = create_app(_bundle_factory(mock_tokenizer), Settings(torch_num_threads=2))
    with TestClient(app):
        pass
    assert calls == [2]


def test_build_default_bundle_loads_slim_checkpoint_offline(monkeypatch, tmp_path, mock_tokenizer):
    torch.manual_seed(0)
    model = LegislaBERTClassifier(backbone=MockBertBackbone(hidden_size=768))
    full = tmp_path / "full.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "metrics": {"f1_macro": np.float64(0.73)},
            "config": {"model_name": "neuralmind/bert-base-portuguese-cased", "max_length": 512},
        },
        full,
    )
    export_slim_checkpoint(full, tmp_path / "slim.pt", target="instavel")

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: mock_tokenizer
    )
    monkeypatch.setattr(
        LegislaBERTClassifier,
        "from_config",
        classmethod(
            lambda cls, *a, **kw: cls(backbone=MockBertBackbone(hidden_size=768))
        ),
    )
    settings = Settings(
        checkpoint_path=tmp_path / "slim.pt", tokenizer_dir=tmp_path, target="instavel"
    )

    bundle = build_default_bundle(settings)

    assert bundle.info.target == "instavel"
    assert bundle.info.val_metrics["f1_macro"] == pytest.approx(0.73)
    assert bundle.predictor.predict_one("Art. 1").label in (0, 1)


def test_settings_defaults_and_overrides():
    assert Settings.from_env({}) == Settings()
    env = {
        "LEGISLA_CHECKPOINT_PATH": "/data/m.pt",
        "LEGISLA_TARGET": "teve_veto",
        "DECISION_THRESHOLD": "0.4",
        "MAX_BATCH": "4",
        "MAX_CHARS": "99",
        "TORCH_NUM_THREADS": "2",
    }
    s = Settings.from_env(env)
    assert str(s.checkpoint_path) == "/data/m.pt" and s.target == "teve_veto"
    assert (s.decision_threshold, s.max_batch, s.max_chars, s.torch_num_threads) == (0.4, 4, 99, 2)


@pytest.mark.parametrize(
    "env",
    [
        {"LEGISLA_TARGET": "other"},
        {"DECISION_THRESHOLD": "1.5"},
        {"DECISION_THRESHOLD": "abc"},
        {"MAX_BATCH": "0"},
        {"MAX_CHARS": "0"},
        {"TORCH_NUM_THREADS": "-1"},
    ],
)
def test_settings_reject_invalid_values(env):
    with pytest.raises(SettingsError):
        Settings.from_env(env)
