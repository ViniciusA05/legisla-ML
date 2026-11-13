"""Seam: Predictor inference path (FEAT-003 FR06, I2, I3, I4, I6)."""

import math

import pytest
import torch
import torch.nn as nn

from src.legislabert import LegislaBERTClassifier, head_tail_encode
from src.serving.inference import Predictor
from tests.fixtures.synthetic_statutes import MockBertBackbone


def _model() -> LegislaBERTClassifier:
    torch.manual_seed(0)
    return LegislaBERTClassifier(backbone=MockBertBackbone(hidden_size=768)).eval()


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


class _ConstantLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float)

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        return self.logits.expand(batch, -1).clone(), torch.zeros(batch, 768)


def test_parity_with_training_forward(mock_tokenizer):
    """I2: serving logits-derived scores equal the training-time forward pass."""
    model = _model()
    text = _words(700)
    ids, mask, _, _ = head_tail_encode(mock_tokenizer, text)
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]), torch.tensor([mask]))
    expected = torch.softmax(logits, dim=-1)[0]

    pred = Predictor(model, mock_tokenizer).predict_one(text)

    assert pred.scores[0] == pytest.approx(expected[0].item(), abs=1e-5)
    assert pred.scores[1] == pytest.approx(expected[1].item(), abs=1e-5)


def test_scores_are_normalized_and_finite(mock_tokenizer):
    """I3."""
    preds = Predictor(_model(), mock_tokenizer).predict_batch([_words(5), _words(900), "x"])
    for pred in preds:
        assert math.isfinite(pred.score_positive)
        assert all(0.0 <= s <= 1.0 for s in pred.scores)
        assert sum(pred.scores) == pytest.approx(1.0, abs=1e-6)
        assert pred.score_positive == pred.scores[1]


def test_batch_matches_single(mock_tokenizer):
    """I4."""
    predictor = Predictor(_model(), mock_tokenizer)
    texts = [_words(3), _words(300), _words(800)]
    batch = predictor.predict_batch(texts)
    singles = [predictor.predict_one(t) for t in texts]
    for b, s in zip(batch, singles, strict=True):
        assert b.label == s.label
        assert b.score_positive == pytest.approx(s.score_positive, abs=1e-5)


def test_batch_preserves_order_and_chunks_by_max_batch(mock_tokenizer):
    predictor = Predictor(_model(), mock_tokenizer, max_batch=2)
    texts = [_words(n) for n in (1, 600, 2, 700, 3)]
    preds = predictor.predict_batch(texts)
    assert [p.n_tokens for p in preds] == [1, 600, 2, 700, 3]
    assert [p.truncated for p in preds] == [False, True, False, True, False]


def test_deterministic_across_calls_and_eval_mode(mock_tokenizer):
    """I6."""
    model = _model()
    predictor = Predictor(model, mock_tokenizer)
    first = predictor.predict_one(_words(50))
    second = predictor.predict_one(_words(50))
    assert model.training is False
    assert first == second


def test_predictor_forces_eval_mode(mock_tokenizer):
    model = _model().train()
    Predictor(model, mock_tokenizer)
    assert model.training is False


def test_threshold_boundary_is_inclusive(mock_tokenizer):
    pred = Predictor(_ConstantLogits([0.0, 0.0]), mock_tokenizer, threshold=0.5).predict_one("a")
    assert pred.score_positive == pytest.approx(0.5)
    assert pred.label == 1


def test_custom_threshold_changes_label(mock_tokenizer):
    model = _ConstantLogits([0.0, 1.0])  # p1 ~ 0.731
    assert Predictor(model, mock_tokenizer, threshold=0.5).predict_one("a").label == 1
    assert Predictor(model, mock_tokenizer, threshold=0.9).predict_one("a").label == 0


def test_empty_batch_returns_empty_list(mock_tokenizer):
    assert Predictor(_model(), mock_tokenizer).predict_batch([]) == []


def test_truncation_flags_and_token_counts(mock_tokenizer):
    predictor = Predictor(_model(), mock_tokenizer)
    short = predictor.predict_one(_words(510))
    long = predictor.predict_one(_words(511))
    assert (short.n_tokens, short.truncated) == (510, False)
    assert (long.n_tokens, long.truncated) == (511, True)


def test_warmup_runs_a_forward_pass(mock_tokenizer):
    Predictor(_model(), mock_tokenizer).warmup()  # must not raise
