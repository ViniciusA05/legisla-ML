"""Seam: shared Head-Tail encoding (FEAT-003 I1, FR01)."""

import pytest
import torch

from src.legislabert import LegislaBERTDataset, head_tail_encode


def _text(n_words: int) -> str:
    return " ".join(f"w{i}" for i in range(n_words))


@pytest.mark.parametrize("n_words", [0, 1, 509, 510, 511, 1000])
def test_head_tail_encode_shapes_and_special_tokens(mock_tokenizer, n_words):
    ids, mask, n_tokens, truncated = head_tail_encode(mock_tokenizer, _text(n_words))

    assert len(ids) == len(mask) == 512
    assert n_tokens == n_words
    assert truncated is (n_words > 510)
    assert ids[0] == mock_tokenizer.cls_token_id
    real = sum(mask)
    assert ids[real - 1] == mock_tokenizer.sep_token_id
    assert all(i == mock_tokenizer.pad_token_id for i in ids[real:])
    assert mask == [1] * real + [0] * (512 - real)


def test_head_tail_encode_keeps_head_and_tail_when_truncating(mock_tokenizer):
    raw = mock_tokenizer.encode(_text(1000))
    ids, mask, _, truncated = head_tail_encode(mock_tokenizer, _text(1000))

    assert truncated is True
    assert sum(mask) == 512
    assert ids[1:129] == raw[:128]
    assert ids[129:511] == raw[-382:]


def test_head_tail_encode_custom_window(mock_tokenizer):
    ids, mask, _, truncated = head_tail_encode(
        mock_tokenizer, _text(50), max_length=16, head_tokens=4, tail_tokens=10
    )
    assert truncated is True
    assert len(ids) == 16 and sum(mask) == 16


def test_dataset_delegates_to_shared_function_without_skew(mock_tokenizer):
    """I1: dataset and serving path must produce identical tensors."""
    for n_words in (3, 510, 700):
        text = _text(n_words)
        item = LegislaBERTDataset([text], [1], mock_tokenizer)[0]
        ids, mask, _, _ = head_tail_encode(mock_tokenizer, text)
        assert torch.equal(item["input_ids"], torch.tensor(ids))
        assert torch.equal(item["attention_mask"], torch.tensor(mask))
