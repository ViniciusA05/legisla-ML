"""Unit tests for Seam 1: LegislaBERTDataset Head-Tail tokenization contract."""

import torch

from src.train_legislabl_bert import LegislaBERTDataset


def test_short_sequence_padding_and_special_tokens(mock_tokenizer, sample_statutes):
    """Short sequences (<510 tokens) preserve all tokens and zero-pad up to 512."""
    text = sample_statutes["short"]
    dataset = LegislaBERTDataset([text], [1], mock_tokenizer, max_length=512)

    assert len(dataset) == 1
    item = dataset[0]
    raw_tokens = mock_tokenizer.encode(text, add_special_tokens=False)
    n_tok = len(raw_tokens)

    assert item["input_ids"].shape == (512,)
    assert item["attention_mask"].shape == (512,)
    assert item["label"].item() == 1
    assert item["input_ids"][0].item() == mock_tokenizer.cls_token_id
    assert item["input_ids"][1 : n_tok + 1].tolist() == raw_tokens
    assert item["input_ids"][n_tok + 1].item() == mock_tokenizer.sep_token_id
    assert (item["input_ids"][n_tok + 2 :] == mock_tokenizer.pad_token_id).all()
    assert (item["attention_mask"][: n_tok + 2] == 1).all()
    assert (item["attention_mask"][n_tok + 2 :] == 0).all()


def test_long_sequence_head_tail_window_boundary(mock_tokenizer, sample_statutes):
    """Long sequences (>510 tokens) keep exact 128 head + 382 tail tokens."""
    text = sample_statutes["long"]
    dataset = LegislaBERTDataset(
        [text], [0], mock_tokenizer, max_length=512, head_tokens=128, tail_tokens=382
    )

    item = dataset[0]
    raw_tokens = mock_tokenizer.encode(text, add_special_tokens=False)
    assert len(raw_tokens) == 600

    assert item["input_ids"].shape == (512,)
    assert (item["attention_mask"] == 1).all()
    assert item["input_ids"][0].item() == mock_tokenizer.cls_token_id
    assert item["input_ids"][511].item() == mock_tokenizer.sep_token_id
    assert item["input_ids"][1:129].tolist() == raw_tokens[:128]
    assert item["input_ids"][129:511].tolist() == raw_tokens[-382:]
    assert item["label"].dtype == torch.long
