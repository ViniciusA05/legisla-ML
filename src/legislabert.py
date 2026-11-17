"""
LegislaBERT core modules shared by training (FEAT-001) and serving (FEAT-003).

This module depends only on torch and transformers so that an inference
process never imports mlflow, pandas, or scikit-learn.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel

DEFAULT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 512
HEAD_TOKENS = 128
TAIL_TOKENS = 382


def head_tail_encode(
    tokenizer,
    text: str,
    max_length: int = MAX_LENGTH,
    head_tokens: int = HEAD_TOKENS,
    tail_tokens: int = TAIL_TOKENS,
) -> tuple[list[int], list[int], int, bool]:
    """
    Encode text with the Head-Tail Window.

    Keeps the first head_tokens and the last max_length - 2 - head_tokens
    tokens when the text exceeds the usable budget (max_length - 2).

    Returns:
        (input_ids, attention_mask, n_tokens, truncated) where both lists have
        length max_length, n_tokens is the tokenizer length before windowing,
        and truncated tells whether the window dropped tokens.
    """
    tokens = tokenizer.encode(str(text), add_special_tokens=False)
    usable_budget = max_length - 2  # Reserve slots for [CLS] and [SEP]
    truncated = len(tokens) > usable_budget

    if truncated:
        head = tokens[:head_tokens]
        tail = tokens[-(usable_budget - head_tokens) :]
        selected_tokens = head + tail
    else:
        selected_tokens = tokens

    input_ids = [tokenizer.cls_token_id] + selected_tokens + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)

    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length

    return input_ids, attention_mask, len(tokens), truncated


class LegislaBERTDataset(Dataset):
    """
    Statute dataset applying Head-Tail Window sampling.

    Preserves the first `head_tokens` (Ementa and preliminary scope) and the
    final `tail_tokens` (penalties, revocation, and enforcement clauses) within
    the 512-token Transformer self-attention ceiling.
    """

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length: int = MAX_LENGTH,
        head_tokens: int = HEAD_TOKENS,
        tail_tokens: int = TAIL_TOKENS,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        label = int(self.labels[idx])
        input_ids, attention_mask, _, _ = head_tail_encode(
            self.tokenizer,
            self.texts[idx],
            max_length=self.max_length,
            head_tokens=self.head_tokens,
            tail_tokens=self.tail_tokens,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


class LegislaBERTClassifier(nn.Module):
    """
    Contextual classification and latent embedding extraction module.

    Extracts the `[CLS]` vector (h in R^768) from the final hidden layer of
    BERTimbau and projects it through Dropout(0.3) to class logits.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_classes: int = 2,
        dropout_prob: float = 0.3,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.bert = backbone if backbone is not None else AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    @classmethod
    def from_config(
        cls,
        config_source,
        num_classes: int = 2,
        dropout_prob: float = 0.3,
    ) -> "LegislaBERTClassifier":
        """
        Build the classifier with a randomly initialised backbone (no weight download).

        config_source is a model name, a local directory containing
        config.json, or a PretrainedConfig instance. The fine-tuned
        state_dict is expected to overwrite every parameter afterwards.
        """
        config = (
            AutoConfig.from_pretrained(config_source)
            if isinstance(config_source, str)
            else config_source
        )
        return cls(
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            backbone=AutoModel.from_config(config),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls_embedding)
        logits = self.classifier(pooled)
        return logits, cls_embedding
