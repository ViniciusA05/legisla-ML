"""Deterministic synthetic fixtures for zero-GPU offline unit testing."""

import pandas as pd
import torch
import torch.nn as nn


class DeterministicMockTokenizer:
    """Deterministic tokenizer mapping each word to a fixed sequential integer ID."""

    def __init__(self):
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = str(text).split()
        tokens: list[int] = []
        for idx, word in enumerate(words):
            # Deterministic token ID encoding both position and word length
            token_id = 1000 + idx
            tokens.append(token_id)
        if add_special_tokens:
            return [self.cls_token_id] + tokens + [self.sep_token_id]
        return tokens


class MockBertConfig:
    """Lightweight configuration matching BERTimbau hidden dimension (768)."""

    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size


class MockBertOutput:
    """Output container exposing last_hidden_state tensor."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class MockBertBackbone(nn.Module):
    """CPU-only mock backbone producing deterministic (B, L, 768) hidden states."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.config = MockBertConfig(hidden_size=hidden_size)
        self.proj = nn.Linear(1, hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> MockBertOutput:
        batch_size, seq_len = input_ids.shape
        x = (input_ids.float().unsqueeze(-1) % 10.0) / 10.0
        if attention_mask is not None:
            x = x * attention_mask.float().unsqueeze(-1)
        hidden = self.proj(x)
        return MockBertOutput(last_hidden_state=hidden)


SAMPLE_SHORT_STATUTE = (
    "Art. 1o Esta lei estabelece diretrizes para a politica nacional de dados abertos."
)

SAMPLE_LONG_STATUTE = " ".join([f"token_{i}" for i in range(600)])

SAMPLE_NOISY_STATUTE = (
    " [Detalhes da Norma]   \n\n\t "
    "Este texto não substitui o original publicado no Diário Oficial.\n\n"
    "Art. 1º Fica instituído o regime de transparência legislativa.   "
)


def create_synthetic_statutes_df(n_rows: int = 12, n_unique: int = 8) -> pd.DataFrame:
    """Create a DataFrame with deliberate duplicate statute IDs."""
    ids = [f"LEI_{100 + (i % n_unique)}" for i in range(n_rows)]
    return pd.DataFrame(
        {
            "id": ids,
            "tipo": ["LEI"] * n_rows,
            "numero": [14000 + i for i in range(n_rows)],
            "ano": [2024] * n_rows,
            "data": ["2024-06-01"] * n_rows,
            "ementa": [f"Dispõe sobre norma {i}" for i in range(n_rows)],
            "texto_original": [f"Art. 1º Texto da norma {i}." for i in range(n_rows)],
            "teve_veto": [i % 2 for i in range(n_rows)],
            "instavel": [(i + 1) % 2 for i in range(n_rows)],
        }
    )
