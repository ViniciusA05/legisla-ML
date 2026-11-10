"""Shared pytest fixtures for LegislaML unit tests."""

import pytest

from tests.fixtures.synthetic_statutes import (
    SAMPLE_LONG_STATUTE,
    SAMPLE_NOISY_STATUTE,
    SAMPLE_SHORT_STATUTE,
    DeterministicMockTokenizer,
    MockBertBackbone,
    create_synthetic_statutes_df,
)


@pytest.fixture
def mock_tokenizer() -> DeterministicMockTokenizer:
    return DeterministicMockTokenizer()


@pytest.fixture
def mock_backbone() -> MockBertBackbone:
    return MockBertBackbone(hidden_size=768)


@pytest.fixture
def sample_statutes() -> dict[str, str]:
    return {
        "short": SAMPLE_SHORT_STATUTE,
        "long": SAMPLE_LONG_STATUTE,
        "noisy": SAMPLE_NOISY_STATUTE,
    }


@pytest.fixture
def synthetic_df():
    return create_synthetic_statutes_df(n_rows=12, n_unique=8)
