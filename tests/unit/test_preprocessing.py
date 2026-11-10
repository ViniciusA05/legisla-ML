"""Unit tests for Seam 5: Statutory text cleaning, feature extraction, and token counters."""

import json

import pandas as pd

import src.preprocessing as prep_mod
from src.features import LegislacaoFeatureExtractor
from src.preprocessing import LegislacaoPreprocessor


class _FakeToken:
    def __init__(self, text: str):
        self.text = text
        self.lemma_ = text.lower().strip(".,;:")
        self.is_punct = not bool(self.lemma_)
        self.is_space = not bool(text.strip())


def _fake_nlp(sentence: str):
    return [_FakeToken(w) for w in sentence.split()]


def test_clean_text_and_process_document(monkeypatch, tmp_path, sample_statutes):
    """Preprocessor strips official disclaimers, normalizes whitespace, and extracts tokens."""
    monkeypatch.setattr(prep_mod.spacy, "load", lambda name: _fake_nlp)
    monkeypatch.setattr(
        prep_mod.nltk,
        "sent_tokenize",
        lambda text, language="portuguese": [s.strip() for s in text.split(".") if s.strip()],
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "lei_1.txt").write_text(sample_statutes["noisy"], encoding="utf-8")
    pd.DataFrame(
        [
            {
                "id": "1",
                "tipo": "LEI",
                "numero": "100",
                "ano": 2024,
                "data": "2024-01-01",
                "ementa": "Ementa",
                "meta_file": "meta_1.json",
                "text_file": "lei_1.txt",
            }
        ]
    ).to_csv(raw_dir / "index.csv", index=False)

    prep = LegislacaoPreprocessor(base_path=str(tmp_path))
    cleaned = prep.clean_text(sample_statutes["noisy"])
    assert "[Detalhes da Norma]" not in cleaned
    assert "Diário Oficial" not in cleaned
    assert cleaned.startswith("Art. 1º")

    count = prep.run_pipeline()
    assert count == 1
    df_out = pd.read_csv(tmp_path / "processed" / "leis_clean.csv")
    assert len(df_out) == 1


def test_feature_extractor_metrics_and_pipeline(tmp_path):
    """FeatureExtractor computes Flesch index, syllable counts, and vides metadata."""
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    meta_payload = {
        "DetalheDocumento": {
            "documentos": {
                "documento": [
                    {
                        "vides": {
                            "vide": [
                                {"nomeNormaPosterior": "Veto parcial", "comentario": "Alteração e revogação parcial"},
                            ]
                        }
                    }
                ]
            }
        }
    }
    (raw_dir / "meta_1.json").write_text(json.dumps(meta_payload), encoding="utf-8")

    texto = "Art. 1º O parágrafo único define a competência e jurisdição do tribunal conforme a Lei nº 9.999."
    pd.DataFrame(
        [
            {
                "id": "1",
                "tipo": "LEI",
                "numero": "100",
                "ano": 2024,
                "data": "2024-01-01",
                "ementa": "Define competência",
                "texto_original": texto,
                "sentencas": json.dumps([texto]),
                "tokens": json.dumps(["art", "parágrafo", "competência", "jurisdição", "tribunal"]),
                "meta_file": "meta_1.json",
            }
        ]
    ).to_csv(proc_dir / "leis_clean.csv", index=False)

    extractor = LegislacaoFeatureExtractor(base_path=str(tmp_path))
    assert extractor.calculate_flesch([], 0) == 0.0
    assert 0.0 <= extractor.calculate_flesch(["lei", "dados", "abertos"], 1) <= 100.0

    count = extractor.run_pipeline()
    assert count == 1
    df_feat = pd.read_csv(proc_dir / "leis_features.csv")
    assert len(df_feat) == 1
    assert df_feat.iloc[0]["teve_veto"] == 1
    assert df_feat.iloc[0]["instavel"] == 1
