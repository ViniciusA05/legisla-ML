"""Unit tests for Seam 4: Dataset deduplication by ID and classic pipeline integrity."""

import json

import pandas as pd

import src.scraper as scraper_mod
from src.model import LegislacaoModelPipeline
from src.scraper import LegislacaoScraper


def test_dataframe_id_deduplication_contract(synthetic_df):
    """Deduplicating on id strips all redundant statute records."""
    assert len(synthetic_df) == 12
    deduped = synthetic_df.drop_duplicates(subset=["id"])
    assert len(deduped) == 8
    assert deduped["id"].is_unique


def test_model_pipeline_deduplicates_and_evaluates(tmp_path):
    """LegislacaoModelPipeline strips duplicate IDs and saves metrics JSON."""
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True)

    rows = []
    for i in range(25):
        law_id = f"ID_{i % 20}"
        rows.append(
            {
                "id": law_id,
                "flesch": float(40 + (i % 30)),
                "palavras_por_sentenca": float(15 + (i % 10)),
                "sentencas_por_artigo": float(2 + (i % 4)),
                "densidade_referencias": float(0.1 * (i % 5)),
                "ttr": float(0.4 + 0.02 * (i % 10)),
                "proporcao_termos_tecnicos": float(0.05 * (i % 4)),
                "teve_veto": int(i % 2),
                "instavel": int((i + 1) % 2),
            }
        )
    pd.DataFrame(rows).to_csv(proc_dir / "leis_features.csv", index=False)

    pipeline = LegislacaoModelPipeline(base_path=str(tmp_path))
    assert pipeline.run_pipeline() is True
    assert (proc_dir / "model_metrics.json").exists()
    metrics = json.loads((proc_dir / "model_metrics.json").read_text(encoding="utf-8"))
    assert "teve_veto" in metrics
    assert "instavel" in metrics


def test_scraper_methods_and_deduplication(monkeypatch, tmp_path):
    """LegislacaoScraper parses HTML/JSON responses and deduplicates index.csv."""
    scraper = LegislacaoScraper(base_path=str(tmp_path))

    class FakeResp:
        def __init__(self, json_data=None, text=""):
            self._json = json_data or {}
            self.text = text

        def raise_for_status(self):
            pass

        def json(self):
            return self._json

    def fake_get(url, params=None, timeout=15):
        if "lista.json" in url:
            return FakeResp(
                {
                    "ListaDocumento": {
                        "documentos": {
                            "documento": [
                                {
                                    "id": "10",
                                    "tipo": "LEI-n",
                                    "numero": "1",
                                    "dataassinatura": "2024-01-01",
                                    "ementa": "Lei teste",
                                },
                                {
                                    "id": "10",
                                    "tipo": "LEI-n",
                                    "numero": "1",
                                    "dataassinatura": "2024-01-01",
                                    "ementa": "Lei teste duplicada",
                                },
                            ]
                        }
                    }
                }
            )
        if url.endswith(".json"):
            return FakeResp({"DetalheDocumento": {"ok": True}})
        if "publicacao/" in url:
            return FakeResp(
                text='<html><div id="conteudoPrincipal">Texto integral da lei</div></html>'
            )
        return FakeResp(text='<html><a href="/publicacao/10">Texto Sigen</a></html>')

    monkeypatch.setattr(scraper_mod.requests, "get", fake_get)
    monkeypatch.setattr(scraper_mod.time, "sleep", lambda s: None)

    total = scraper.run_pipeline(start_year=2024, end_year=2024, limit_per_year=5)
    assert total == 1
    df_idx = pd.read_csv(tmp_path / "raw" / "index.csv")
    assert len(df_idx) == 1
    assert df_idx["id"].is_unique
