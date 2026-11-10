"""Unit tests for Seam 2: LegislaBERTClassifier forward/backward and training loop on CPU."""

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

import src.train_legislabl_bert as bert_mod
from src.train_legislabl_bert import (
    LegislaBERTClassifier,
    LegislaBERTDataset,
    evaluate,
    run_fine_tuning,
    train_epoch,
)


def test_classifier_forward_shapes_and_gradients(monkeypatch, mock_backbone):
    """Classifier outputs (B, 2) logits and (B, 768) CLS embeddings without NaNs."""
    monkeypatch.setattr(bert_mod.AutoModel, "from_pretrained", lambda *a, **kw: mock_backbone)

    model = LegislaBERTClassifier(model_name="mock-bertimbau", num_classes=2, dropout_prob=0.1)
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(1, 500, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    logits, cls_emb = model(input_ids, attention_mask)

    assert logits.shape == (batch_size, 2)
    assert cls_emb.shape == (batch_size, 768)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    loss = logits.sum()
    loss.backward()
    assert model.classifier.weight.grad is not None


def test_train_epoch_and_evaluate_on_cpu(monkeypatch, mock_backbone, mock_tokenizer):
    """Execute one CPU epoch and evaluation pass using synthetic dataset."""
    monkeypatch.setattr(bert_mod.AutoModel, "from_pretrained", lambda *a, **kw: mock_backbone)

    texts = ["Art 1 texto curto um", "Art 2 texto curto dois", "Art 3 texto tres", "Art 4 quatro"]
    labels = [0, 1, 0, 1]
    ds = LegislaBERTDataset(
        texts, labels, mock_tokenizer, max_length=16, head_tokens=4, tail_tokens=10
    )
    loader = DataLoader(ds, batch_size=2)

    device = torch.device("cpu")
    model = LegislaBERTClassifier(model_name="mock").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    scaler = GradScaler("cuda", enabled=False)

    train_loss = train_epoch(
        model, loader, criterion, optimizer, scheduler, scaler, device, grad_accum_steps=2
    )
    assert isinstance(train_loss, float)
    assert train_loss >= 0.0

    metrics, preds, gold = evaluate(model, loader, criterion, device)
    assert "f1_macro" in metrics
    assert "recall_veto" in metrics
    assert len(preds) == 4
    assert gold == labels


def test_run_fine_tuning_end_to_end_cpu(monkeypatch, tmp_path, mock_backbone, mock_tokenizer):
    """Verify full run_fine_tuning orchestration on CPU with deduplication and MLflow mocks."""
    monkeypatch.setattr(bert_mod.AutoModel, "from_pretrained", lambda *a, **kw: mock_backbone)
    monkeypatch.setattr(bert_mod.AutoTokenizer, "from_pretrained", lambda *a, **kw: mock_tokenizer)

    class _DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bert_mod.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(bert_mod.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(bert_mod.mlflow, "start_run", lambda *a, **kw: _DummyRun())
    monkeypatch.setattr(bert_mod.mlflow, "log_params", lambda *a, **kw: None)
    monkeypatch.setattr(bert_mod.mlflow, "log_metrics", lambda *a, **kw: None)
    monkeypatch.setattr(bert_mod.mlflow, "log_artifact", lambda *a, **kw: None)

    clean_csv = tmp_path / "leis_clean.csv"
    feat_csv = tmp_path / "leis_features.csv"
    save_dir = tmp_path / "models"

    # 12 records with 10 unique IDs (2 duplicates)
    ids = [f"L_{i % 10}" for i in range(12)]
    pd.DataFrame(
        {"id": ids, "texto_original": [f"Art {i} lei federal texto" for i in range(12)]}
    ).to_csv(clean_csv, index=False)
    pd.DataFrame({"id": ids, "teve_veto": [i % 2 for i in range(12)]}).to_csv(
        feat_csv, index=False
    )

    best_f1 = run_fine_tuning(
        data_path=str(clean_csv),
        features_path=str(feat_csv),
        target_column="teve_veto",
        model_name="mock-bertimbau",
        epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        save_dir=str(save_dir),
    )
    assert 0.0 <= best_f1 <= 1.0
    assert (save_dir / "best_legislabl_bert_teve_veto.pt").exists()
