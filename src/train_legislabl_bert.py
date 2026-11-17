"""
LegislaBERT Fine-Tuning Pipeline (FEAT-001).

Trains a BERTimbau sequence classifier with Head-Tail Window sampling (512 tokens),
inverse-frequency weighted cross-entropy loss, mixed-precision (fp16) gradient
accumulation, and continuous MLflow tracking.
"""

import argparse
import os
import random

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.legislabert import LegislaBERTClassifier, LegislaBERTDataset

__all__ = [
    "AutoModel",
    "AutoTokenizer",
    "LegislaBERTClassifier",
    "LegislaBERTDataset",
]


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights(y: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Compute inverse-frequency class weights: w_c = N_total / (C * N_c)."""
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 4,
) -> float:
    """Execute one training epoch with mixed precision (fp16) and gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(len(dataloader), 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evaluate model on the validation split and return classification metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision_1 = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall_1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )

    metrics = {
        "loss": float(avg_loss),
        "f1_macro": float(f1_macro),
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "precision_veto": float(precision_1),
        "recall_veto": float(recall_1),
        "classification_report": report,
    }
    return metrics, all_preds, all_labels


def run_fine_tuning(
    data_path: str = "data/processed/leis_clean.csv",
    features_path: str = "data/processed/leis_features.csv",
    target_column: str = "teve_veto",
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-5,
    save_dir: str = "models",
    seed: int = 42,
) -> float:
    """Run the end-to-end LegislaBERT fine-tuning pipeline with MLflow tracking."""
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LegislaBERT] Execution device: {device}")

    if not os.path.exists(data_path) or not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Required processed files missing at {data_path} or {features_path}."
        )

    df_clean = pd.read_csv(data_path)
    df_feat = pd.read_csv(features_path)

    df = df_clean.merge(df_feat[["id", target_column]], on="id")
    df = df.dropna(subset=["texto_original", target_column]).drop_duplicates(
        subset=["id"]
    )
    df[target_column] = df[target_column].astype(int)

    texts = df["texto_original"].tolist()
    labels = df[target_column].to_numpy(dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    print(
        f"[LegislaBERT] Loaded {len(df)} unique statutes | "
        f"Target '{target_column}' distribution: {class_counts.tolist()}"
    )

    can_stratify = len(np.unique(labels)) > 1 and np.min(class_counts) >= 2
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = LegislaBERTDataset(train_texts, train_labels, tokenizer)
    val_dataset = LegislaBERTDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LegislaBERTClassifier(model_name=model_name, num_classes=2).to(device)

    class_weights = calculate_class_weights(train_labels, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    steps_per_epoch = max(len(train_loader) // grad_accum_steps, 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    mlflow.set_experiment("legislabl-bert-v2")
    with mlflow.start_run(run_name=f"bert-finetuning-{target_column}"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "target": target_column,
                "epochs": epochs,
                "batch_size_per_step": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": batch_size * grad_accum_steps,
                "learning_rate": learning_rate,
                "class_weights": class_weights.cpu().numpy().tolist(),
                "train_samples": len(train_texts),
                "val_samples": len(val_texts),
                "seed": seed,
            }
        )

        best_f1 = -1.0
        best_model_path = os.path.join(
            save_dir, f"best_legislabl_bert_{target_column}.pt"
        )

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                grad_accum_steps,
            )
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Macro-F1: {val_metrics['f1_macro']:.4f} | "
                f"Val Minority Recall: {val_metrics['recall_veto']:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_balanced_acc": val_metrics["balanced_accuracy"],
                    "val_precision_veto": val_metrics["precision_veto"],
                    "val_recall_veto": val_metrics["recall_veto"],
                },
                step=epoch,
            )

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "config": {"model_name": model_name, "max_length": 512},
                    },
                    best_model_path,
                )
                mlflow.log_artifact(best_model_path)
                print(
                    f"[Checkpoint] New best Macro-F1 ({best_f1:.4f}) saved to {best_model_path}"
                )

        print(f"[LegislaBERT] Training complete. Best Macro-F1: {best_f1:.4f}")
        return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LegislaBERT Fine-Tuning Pipeline with MLflow Tracking"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Physical micro-batch size per forward pass"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="AdamW initial learning rate"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="teve_veto",
        choices=["teve_veto", "instavel"],
        help="Target variable to predict",
    )
    args = parser.parse_args()

    run_fine_tuning(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        target_column=args.target,
    )
