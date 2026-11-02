# QUICKSTART — Execution & Verification Guide (FEAT-001)

| Metadata | Detail |
| :--- | :--- |
| **Feature** | BERTimbau Fine-Tuning for Veto Risk and Normative Instability Prediction |
| **Identifier** | `FEAT-001` |
| **Estimated Runtime** | ~10–15 minutes (5 epochs under mixed precision) |

---

## 1. Environment Verification

From the repository root, activate the virtual environment and confirm CUDA visibility:

```powershell
cd legisla-ML
.\.venv\Scripts\Activate.ps1
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

---

## 2. Corpus Preprocessing & Feature Extraction

Ensure `data/processed/leis_clean.csv` and `data/processed/leis_features.csv` are hydrated from `data/raw/`:

```powershell
# 1. Clean statutory text, tokenize sentences, and lemmatize via spaCy
python src/preprocessing.py

# 2. Extract surface features and target labels (teve_veto, instavel)
python src/features.py
```

---

## 3. Execute Fine-Tuning Pipeline

Run the fine-tuning script with memory-constrained CUDA defaults ($B=4, A=4$, AMP `fp16`):

```powershell
# Train for Veto Risk prediction
python src/train_legislabl_bert.py --epochs 5 --batch-size 4 --grad-accum 4 --lr 2e-5 --target teve_veto

# Train for Normative Instability prediction
python src/train_legislabl_bert.py --epochs 5 --batch-size 4 --grad-accum 4 --lr 2e-5 --target instavel
```

---

## 4. Inspect Experiments in MLflow UI

Launch the MLflow tracking server from the repository root:

```powershell
mlflow ui
```

Open `http://localhost:5000` in your browser:
1. Select the **`legislabl-bert-v2`** experiment.
2. Compare `train_loss`, `val_loss`, `val_f1_macro`, and `val_recall_veto` across epochs.
3. Verify the persisted checkpoint artifact under `models/best_legislabl_bert_<target>.pt`.
