# IMPLEMENTATION — Technical Architecture & Code Documentation (FEAT-001)

| Metadata | Detail |
| :--- | :--- |
| **Feature** | BERTimbau Fine-Tuning for Veto Risk and Normative Instability Prediction |
| **Identifier** | `FEAT-001` |
| **Primary Module** | `src/train_legislabl_bert.py` |
| **Stack** | Python 3.12, PyTorch 2.5+, HuggingFace Transformers, MLflow |

---

## 1. Modular Component Architecture

`src/train_legislabl_bert.py` is structured into four decoupled components:

### 1.1. Dataset & Sampling: `LegislaBERTDataset`
Implements `torch.utils.data.Dataset` with **Head-Tail Window** sampling:
1. Raw statutory text is tokenized without special tokens (`add_special_tokens=False`).
2. The usable token budget is $512 - 2 = 510$ tokens (reserving slots for `[CLS]` and `[SEP]`).
3. If the statute exceeds 510 tokens:
   - `head = tokens[:128]` (*Ementa* and preliminary scope).
   - `tail = tokens[-382:]` (penalties, revocation, and enforcement clauses).
   - `selected_tokens = head + tail`.
4. Prepends `[CLS]`, appends `[SEP]`, pads short sequences to 512, and emits `input_ids`, `attention_mask`, and `label` tensors.

### 1.2. Neural Architecture: `LegislaBERTClassifier`
Subclasses `torch.nn.Module`:
- **Backbone**: `AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")`.
- **Latent Extraction**: Slices `outputs.last_hidden_state[:, 0, :]` ($\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{768}$).
- **Regularization & Head**: `nn.Dropout(p=0.3)` followed by `nn.Linear(768, 2)`.
- **Dual Return**: Emits `(logits, cls_embedding)` so downstream v3 semantic-diff modules can extract the fine-tuned $\mathbb{R}^{768}$ statutory embedding directly.

### 1.3. Optimization & Memory-Constrained Execution
- **`calculate_class_weights`**: Computes inverse-frequency weights $w_c = \frac{N_{\text{total}}}{C \cdot N_c}$ injected into `nn.CrossEntropyLoss`.
- **`train_epoch`**:
  - Executes forward passes under `torch.cuda.amp.autocast()` (`fp16`).
  - Scales loss by `1 / grad_accum_steps` and accumulates gradients across $A = 4$ micro-batches ($B = 4 \implies B_{\text{effective}} = 16$).
  - Applies $L_2$ gradient clipping (`max_norm=1.0`) prior to `scaler.step(optimizer)` and linear warmup scheduling.

### 1.4. Continuous MLOps Tracking
- Logs hyperparameters, class weight vectors, and dataset split sizes to the `legislabl-bert-v2` MLflow experiment.
- Logs `train_loss`, `val_loss`, `val_f1_macro`, `val_accuracy`, `val_balanced_acc`, `val_precision_veto`, and `val_recall_veto` at each epoch.
- Saves and logs `models/best_legislabl_bert_<target>.pt` whenever validation Macro-F1 improves.

---

## 2. CLI Interface

```bash
python src/train_legislabl_bert.py \
  --epochs 5 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-5 \
  --target teve_veto
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `5` | Number of full training epochs |
| `--batch-size` | `int` | `4` | Physical micro-batch size per forward pass |
| `--grad-accum` | `int` | `4` | Gradient accumulation steps ($B_{\text{effective}} = B \times A$) |
| `--lr` | `float` | `2e-5` | Initial learning rate for `AdamW` |
| `--target` | `str` | `teve_veto` | Target variable (`teve_veto` or `instavel`) |
