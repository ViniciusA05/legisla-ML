# TESTS — Validation Strategy & Quality Gates (FEAT-001)

| Metadata | Detail |
| :--- | :--- |
| **Feature** | BERTimbau Fine-Tuning for Veto Risk and Normative Instability Prediction |
| **Identifier** | `FEAT-001` |
| **Evaluation Stack** | PyTest (CI/TDD seam suite), Scikit-Learn Metrics, MLflow Tracking |

---

## 1. Public Seams Under Test

Following the repository TDD principles (`tdd` skill), verification targets three public seams:

| Seam | Public Interface | What It Catches |
| :--- | :--- | :--- |
| **Seam 1: Head-Tail Tokenization** | `LegislaBERTDataset.__getitem__` | Sequence overflow ($>512$), missing `[CLS]`/`[SEP]` delimiters, loss of tail clauses, or malformed padding masks. |
| **Seam 2: Classifier Forward Contract** | `LegislaBERTClassifier.forward` | Shape mismatches on `logits` $(B, 2)$ or `cls_embedding` $(B, 768)$, and `NaN`/`Inf` propagation under `fp16` autocast. |
| **Seam 3: Class Imbalance Weighting** | `calculate_class_weights` | Incorrect inverse-frequency weighting ($w_1 \le w_0$ when $N_1 < N_0$) or division-by-zero on single-class batches. |

---

## 2. Test Cases

### TC01: Head-Tail Window Boundary Enforcement
- **Given**: A synthetic statutory string exceeding 1,500 tokens with distinct markers at the head and tail.
- **When**: Indexed via `LegislaBERTDataset[0]`.
- **Then**:
  - `input_ids.shape == (512,)` and `attention_mask.shape == (512,)`.
  - `input_ids[0] == tokenizer.cls_token_id` and `input_ids[-1] == tokenizer.sep_token_id`.
  - Both head and tail marker tokens are present in `input_ids`.

### TC02: Latent Embedding & Logit Shape Invariance
- **Given**: A batch of $B = 2$ tokenized statutes ($2 \times 512$).
- **When**: Passed through `LegislaBERTClassifier(input_ids, attention_mask)`.
- **Then**:
  - `logits.shape == (2, 2)` and `cls_embedding.shape == (2, 768)`.
  - `torch.isnan(logits).any() == False`.

### TC03: Inverse-Frequency Weight Ordering
- **Given**: An imbalanced label array `y = np.array([0, 0, 0, 0, 1])` ($N_0 = 4, N_1 = 1$).
- **When**: Passed to `calculate_class_weights(y)`.
- **Then**: `weights[1] > weights[0]` ($w_1 = 2.5, w_0 = 0.625$).

---

## 3. Quality Gates (Definition of Done)

| Indicator | Minimum Threshold | Target Goal | Remediation if Below Threshold |
| :--- | :--- | :--- | :--- |
| **Validation Macro F1-Score** | $\ge 0.90$ | $\ge 0.95$ | Tune learning rate to $1 \times 10^{-5}$ or extend epochs to 8. |
| **Minority Recall (`teve_veto = 1`)** | $\ge 0.85$ | $\ge 0.95$ | Increase minority class weight multiplier in `CrossEntropyLoss`. |
| **Peak CUDA VRAM Usage** | $< 3.8\text{ GB}$ | $\le 2.8\text{ GB}$ | Reduce `--batch-size` to `2` and increase `--grad-accum` to `8`. |
| **MLflow Artifact Persistence** | 1 `.pt` checkpoint logged | Registered in `legislabl-bert-v2` | Verify write permissions in `models/` and MLflow tracking URI. |
