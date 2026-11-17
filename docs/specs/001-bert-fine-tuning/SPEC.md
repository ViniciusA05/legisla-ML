# SPEC — LegislaBERT Fine-Tuning and Continuous MLOps Governance

| Metadata | Detail |
| :--- | :--- |
| **Feature** | BERTimbau Fine-Tuning for Veto Risk and Normative Instability Prediction |
| **Identifier** | `FEAT-001` |
| **Role** | ML Engineer / Tech Lead |
| **Status** | Phase 1 — Ready for Review |
| **Date** | 2026-11-02 |
| **Branch** | `feat/data-hydration` ➔ `develop` |

---

## 1. Problem Statement & Motivation

**LegislaML v1** established a classical tabular baseline (Random Forest and XGBoost) achieving 97.6% accuracy using six surface-level lexical metrics (Flesch readability, cross-reference density, Type-Token Ratio). While effective on surface statistics, the classical baseline is blind to contextual semantics and the hierarchical dependencies mandated by Brazilian Complementary Law No. 95/1998 (`LC 95/1998`).

**FEAT-001 (LegislaML v2)** transitions the representation to **dense contextual Deep Learning** by fine-tuning `neuralmind/bert-base-portuguese-cased` (`BERTimbau`) to map full statutory texts into latent vectors $\mathbf{h} \in \mathbb{R}^{768}$, predicting two targets:
1. `teve_veto` (**Veto Risk**): Probability that a Statute receives a partial or total presidential veto.
2. `instavel` (**Normative Instability**): Probability that a Statute requires subsequent amendments or revocations.

---

## 2. System Requirements

### 2.1. Functional Requirements (FR)
- **FR01**: Load statutory text from `data/processed/leis_clean.csv` and labels from `data/processed/leis_features.csv`, mapping sequences into BERTimbau token IDs and attention masks.
- **FR02**: Implement structured **Head-Tail Window** sampling ($L_{\max} = 512$) preserving the first 128 tokens (*Ementa* and preliminary scope) and the final 382 tokens (penalties, revocation, and enforcement clauses).
- **FR03**: Extract the final hidden state of the `[CLS]` token as the latent statutory embedding $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{768}$.
- **FR04**: Apply stochastic regularization (`Dropout(p=0.3)`) followed by an affine linear projection $\mathbb{R}^{768} \to \mathbb{R}^2$.
- **FR05**: Compute inverse-frequency weighted cross-entropy loss to penalize false negatives on the minority class.
- **FR06**: Log hyperparameters, per-epoch loss/metrics, and the highest-F1 checkpoint artifact to MLflow under the `legislabl-bert-v2` experiment.

### 2.2. Non-Functional & Resource Constraints (NFR)
- **NFR01 (VRAM Ceiling)**: Operate strictly within a 4 GB CUDA VRAM envelope without triggering Out-Of-Memory (`CUDA OOM`) faults.
- **NFR02 (Effective Batch Size)**: Maintain statistical gradient stability at $B_{\text{effective}} = 16$ via physical micro-batches ($B = 4$) and gradient accumulation ($A = 4$).
- **NFR03 (Mixed Precision)**: Execute forward passes under Automatic Mixed Precision (`fp16` via `torch.cuda.amp.autocast()`) with `GradScaler()`.
- **NFR04 (Reproducibility)**: Fix random seeds (`seed=42`) across Python, NumPy, and PyTorch with stratified 80/20 train/validation splits.

---

## 3. Mathematical Formulation (Multivariable Calculus & Optimization)

### 3.1. Latent Embedding Space
Given a tokenized statute $x = (t_{\text{CLS}}, t_1, \dots, t_{510}, t_{\text{SEP}})$, the encoder outputs:
$$\mathbf{H} = \text{BERT}(x) \in \mathbb{R}^{512 \times 768}, \quad \mathbf{h}_{\text{CLS}} = \mathbf{H}[0, :] \in \mathbb{R}^{768}$$

### 3.2. Classification Head & Posterior Probabilities
$$\mathbf{z} = \mathbf{W}^T (\text{Dropout}(\mathbf{h}_{\text{CLS}})) + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{768 \times 2}, \mathbf{b} \in \mathbb{R}^2$$
$$\hat{y}_c = \frac{e^{z_c}}{\sum_{j=0}^1 e^{z_j}}, \quad c \in \{0, 1\}$$

### 3.3. Inverse-Frequency Weighted Cross-Entropy Loss
Given class counts $N_0 \gg N_1$ in the training set:
$$w_c = \frac{N_{\text{total}}}{2 \cdot N_c}$$
$$\mathcal{L}(\boldsymbol{\theta}) = - \frac{1}{B} \sum_{i=1}^B \sum_{c=0}^1 w_c \cdot y_{i,c} \log(\hat{y}_{i,c})$$

### 3.4. Multivariable Gradient Dynamics (`AdamW`)
The gradient $\nabla_{\boldsymbol{\theta}} \mathcal{L} \in \mathbb{R}^P$ ($P \approx 1.1 \times 10^8$) updates parameters via decoupled weight decay ($\eta = 2 \times 10^{-5}$, $\lambda = 0.01$) and $L_2$ norm clipping ($\|\mathbf{g}\|_2 \le 1.0$):
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t - \eta \lambda \boldsymbol{\theta}_t$$

---

## 4. Architectural Trade-Off Matrix

| Component | Selected Option | Discarded Alternative | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Backbone** | `neuralmind/bert-base-portuguese-cased` | Uncased BERT / Local LLaMA | Case sensitivity preserves legal acronyms; 110M params fit in 4 GB VRAM. |
| **Sampling** | Head (128) + Tail (382) | Naive Head-512 Truncation | Retains both the *Ementa* and final sanction/revocation clauses where vetoes concentrate. |
| **Loss** | Weighted Cross-Entropy | Uniform Cross-Entropy | Prevents trivial majority-class collapse under severe veto imbalance. |
| **Memory** | $B=4, A=4$ + AMP `fp16` | $B=16$ in `fp32` | Halves activation memory while preserving $B_{\text{effective}} = 16$ gradient statistics. |

---

## 5. Quality Gates (Definition of Done)

| Metric / Criterion | Minimum Threshold | Target Goal |
| :--- | :--- | :--- |
| **Validation Macro F1-Score** | $\ge 0.90$ | $\ge 0.95$ |
| **Minority Class Recall (`teve_veto = 1`)** | $\ge 0.85$ | $\ge 0.95$ |
| **Peak CUDA VRAM Allocation** | $< 3.8\text{ GB}$ | $\le 2.8\text{ GB}$ |
| **MLflow Experiment Tracking** | 100% of epochs & best `.pt` logged | Registered in `legislabl-bert-v2` |
