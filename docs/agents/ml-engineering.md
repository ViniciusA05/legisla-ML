# ML & MLOps Engineering Playbook

Standards for designing, training, and evaluating classical ML and deep learning models in LegislaML.

## 1. First-Principles Mathematical Formulation

- **Latent Space Geometry**: Express embeddings explicitly as vectors $\mathbf{h} \in \mathbb{R}^d$ (with $d = 768$ for BERTimbau). Define similarity operations via inner products $\langle \mathbf{u}, \mathbf{v} \rangle$, $L_2$ norms $\|\mathbf{u}\|_2$, and cosine projections.
- **Multivariable Optimization**: Formulate loss surfaces $\mathcal{L}: \mathbb{R}^P \to \mathbb{R}$, gradients $\nabla_{\boldsymbol{\theta}} \mathcal{L}$, and first/second-moment adaptive updates (`AdamW` with weight decay $\lambda = 0.01$ and gradient norm clipping $\|\mathbf{g}\|_2 \le 1.0$).
- **Class Imbalance Mitigation**: Weight the cross-entropy criterion inversely to class frequency on the training split:
  $$w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$

## 2. Legal Context Window & Tokenization

- **Backbone**: Default to `neuralmind/bert-base-portuguese-cased` (preserving case sensitivity for legal acronyms and proper nouns).
- **Head + Tail Truncation ($L_{\max} = 512$)**: Because self-attention scales as $O(L^2)$, sample long statutes by preserving the first $128$ tokens (head: *ementa* and preliminary scope) and the last $382$ tokens (tail: penalties, revocation, and enforcement clauses), framed by `[CLS]` and `[SEP]`.

## 3. Memory-Constrained CUDA Execution Defaults

Design all PyTorch training loops to run reliably under a 4 GB VRAM ceiling without hardware-specific assumptions:

- **Physical Batch Size**: Default to `--batch-size 4` per forward pass.
- **Gradient Accumulation**: Default to `--grad-accum 4` to maintain an effective statistical batch size of $B_{\text{effective}} = 16$.
- **Mixed Precision**: Wrap forward passes in `torch.cuda.amp.autocast()` (`fp16`) and scale gradients via `GradScaler()`.
- **Determinism**: Seed `random`, `numpy`, and `torch` (`seed=42`) and use stratified train/validation splits.

## 4. Continuous MLOps Tracking (MLflow)

Every training run instruments MLflow under the `legislabl-bert-v2` experiment:

- **Parameters**: Log backbone name, target variable, physical batch size, accumulation steps, effective batch size, learning rate, and computed class weights before epoch 1.
- **Per-Epoch Metrics**: Log `train_loss`, `val_loss`, `val_f1_macro`, `val_balanced_acc`, `val_precision_veto`, and `val_recall_veto` at each step.
- **Artifact Checkpointing**: Persist and log `models/best_legislabl_bert_<target>.pt` whenever `val_f1_macro` reaches a new maximum.
