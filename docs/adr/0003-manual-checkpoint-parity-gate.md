---
status: accepted
---

# Verify Slim Checkpoint Serving Parity via a manual local gate

Before promoting `develop` to `main`, the exported Slim Checkpoint must reproduce the training-time validation Macro-F1 within a bounded tolerance (`|F1_serving - F1_val| <= 0.002`) via `scripts/verify_parity.py` over the full serving stack (`TestClient` or live HTTP container), with the measured output recorded in `TESTS.md`. We rejected running parity checks inside GitHub Actions CI because it would require publishing or baking the ~416 MiB Slim Checkpoint and raw corpus (contradicting ADR-0001 and the `< 90 s` CI budget), and rejected strict bitwise metric equality across GPU training and CPU inference. The ±0.002 margin absorbs floating-point boundary noise while catching tokenization, windowing, weight-loading, or threshold defects; the trade-off is that parity remains a manual release gate whenever a new checkpoint is exported.
