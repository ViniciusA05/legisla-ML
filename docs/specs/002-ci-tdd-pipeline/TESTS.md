# TESTS — Verification & Coverage Report (FEAT-002)

| Metadata | Detail |
| :--- | :--- |
| **Feature** | Continuous Integration, Automated TDD Suite, and Fast Quality Gates |
| **Identifier** | `FEAT-002` |
| **Total Tests** | 13 unit tests across 5 public seams (`100% PASS`) |
| **Total Coverage** | **88.74%** (Gate threshold: `>= 85.00%`) |
| **Execution Time** | `12.48s` on CPU (SLA threshold: `< 35.00s`) |

---

## 1. Seam-to-Test Traceability Matrix

| Seam | Test File | Test Cases | Invariant Verified |
| :--- | :--- | :--- | :--- |
| **Seam 1 (Dataset)** | `tests/unit/test_dataset.py` | `test_short_sequence_padding_and_special_tokens`<br>`test_long_sequence_head_tail_window_boundary` | `[CLS]` at index 0, `[SEP]` boundary, zero-padding on short texts, and exact `128 head + 382 tail` slicing on texts >510 tokens. |
| **Seam 2 (Model & Loop)** | `tests/unit/test_model.py` | `test_classifier_forward_shapes_and_gradients`<br>`test_train_epoch_and_evaluate_on_cpu`<br>`test_run_fine_tuning_end_to_end_cpu` | Logits shape `(B, 2)`, CLS embedding `(B, 768)`, absence of `NaN`/`Inf`, valid backward gradients, and end-to-end CPU training/checkpointing with mocked MLflow. |
| **Seam 3 (Class Weights)** | `tests/unit/test_weights.py` | `test_inverse_frequency_minority_prioritization`<br>`test_balanced_classes_and_zero_division_guard`<br>`test_set_seed_reproducibility` | Exact formula $w_c = N / (2 N_c)$, minority prioritization ($w_1 > w_0$), weighted sum conservation, zero-count guard, and RNG seed determinism. |
| **Seam 4 (Deduplication)** | `tests/unit/test_deduplication.py` | `test_dataframe_id_deduplication_contract`<br>`test_model_pipeline_deduplicates_and_evaluates`<br>`test_scraper_methods_and_deduplication` | Zero duplicate `id` leakage across raw scraping, feature tables, and classic model evaluation. |
| **Seam 5 (Preprocessing)** | `tests/unit/test_preprocessing.py` | `test_clean_text_and_process_document`<br>`test_feature_extractor_metrics_and_pipeline` | Removal of `[Detalhes da Norma]` and official gazette disclaimers, Flesch index bounds `[0, 100]`, and `vides` veto/instability detection. |

---

## 2. Empirical Coverage Breakdown

```text
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/features.py                 111     13    88%   60, 68, 73, 77, 84, 94-95, 99, 112, 153, 186-188
src/model.py                     71      8    89%   92, 104, 116, 119-124
src/preprocessing.py             59      6    90%   50, 58, 83, 86-88
src/scraper.py                  105     21    80%   27-28, 36-37, 50-52, 61, 65-67, 76, 81, 84, 92, 96, 100, 129, 132-134
src/train_legislabl_bert.py     169     10    94%   38, 244, 376-398
-----------------------------------------------------------
TOTAL                           515     58    89%
Required test coverage of 85% reached. Total coverage: 88.74%
======================= 13 passed in 12.48s =======================
```
