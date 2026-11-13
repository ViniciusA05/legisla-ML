# LegislaML

Predictive quality modeling, semantic diffing, and structural analysis of Brazilian federal legislation.

## Language

### Legislative Structure (LC 95/1998)

**Statute** (`Norma` / `Lei`):
A federal ordinary (`LEI`) or supplementary (`LCP`) law enacted by the Brazilian National Congress.
_Avoid_: Bill, decree, regulation

**Ementa**:
The statutory header paragraph that summarizes the core subject matter and scope of a Statute.
_Avoid_: Abstract, summary, preamble

**Caput**:
The main body of an Article (`Artigo`) that states the primary legal rule before any exceptions or subdivisions (`Parágrafo`, `Inciso`, `Alínea`).
_Avoid_: Article header, main paragraph

**Vide**:
A relational metadata entry from the Federal Senate API recording subsequent normative events (vetoes, amendments, or revocations) affecting a Statute.
_Avoid_: Cross-reference, citation

### Predictive Targets & Anomaly Detection

**Veto Risk** (`teve_veto`):
Binary target indicating whether a Statute received a partial or total presidential veto message upon enactment.
_Avoid_: Rejection, cancellation

**Normative Instability** (`instavel`):
Binary target indicating whether a Statute required subsequent statutory amendments or revocations due to drafting ambiguity or conflict.
_Avoid_: Volatility, churn

**Legislative Rider** (`Jabuti`):
An extraneous provision inserted into a substitute bill (`Substitutivo`) whose semantic embedding exhibits high cosine distance from the original Ementa.
_Avoid_: Out-of-scope clause, amendment anomaly

**Head-Tail Window**:
A 512-token sampling window combining the first 128 tokens (Ementa and preliminary scope) and the final 382 tokens (penalties, transitional rules, and enforcement clauses) of a Statute.
_Avoid_: Random crop, simple truncation

**Positive-Class Score** (`score_positive`):
The classifier output for the positive class of a Predictive Target (Veto Risk or Normative Instability), trained under class-weighted loss; a ranking score on a balanced prior, not a calibrated probability.
_Avoid_: Probability, confidence, risk percentage

**Slim Checkpoint**:
A serving artifact holding only model weights, configuration, validation metrics, and the Predictive Target it was trained for.
_Avoid_: Model file, full checkpoint
