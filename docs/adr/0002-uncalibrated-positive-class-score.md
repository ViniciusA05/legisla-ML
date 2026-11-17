---
status: accepted
---

# Expose an uncalibrated Positive-Class Score, not a probability

The API returns `score_positive` and documents it as a ranking score, never as `probability`. Training with inverse-frequency class weights makes the network learn the posterior of a re-weighted prior (`p(y|x)` is proportional to `pi_y * p_weighted(y|x)`), so a score of 0.55 does not mean a 55% chance of veto in the real population. Renaming later would break API consumers, and the default threshold of 0.5 matches the Macro-F1 selection criterion. Proper calibration (Platt, isotonic, or prior correction) is deferred until a gold-standard sample exists.
