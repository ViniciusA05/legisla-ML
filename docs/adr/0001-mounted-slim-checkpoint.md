---
status: accepted
---

# Serve the model from a mounted Slim Checkpoint

The inference image never contains model weights: the Slim Checkpoint is mounted read-only at runtime (`LEGISLA_CHECKPOINT_PATH`), and only the tokenizer and config are baked in at build time. We rejected baking weights into the image (a ~416 MiB layer per model version that couples releases to retraining) and pulling from the MLflow Model Registry at startup (a network and auth dependency before any registry-promotion workflow exists). The cost is that the operator must provision the volume; Registry loading is deferred to a later feature.
