"""Slim checkpoint loading with fail-fast validation (FEAT-003 FR03, FR04, I5)."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.legislabert import MAX_LENGTH
from src.serving import VALID_TARGETS

REQUIRED_KEYS = ("model_state_dict", "config", "target")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint violates the serving contract."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: nn.Module
    target: str
    model_name: str
    max_length: int
    sha256: str
    metrics: dict


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_slim_checkpoint(
    path: str | Path,
    model_factory: Callable[[], nn.Module],
    expected_target: str | None = None,
) -> LoadedCheckpoint:
    """
    Validate and load a slim checkpoint into a freshly built, eval-mode model.

    ``model_factory`` builds the architecture without downloading weights; the
    checkpoint must then overwrite every parameter (``strict=True``). Loading uses
    ``weights_only=True`` so no arbitrary pickle code can execute in the container.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found at {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any read failure must abort startup
        raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a dict")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise CheckpointError(f"Checkpoint missing keys: {missing}")

    config = payload["config"]
    max_length = config.get("max_length") if isinstance(config, dict) else None
    if max_length != MAX_LENGTH:
        raise CheckpointError(f"Unsupported max_length {max_length!r}; expected {MAX_LENGTH}")

    target = payload["target"]
    if target not in VALID_TARGETS:
        raise CheckpointError(f"Unknown checkpoint target {target!r}")
    if expected_target is not None and expected_target != target:
        raise CheckpointError(
            f"Configured target {expected_target!r} does not match checkpoint target {target!r}"
        )

    model = model_factory()
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"state_dict does not match the architecture: {exc}") from exc

    model.eval()
    return LoadedCheckpoint(
        model=model,
        target=target,
        model_name=str(config.get("model_name", "")),
        max_length=max_length,
        sha256=sha256_of_file(path),
        metrics=dict(payload.get("metrics", {})),
    )
