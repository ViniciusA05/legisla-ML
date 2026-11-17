"""Typed runtime configuration parsed from environment variables (FEAT-003 FR08)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.serving import VALID_TARGETS


class SettingsError(ValueError):
    """Raised when an environment variable holds an invalid value."""


@dataclass(frozen=True)
class Settings:
    checkpoint_path: Path = Path("/models/model.pt")
    target: str | None = None
    tokenizer_dir: Path = Path("/opt/legisla/tokenizer")
    decision_threshold: float = 0.5
    max_batch: int = 16
    max_chars: int = 500_000
    torch_num_threads: int = 0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        defaults = cls()

        def parse(name: str, cast, default):
            raw = env.get(name)
            if raw is None or raw == "":
                return default
            try:
                return cast(raw)
            except ValueError as exc:
                raise SettingsError(f"Invalid value for {name}: {raw!r}") from exc

        settings = cls(
            checkpoint_path=parse("LEGISLA_CHECKPOINT_PATH", Path, defaults.checkpoint_path),
            target=parse("LEGISLA_TARGET", str, None),
            tokenizer_dir=parse("LEGISLA_TOKENIZER_DIR", Path, defaults.tokenizer_dir),
            decision_threshold=parse("DECISION_THRESHOLD", float, defaults.decision_threshold),
            max_batch=parse("MAX_BATCH", int, defaults.max_batch),
            max_chars=parse("MAX_CHARS", int, defaults.max_chars),
            torch_num_threads=parse("TORCH_NUM_THREADS", int, defaults.torch_num_threads),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.target is not None and self.target not in VALID_TARGETS:
            raise SettingsError(f"LEGISLA_TARGET must be one of {VALID_TARGETS}")
        if not 0.0 < self.decision_threshold < 1.0:
            raise SettingsError("DECISION_THRESHOLD must be in (0, 1)")
        if self.max_batch < 1:
            raise SettingsError("MAX_BATCH must be >= 1")
        if self.max_chars < 1:
            raise SettingsError("MAX_CHARS must be >= 1")
        if self.torch_num_threads < 0:
            raise SettingsError("TORCH_NUM_THREADS must be >= 0")
