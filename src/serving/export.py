"""Export a training checkpoint into a slim, safely loadable serving artifact (FR05)."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from src.serving import VALID_TARGETS
from src.serving.checkpoint import CheckpointError


def _to_plain(value):
    """Recursively convert NumPy/torch scalars to plain Python types."""
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def export_slim_checkpoint(src: str | Path, dst: str | Path, target: str) -> Path:
    """
    Strip ``optimizer_state_dict`` and write ``{model_state_dict, config, metrics,
    target, sha256_of_source}``.

    The source file is produced by our own training script (trusted context), so it
    is read with ``weights_only=False`` because it embeds NumPy scalar metrics. The
    exported artifact contains plain types only and loads with ``weights_only=True``.
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"Unknown target {target!r}; expected one of {VALID_TARGETS}")

    src, dst = Path(src), Path(dst)
    payload = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "config"):
        if not isinstance(payload, dict) or key not in payload:
            raise CheckpointError(f"Source checkpoint missing key: {key}")

    slim = {
        "model_state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in payload["model_state_dict"].items()
        },
        "config": _to_plain(payload["config"]),
        "metrics": _to_plain(payload.get("metrics", {})),
        "target": target,
        "sha256_of_source": hashlib.sha256(src.read_bytes()).hexdigest(),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a slim LegislaBERT serving checkpoint")
    parser.add_argument("--src", required=True, help="Training checkpoint (.pt)")
    parser.add_argument("--dst", required=True, help="Slim checkpoint output path")
    parser.add_argument("--target", required=True, choices=VALID_TARGETS)
    args = parser.parse_args(argv)

    out = export_slim_checkpoint(args.src, args.dst, args.target)
    print(f"[export] Slim checkpoint written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
