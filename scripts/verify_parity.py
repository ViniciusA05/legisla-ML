"""Verify QG6 serving parity on the held-out validation split (FEAT-003).

Usage (in-process TestClient, offline):
    .venv/bin/python scripts/verify_parity.py \
        --checkpoint models/slim_teve_veto.pt \
        --tokenizer-dir .local/tokenizer

Usage (running server or container):
    .venv/bin/python scripts/verify_parity.py --url http://127.0.0.1:8000
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_validation_split(
    data_path: Path,
    features_path: Path,
    target: str,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    """Reproduce the exact deduplicated validation split from train_legislabl_bert.py."""
    if not data_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(
            f"Required processed files missing at {data_path} or {features_path}."
        )

    df_clean = pd.read_csv(data_path)
    df_feat = pd.read_csv(features_path)
    df = df_clean.merge(df_feat[["id", target]], on="id")
    df = df.dropna(subset=["texto_original", target]).drop_duplicates(subset=["id"])
    df[target] = df[target].astype(int)

    texts = df["texto_original"].tolist()
    labels = df[target].to_numpy(dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    can_stratify = len(np.unique(labels)) > 1 and np.min(class_counts) >= 2

    _, val_texts, _, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )
    return val_texts, val_labels


def _http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"Unexpected status {response.status} from {url}")
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _score_via_http(
    base_url: str,
    texts: list[str],
    batch_size: int,
    target: str,
) -> tuple[dict, list[int]]:
    base = base_url.rstrip("/")
    _http_json("GET", f"{base}/readyz")
    model_info = _http_json("GET", f"{base}/v1/model")
    if model_info.get("target") != target:
        raise RuntimeError(
            f"Service target {model_info.get('target')!r} does not match {target!r}"
        )

    preds: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        body = _http_json("POST", f"{base}/v1/predict/batch", {"texts": batch})
        preds.extend(int(item["label"]) for item in body["predictions"])
    return model_info, preds


def _score_via_testclient(
    checkpoint: Path,
    tokenizer_dir: Path,
    target: str,
    threshold: float,
    batch_size: int,
    threads: int,
    texts: list[str],
) -> tuple[dict, list[int], str | None]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    from fastapi.testclient import TestClient

    from src.serving.app import create_app
    from src.serving.settings import Settings

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_sha256 = payload.get("sha256_of_source") if isinstance(payload, dict) else None

    settings = Settings(
        checkpoint_path=checkpoint,
        tokenizer_dir=tokenizer_dir,
        target=target,
        decision_threshold=threshold,
        max_batch=batch_size,
        torch_num_threads=threads,
    )

    preds: list[int] = []
    with TestClient(create_app(settings=settings)) as client:
        ready = client.get("/readyz")
        if ready.status_code != 200:
            raise RuntimeError(f"GET /readyz returned {ready.status_code}: {ready.text}")

        info_resp = client.get("/v1/model")
        if info_resp.status_code != 200:
            raise RuntimeError(f"GET /v1/model returned {info_resp.status_code}: {info_resp.text}")
        model_info = info_resp.json()

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = client.post("/v1/predict/batch", json={"texts": batch})
            if resp.status_code != 200:
                raise RuntimeError(
                    f"POST /v1/predict/batch returned {resp.status_code}: {resp.text}"
                )
            preds.extend(int(item["label"]) for item in resp.json()["predictions"])

    return model_info, preds, source_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify LegislaBERT serving parity against training validation metrics"
    )
    parser.add_argument("--checkpoint", default="models/slim_teve_veto.pt")
    parser.add_argument("--tokenizer-dir", default=".local/tokenizer")
    parser.add_argument("--data-path", default="data/processed/leis_clean.csv")
    parser.add_argument("--features-path", default="data/processed/leis_features.csv")
    parser.add_argument("--target", default="teve_veto", choices=["teve_veto", "instavel"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--expected-f1", type=float, default=0.7340)
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument(
        "--url",
        default=None,
        help="Optional running service base URL (e.g. http://127.0.0.1:8000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    val_texts, val_labels = load_validation_split(
        Path(args.data_path),
        Path(args.features_path),
        args.target,
        args.seed,
    )

    start = time.perf_counter()
    source_sha256: str | None = None
    if args.url:
        model_info, preds = _score_via_http(
            args.url, val_texts, args.batch_size, args.target
        )
    else:
        model_info, preds, source_sha256 = _score_via_testclient(
            Path(args.checkpoint),
            Path(args.tokenizer_dir),
            args.target,
            args.threshold,
            args.batch_size,
            args.threads,
            val_texts,
        )
    elapsed_s = time.perf_counter() - start

    f1_macro = float(f1_score(val_labels, preds, average="macro", zero_division=0))
    accuracy = float(accuracy_score(val_labels, preds))
    balanced_accuracy = float(balanced_accuracy_score(val_labels, preds))
    precision_veto = float(precision_score(val_labels, preds, pos_label=1, zero_division=0))
    recall_veto = float(recall_score(val_labels, preds, pos_label=1, zero_division=0))

    raw_ckpt_f1 = model_info.get("val_metrics", {}).get("f1_macro")
    checkpoint_f1 = float(raw_ckpt_f1) if raw_ckpt_f1 is not None else None
    abs_diff_expected = abs(f1_macro - args.expected_f1)
    abs_diff_checkpoint = (
        abs(f1_macro - checkpoint_f1) if checkpoint_f1 is not None else None
    )
    passed = abs_diff_expected <= args.tolerance

    report = {
        "target": args.target,
        "val_samples": len(val_texts),
        "f1_macro": f1_macro,
        "expected_f1": args.expected_f1,
        "checkpoint_f1": checkpoint_f1,
        "abs_diff_expected": abs_diff_expected,
        "abs_diff_checkpoint": abs_diff_checkpoint,
        "tolerance": args.tolerance,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_veto": precision_veto,
        "recall_veto": recall_veto,
        "checkpoint_sha256": model_info.get("checkpoint_sha256"),
        "source_sha256": source_sha256,
        "elapsed_s": round(elapsed_s, 3),
        "passed": passed,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
