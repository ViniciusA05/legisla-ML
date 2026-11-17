"""Latency benchmark for a running LegislaBERT inference service (FEAT-003 NFR02).

Usage:
    python scripts/bench_serving.py --url http://127.0.0.1:8000 --requests 50
"""

import argparse
import json
import statistics
import time
import urllib.request


def _post(url: str, payload: dict) -> float:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()
    return time.perf_counter() - start


def _percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the inference service")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--words", type=int, default=900, help="Words per synthetic statute")
    args = parser.parse_args()

    text = " ".join(f"artigo{i % 97}" for i in range(args.words))  # exceeds 510 tokens
    _post(f"{args.url}/v1/predict", {"text": text})  # warm-up

    singles = sorted(_post(f"{args.url}/v1/predict", {"text": text}) for _ in range(args.requests))
    batch = _post(f"{args.url}/v1/predict/batch", {"texts": [text] * args.batch_size})

    print(
        json.dumps(
            {
                "requests": args.requests,
                "single_p50_s": round(statistics.median(singles), 3),
                "single_p95_s": round(_percentile(singles, 0.95), 3),
                "single_p99_s": round(_percentile(singles, 0.99), 3),
                f"batch_{args.batch_size}_total_s": round(batch, 3),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
