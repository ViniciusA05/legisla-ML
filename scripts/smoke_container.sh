#!/usr/bin/env bash
# Smoke test: the image starts WITHOUT a checkpoint and must report
# liveness (200) but not readiness (503). Usage: scripts/smoke_container.sh [image]
set -euo pipefail

IMAGE="${1:-legislaml-serving:ci}"
NAME="legislaml-smoke-$$"
PORT="${SMOKE_PORT:-18000}"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" -p "${PORT}:8000" "$IMAGE" >/dev/null

code() { curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}$1" || true; }

for _ in $(seq 1 60); do
  [ "$(code /healthz)" = "200" ] && break
  sleep 1
done

health="$(code /healthz)"
ready="$(code /readyz)"
echo "healthz=${health} readyz=${ready}"

if [ "$health" != "200" ] || [ "$ready" != "503" ]; then
  echo "Smoke test failed (expected healthz=200 readyz=503)" >&2
  docker logs "$NAME" >&2 || true
  exit 1
fi

user="$(docker exec "$NAME" id -u)"
if [ "$user" = "0" ]; then
  echo "Container runs as root" >&2
  exit 1
fi
echo "Smoke test passed (non-root uid=${user})"
