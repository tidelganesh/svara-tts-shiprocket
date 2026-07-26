#!/bin/bash
set -e

echo "Waiting for vLLM on port ${VLLM_PORT:-8000}..."
until curl -sf "http://localhost:${VLLM_PORT:-8000}/health" > /dev/null; do
  sleep 3
done
echo "vLLM ready — starting FastAPI"

exec python3 -m uvicorn server:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8080}" --log-level info --no-access-log
