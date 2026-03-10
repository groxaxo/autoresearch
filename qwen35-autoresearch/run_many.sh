#!/usr/bin/env bash
set -euo pipefail

N="${1:-10}"

for ((i=1; i<=N; i++)); do
  echo "=== autoresearch run ${i}/${N} ==="
  python run_loop.py || break
done
