#!/usr/bin/env bash
# scripts/smoke_test.sh — 100-step smoke test.
# Run on GPU pod AFTER data shards exist.
#
# Pass criterion (hardcoded in train.py --smoke_test):
#   final loss < ln(32000) ≈ 10.37
#   (loss must drop below the random-baseline within 100 steps)
#
# If this fails: check init, LR, or data loading — do NOT continue to full run.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo "  Smoke test: 100 steps"
echo "========================================"

python train.py --max_steps 100 --smoke_test

echo "========================================"
echo "  Smoke test passed. Safe to full train."
echo "========================================"
