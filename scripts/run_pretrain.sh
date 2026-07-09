#!/usr/bin/env bash
# scripts/run_pretrain.sh — Full 9,537-step pretraining run.
#
# Expected runtime (measured):
#   H100, torch.compile + TF32 enabled: ~4.25–4.28 hours
#   RTX Pro 6000, no compile (baseline): ~10.5 hours
#
# Run on GPU pod after:
#   1. Data shards prepared (prepare_fineweb.py + prepare_stories.py)
#   2. Validation suite passed (validate_components.py + validate_forward.py)
#   3. Smoke test passed (smoke_test.sh)
#   4. HF_TOKEN set in .env

set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo "  Starting full pretraining run"
echo "  Steps: 9,537  |  Expected: ~4.25h (H100)"
echo "========================================"

# Optional: resume from a checkpoint
# Uncomment and set path if resuming:
# RESUME_FLAG="--resume checkpoints/step_005000.pt"

python train.py ${RESUME_FLAG:-}

echo "========================================"
echo "  Pretraining complete."
echo "========================================"
