#!/bin/bash
# ================================================================
# BENCHMARK Configuration - LOCKED
# ================================================================
# Faster but still strong attack for multi-face benchmarking
# Config: iters=3000, restarts=3, TV=1e-5, metric=l2, LPIPS on
# ================================================================

set -e

# Defaults (can override via env vars)
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
DATA_SUBSET="${DATA_SUBSET:-200}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
CAPTURE_CLIENT="${CAPTURE_CLIENT:-0}"
OUT_DIR="${OUT_DIR:-results/benchmark}"

echo "=============================================="
echo "  BENCHMARK CONFIG (Fast, Multi-Face Ready)"
echo "=============================================="
echo "  Iterations:  3000"
echo "  Restarts:    3"
echo "  TV Weight:   1e-5"
echo "  Metric:      l2"
echo "  LPIPS:       ON"
echo "  LR Schedule: cosine"
echo "=============================================="
echo ""

python run_experiment.py \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --data-subset "$DATA_SUBSET" \
    --num-clients "$NUM_CLIENTS" \
    --local-epochs 1 \
    --client-lr 0.01 \
    --batch-size 1 \
    --client-momentum 0.0 \
    --attack-source gradients \
    --num-rounds 1 \
    --capture-round 0 \
    --capture-client "$CAPTURE_CLIENT" \
    --attack-iterations 3000 \
    --attack-restarts 3 \
    --attack-lr 0.1 \
    --attack-optimizer adam \
    --tv-weight 1e-5 \
    --lr-schedule cosine \
    --match-metric l2 \
    --compute-lpips \
    --save-config

echo ""
echo "Benchmark attack complete. Results saved to: $OUT_DIR"
