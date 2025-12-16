#!/bin/bash
# ================================================================
# SHOWCASE (HERO IMAGE) - Best Visual for Poster
# ================================================================
# LOCKED config for poster centerpiece:
#   iters=4500, restarts=5, TV=1e-5, metric=l2, LPIPS on
#
# After running multi-client benchmark, pick the best-looking
# client and run this script with that client ID.
# ================================================================

set -e

# Defaults (can override via env vars)
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
DATA_SUBSET="${DATA_SUBSET:-200}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
OUT_DIR="${OUT_DIR:-results/showcase}"

# Target client (override with: TARGET_CLIENT=3 ./run_showcase.sh)
TARGET_CLIENT="${TARGET_CLIENT:-0}"

echo "============================================================"
echo "  SHOWCASE (HERO IMAGE)"
echo "============================================================"
echo "  Config: iters=4500, restarts=5, TV=1e-5, l2+LPIPS"
echo "  Target Client: ${TARGET_CLIENT}"
echo "  Output: ${OUT_DIR}"
echo "============================================================"
echo ""

mkdir -p "$OUT_DIR"

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
    --capture-client "$TARGET_CLIENT" \
    --attack-iterations 4500 \
    --attack-restarts 5 \
    --attack-lr 0.1 \
    --attack-optimizer adam \
    --tv-weight 1e-5 \
    --lr-schedule cosine \
    --match-metric l2 \
    --compute-lpips \
    --save-config

echo ""
echo "============================================================"
echo "  SHOWCASE COMPLETE!"
echo "============================================================"
echo "  Result saved to: ${OUT_DIR}/baseline_attack_result.png"
echo "  Metrics: ${OUT_DIR}/metrics.txt"
echo "============================================================"
