#!/bin/bash
# ================================================================
# MULTI-CLIENT BENCHMARK - Attack Multiple Faces
# ================================================================
# Runs benchmark config across all 10 clients (0-9) to get
# statistical measures (mean ± std) for PSNR/SSIM/LPIPS
# ================================================================

set -e

# Defaults (can override via env vars)
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
DATA_SUBSET="${DATA_SUBSET:-200}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
BASE_OUT_DIR="${BASE_OUT_DIR:-results/multi_client}"

# Number of clients to attack (0 to NUM_ATTACK_CLIENTS-1)
NUM_ATTACK_CLIENTS="${NUM_ATTACK_CLIENTS:-10}"

echo "============================================================"
echo "  MULTI-CLIENT BENCHMARK (Attacking ${NUM_ATTACK_CLIENTS} clients)"
echo "============================================================"
echo "  Config: iters=3000, restarts=3, TV=1e-5, l2+LPIPS"
echo "  Output: ${BASE_OUT_DIR}"
echo "============================================================"
echo ""

# Create base output directory
mkdir -p "$BASE_OUT_DIR"

# Run attack for each client
for c in $(seq 0 $((NUM_ATTACK_CLIENTS - 1))); do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Attacking Client $c / $((NUM_ATTACK_CLIENTS - 1))"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python run_experiment.py \
        --out-dir "${BASE_OUT_DIR}/bmk_c${c}" \
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
        --capture-client "$c" \
        --attack-iterations 3000 \
        --attack-restarts 3 \
        --attack-lr 0.1 \
        --attack-optimizer adam \
        --tv-weight 1e-5 \
        --lr-schedule cosine \
        --match-metric l2 \
        --compute-lpips \
        --save-config
    
    echo "  ✓ Client $c complete"
done

echo ""
echo "============================================================"
echo "  All ${NUM_ATTACK_CLIENTS} clients attacked successfully!"
echo "============================================================"
echo ""
echo "Run the analysis script to compute aggregate statistics:"
echo "  python scripts/analyze_multi_client.py --results-dir ${BASE_OUT_DIR}"
echo ""
