#!/bin/bash
# ================================================================
# ABLATION STUDY - Lean & Fair Configuration
# ================================================================
# Fixed targets: clients 0, 3, 7
# Budget: iters=2000, restarts=3 (fast, still informative)
# 
# Ablation dimensions:
#   1. TV weight:      {1e-6, 1e-5, 1e-4}
#   2. Initialization: plain (default) vs --fft-init
#   3. Match metric:   l2 vs cosine vs both vs sim
#
# Note: Layer weights ablation skipped (parsing issues)
# ================================================================

set -e

# Config
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
DATA_SUBSET="${DATA_SUBSET:-200}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
BASE_OUT_DIR="${BASE_OUT_DIR:-results/ablation}"

# Fixed attack budget (lean: fast but still informative)
ITERS=2000
RESTARTS=3

# Fixed target clients for fair comparison
TARGET_CLIENTS=(0 3 7)

echo "============================================================"
echo "  ABLATION STUDY (Lean & Fair)"
echo "============================================================"
echo "  Clients:  ${TARGET_CLIENTS[*]}"
echo "  Budget:   iters=${ITERS}, restarts=${RESTARTS}"
echo "  Output:   ${BASE_OUT_DIR}"
echo "============================================================"
echo ""

# Create base directory
mkdir -p "$BASE_OUT_DIR"

# Function to run attack on all target clients
run_ablation() {
    local name="$1"
    local extra_args="$2"
    local out="${BASE_OUT_DIR}/${name}"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ABLATION: ${name}"
    echo "  Args: ${extra_args}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    mkdir -p "$out"
    
    for c in "${TARGET_CLIENTS[@]}"; do
        echo "  → Client ${c}..."
        python run_experiment.py \
            --out-dir "${out}/c${c}" \
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
            --attack-iterations "$ITERS" \
            --attack-restarts "$RESTARTS" \
            --attack-lr 0.1 \
            --attack-optimizer adam \
            --lr-schedule cosine \
            --compute-lpips \
            --save-config \
            $extra_args
    done
    echo "  ✓ ${name} complete"
}

# ================================================================
# BASELINE (reference config from showcase)
# ================================================================
echo ""
echo "============================================================"
echo "  [1/7] BASELINE: TV=1e-5, L2, plain init"
echo "============================================================"
run_ablation "baseline" "--tv-weight 1e-5 --match-metric l2"

# ================================================================
# ABLATION 1: TV Weight
# ================================================================
echo ""
echo "============================================================"
echo "  [2/7] TV ABLATION: 1e-6 (weaker regularization)"
echo "============================================================"
run_ablation "tv_1e-6" "--tv-weight 1e-6 --match-metric l2"

echo ""
echo "============================================================"
echo "  [3/7] TV ABLATION: 1e-4 (stronger regularization)"
echo "============================================================"
run_ablation "tv_1e-4" "--tv-weight 1e-4 --match-metric l2"

# ================================================================
# ABLATION 2: Initialization
# ================================================================
echo ""
echo "============================================================"
echo "  [4/7] INIT ABLATION: FFT initialization"
echo "============================================================"
run_ablation "init_fft" "--tv-weight 1e-5 --match-metric l2 --fft-init"

# ================================================================
# ABLATION 3: Match Metric
# ================================================================
echo ""
echo "============================================================"
echo "  [5/7] METRIC ABLATION: Cosine similarity"
echo "============================================================"
run_ablation "metric_cosine" "--tv-weight 1e-5 --match-metric cosine"

echo ""
echo "============================================================"
echo "  [6/7] METRIC ABLATION: Both (L2 + Cosine)"
echo "============================================================"
run_ablation "metric_both" "--tv-weight 1e-5 --match-metric both"

echo ""
echo "============================================================"
echo "  [7/7] METRIC ABLATION: Sim (hybrid similarity)"
echo "============================================================"
run_ablation "metric_sim" "--tv-weight 1e-5 --match-metric sim"

# ================================================================
# Summary
# ================================================================
echo ""
echo "============================================================"
echo "  ALL ABLATIONS COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved to: ${BASE_OUT_DIR}/"
echo ""
echo "Run analysis with:"
echo "  python scripts/analyze_ablation.py --results-dir ${BASE_OUT_DIR}"
echo ""
