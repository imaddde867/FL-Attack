#!/usr/bin/env bash
# ============================================================
# Defense Experiments: DP and HE Protection
# ============================================================
# Shows how defenses mitigate gradient inversion attacks
#
# IMPORTANT: Uses same config as showcase (4500×5) for fair comparison
# The baseline is copied from results/showcase/ (already run)
#
# Conditions:
#   - Baseline: No protection (from showcase, PSNR=29.38)
#   - DP (ε=8.0): Moderate DP protection
#   - DP (ε=1.0): Strong DP protection  
#   - DP (ε=0.1): Very strong DP (should break attack)
#   - HE: Homomorphic encryption with quantization noise
#   - DP+HE: Combined defenses
# ============================================================

set -e
cd "$(dirname "$0")/.."

# ============================================================
# LOCKED CONFIG: Must match showcase for fair comparison!
# ============================================================
ITERS=4500
RESTARTS=5
TV_WEIGHT="1e-5"
MATCH_METRIC="l2"

# Target client 1 (same as showcase hero)
CLIENT=1

OUT_BASE="results/defenses"

echo "============================================================"
echo "  DEFENSE EXPERIMENTS"
echo "============================================================"
echo "  Config: iters=${ITERS}, restarts=${RESTARTS}, TV=${TV_WEIGHT}"
echo "  (Matching showcase config for fair comparison)"
echo "  Target Client: ${CLIENT}"
echo "  Output: ${OUT_BASE}"
echo "============================================================"
echo ""

# ============================================================
# 0. USE EXISTING SHOWCASE AS BASELINE
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [0/5] Using showcase as baseline (already computed)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p "${OUT_BASE}/baseline"
if [ -f "results/showcase/metrics.txt" ]; then
    cp results/showcase/metrics.txt "${OUT_BASE}/baseline/"
    cp results/showcase/baseline_attack_result.png "${OUT_BASE}/baseline/" 2>/dev/null || true
    cp results/showcase/config.json "${OUT_BASE}/baseline/" 2>/dev/null || true
    echo "  Copied from results/showcase:"
    cat results/showcase/metrics.txt
    echo "  ✓ Baseline ready"
else
    echo "  ⚠ Warning: results/showcase/metrics.txt not found"
    echo "    Run scripts/run_showcase.sh first, or this will run fresh baseline"
fi
echo ""

# ============================================================
# 1. DP (ε=8.0) - Moderate Protection
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/5] DP ε=8.0 (Moderate Protection)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiment.py \
    --capture-client ${CLIENT} \
    --attack-iterations ${ITERS} \
    --attack-restarts ${RESTARTS} \
    --tv-weight ${TV_WEIGHT} \
    --match-metric ${MATCH_METRIC} \
    --lr-schedule cosine \
    --compute-lpips \
    --save-config \
    --dp-epsilon 8.0 \
    --dp-max-norm 1.0 \
    --out-dir "${OUT_BASE}/dp_eps8"
echo "  ✓ DP ε=8.0 complete"
echo ""

# ============================================================
# 2. DP (ε=1.0) - Strong Protection
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/5] DP ε=1.0 (Strong Protection)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiment.py \
    --capture-client ${CLIENT} \
    --attack-iterations ${ITERS} \
    --attack-restarts ${RESTARTS} \
    --tv-weight ${TV_WEIGHT} \
    --match-metric ${MATCH_METRIC} \
    --lr-schedule cosine \
    --compute-lpips \
    --save-config \
    --dp-epsilon 1.0 \
    --dp-max-norm 1.0 \
    --out-dir "${OUT_BASE}/dp_eps1"
echo "  ✓ DP ε=1.0 complete"
echo ""

# ============================================================
# 3. DP (ε=0.1) - Very Strong Protection
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/5] DP ε=0.1 (Very Strong - Attack Should Fail)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiment.py \
    --capture-client ${CLIENT} \
    --attack-iterations ${ITERS} \
    --attack-restarts ${RESTARTS} \
    --tv-weight ${TV_WEIGHT} \
    --match-metric ${MATCH_METRIC} \
    --lr-schedule cosine \
    --compute-lpips \
    --save-config \
    --dp-epsilon 0.1 \
    --dp-max-norm 1.0 \
    --out-dir "${OUT_BASE}/dp_eps01"
echo "  ✓ DP ε=0.1 complete"
echo ""

# ============================================================
# 4. HE (Homomorphic Encryption)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/5] Homomorphic Encryption"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiment.py \
    --capture-client ${CLIENT} \
    --attack-iterations ${ITERS} \
    --attack-restarts ${RESTARTS} \
    --tv-weight ${TV_WEIGHT} \
    --match-metric ${MATCH_METRIC} \
    --lr-schedule cosine \
    --compute-lpips \
    --save-config \
    --use-he \
    --he-bits 512 \
    --out-dir "${OUT_BASE}/he"
echo "  ✓ HE complete"
echo ""

# ============================================================
# 5. DP + HE (Combined Defenses)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/5] DP ε=1.0 + HE (Combined)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiment.py \
    --capture-client ${CLIENT} \
    --attack-iterations ${ITERS} \
    --attack-restarts ${RESTARTS} \
    --tv-weight ${TV_WEIGHT} \
    --match-metric ${MATCH_METRIC} \
    --lr-schedule cosine \
    --compute-lpips \
    --save-config \
    --dp-epsilon 1.0 \
    --dp-max-norm 1.0 \
    --use-he \
    --he-bits 512 \
    --out-dir "${OUT_BASE}/dp_he"
echo "  ✓ DP + HE complete"
echo ""

echo "============================================================"
echo "  ALL DEFENSE EXPERIMENTS COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved to: ${OUT_BASE}/"
echo ""
echo "Run analysis with:"
echo "  python scripts/analyze_defenses.py --results-dir ${OUT_BASE}"
echo ""
