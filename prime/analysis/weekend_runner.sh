#!/usr/bin/env bash
#
# Prime Weekend Runner
# ========================
# Runs on AWS (or any machine with the repos cloned).
# Orchestrates: window optimization -> full pipeline -> all analyses
#
# Prerequisites:
#   - Python 3.11+ with venv
#   - Both repos cloned:
#       ~/engines/     (Manifold compute)
#       ~/framework/   (Prime interpreter)
#   - FD001 data in ~/data/FD001/:
#       observations.parquet
#       typology.parquet
#       typology_raw.parquet
#       manifest.yaml
#
# Usage:
#   chmod +x weekend_runner.sh
#   ./weekend_runner.sh              # Full run (window opt + pipeline + analysis)
#   ./weekend_runner.sh --skip-opt   # Skip window optimization (use manifest as-is)
#   ./weekend_runner.sh --quick      # Quick test (fewer grid points)

set -euo pipefail

# ============================================================
# CONFIG -- edit these paths
# ============================================================
ENGINES_DIR="${ENGINES_DIR:-$HOME/engines}"
FRAMEWORK_DIR="${FRAMEWORK_DIR:-$HOME/framework}"
DATA_DIR="${DATA_DIR:-$HOME/data/FD001}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/output}"
ANALYSIS_DIR="${OUTPUT_DIR}/analysis"

SKIP_OPT=false
QUICK=false

for arg in "$@"; do
    case $arg in
        --skip-opt) SKIP_OPT=true ;;
        --quick) QUICK=true ;;
    esac
done

# ============================================================
# SETUP
# ============================================================
echo ""
echo "########################################################################"
echo "  PRIME WEEKEND RUNNER"
echo "  $(date)"
echo "########################################################################"
echo ""
echo "  Engines:    $ENGINES_DIR"
echo "  Framework:  $FRAMEWORK_DIR"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUTPUT_DIR"
echo ""

# Verify files exist
for f in observations.parquet typology.parquet manifest.yaml; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "ERROR: Missing $DATA_DIR/$f"
        exit 1
    fi
done

# Create output dirs
mkdir -p "$OUTPUT_DIR" "$ANALYSIS_DIR"

# Set PYTHONPATH so both repos are importable
export PYTHONPATH="$ENGINES_DIR:$FRAMEWORK_DIR:${PYTHONPATH:-}"

echo "  PYTHONPATH: $PYTHONPATH"
echo ""

# Activate venv if it exists
if [ -f "$FRAMEWORK_DIR/venv/bin/activate" ]; then
    source "$FRAMEWORK_DIR/venv/bin/activate"
    echo "  Activated venv: $(which python)"
fi

# ============================================================
# PHASE 1: WINDOW OPTIMIZATION
# ============================================================
if [ "$SKIP_OPT" = false ]; then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "  PHASE 1: WINDOW OPTIMIZATION"
    echo "------------------------------------------------------------------------"
    echo ""

    # Option A: Raw eigendecomp (fast baseline -- ~10 seconds)
    echo "  [1a] Raw eigendecomp baseline..."
    python -m prime.analysis.window_optimization \
        --observations "$DATA_DIR/observations.parquet" \
        --typology "$DATA_DIR/typology.parquet" \
        --output "$ANALYSIS_DIR/window_optimization_raw.parquet"

    echo ""

    # Option B: Full Manifold pipeline (the real deal -- ~15-45 min)
    echo "  [1b] Full Manifold pipeline sweep..."
    if [ "$QUICK" = true ]; then
        WINDOW_ARG="--windows 10,20,40"
        MODE_ARG="--modes non_overlapping,half_overlap"
    else
        WINDOW_ARG=""  # auto
        MODE_ARG=""    # all three
    fi

    python -m prime.analysis.window_optimization_manifold \
        --data-dir "$DATA_DIR" \
        --output "$ANALYSIS_DIR/window_optimization_manifold.parquet" \
        $WINDOW_ARG $MODE_ARG

    # Extract optimal window/stride from results
    OPTIMAL=$(python3 -c "
import json
with open('$ANALYSIS_DIR/window_optimization_manifold_summary.json') as f:
    s = json.load(f)
print(f\"{s['best_window']} {s['best_stride']}\")
")
    OPT_WINDOW=$(echo $OPTIMAL | cut -d' ' -f1)
    OPT_STRIDE=$(echo $OPTIMAL | cut -d' ' -f2)

    echo ""
    echo "  Optimal: window=$OPT_WINDOW stride=$OPT_STRIDE"
    echo ""

    # Patch manifest with optimal values
    python3 -c "
import yaml
with open('$DATA_DIR/manifest.yaml') as f:
    m = yaml.safe_load(f)
m['system']['window'] = $OPT_WINDOW
m['system']['stride'] = $OPT_STRIDE
with open('$DATA_DIR/manifest.yaml', 'w') as f:
    yaml.dump(m, f, default_flow_style=False, sort_keys=False)
print(f'  Patched manifest: window={$OPT_WINDOW}, stride={$OPT_STRIDE}')
"
else
    echo ""
    echo "  Skipping window optimization (--skip-opt)"
    echo ""
fi

# ============================================================
# PHASE 2: FULL MANIFOLD PIPELINE
# ============================================================
echo ""
echo "------------------------------------------------------------------------"
echo "  PHASE 2: FULL MANIFOLD PIPELINE (stages 00-14)"
echo "------------------------------------------------------------------------"
echo ""

python -m engines.entry_points.run_pipeline \
    "$DATA_DIR/manifest.yaml" \
    --verbose

echo ""
echo "  Pipeline complete. Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.parquet 2>/dev/null || echo "  (no parquet files found)"

# ============================================================
# PHASE 3: PRIME ANALYSIS
# ============================================================
echo ""
echo "------------------------------------------------------------------------"
echo "  PHASE 3: PRIME ANALYSIS (20/20 + canary + thermodynamics)"
echo "------------------------------------------------------------------------"
echo ""

# Copy observations to output dir if analysis expects it there
if [ ! -f "$OUTPUT_DIR/observations.parquet" ]; then
    cp "$DATA_DIR/observations.parquet" "$OUTPUT_DIR/observations.parquet"
fi

python -m prime.analysis.study \
    --data "$OUTPUT_DIR" \
    --output "$ANALYSIS_DIR"

# ============================================================
# PHASE 4: SUMMARY
# ============================================================
echo ""
echo "########################################################################"
echo "  WEEKEND RUN COMPLETE"
echo "  $(date)"
echo "########################################################################"
echo ""
echo "  Files in $ANALYSIS_DIR:"
ls -lh "$ANALYSIS_DIR"/ 2>/dev/null
echo ""
echo "  Key results:"

# Print 20/20 summary if available
if [ -f "$ANALYSIS_DIR/study_summary.json" ]; then
    python3 -c "
import json
with open('$ANALYSIS_DIR/study_summary.json') as f:
    s = json.load(f)
tt = s.get('twenty_twenty', {})
print(f\"  20/20 r = {tt.get('r_early_lifecycle', 'N/A')}\")
print(f\"  20/20 R-squared = {tt.get('r_squared', 'N/A')}\")
print(f\"  Bootstrap CI = [{tt.get('bootstrap_ci_low', 'N/A')}, {tt.get('bootstrap_ci_high', 'N/A')}]\")
canary = s.get('canary', {})
print(f\"  Canary signal = {canary.get('top_rul', 'N/A')}\")
"
fi

echo ""
echo "  Next: download $ANALYSIS_DIR and generate figures"
echo ""
