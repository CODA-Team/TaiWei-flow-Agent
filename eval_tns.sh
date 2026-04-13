#!/bin/bash
# eval_tns.sh - Re-evaluate TNS for all runs under a fixed reference clock CP_0.
#
# Flow:
#   Iteration 1: extract default run's ECP -> CP_0 = ECP * 0.9, save to file,
#                 then evaluate TNS for all (parallel_runs+1) tasks.
#   Iteration 2+: read saved CP_0, evaluate TNS for all parallel_runs tasks.
#
# Usage: ./eval_tns.sh <platform> <design> <parallel_runs> <iteration>

set -euo pipefail

if [ $# -ne 4 ]; then
    echo "Usage: $0 <platform> <design> <parallel_runs> <iteration>"
    exit 1
fi

platform=$1
design=$2
parallel_runs=$3
iteration=$4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLATFORM_DIR="${SCRIPT_DIR}/platforms/${platform}"
CP0_FILE="${SCRIPT_DIR}/designs/${platform}/${design}/cp0.txt"

# Hardcoded CP_0 discount factor
CP0_FACTOR=0.9

# --- Determine liberty file glob patterns per platform ---
case "$platform" in
    asap7)
        LIB_FILES="${PLATFORM_DIR}/lib/NLDM/*_RVT_TT_nldm_*.lib.gz"
        ;;
    sky130hd)
        LIB_FILES="${PLATFORM_DIR}/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
        ;;
    nangate45)
        LIB_FILES="${PLATFORM_DIR}/lib/NangateOpenCellLibrary_typical.lib"
        ;;
    *)
        echo "Error: unsupported platform '$platform' for TNS evaluation"
        exit 1
        ;;
esac
export LIB_FILES

# --- Determine base SDC file (used as template for CP_0 SDC) ---
if [[ "$platform" == "asap7" && "$design" == "jpeg" ]]; then
    BASE_SDC="${SCRIPT_DIR}/designs/${platform}/${design}/jpeg_encoder15_7nm.sdc"
else
    BASE_SDC="${SCRIPT_DIR}/designs/${platform}/${design}/constraint.sdc"
fi

# --- Determine effective number of tasks ---
if [ "$iteration" -eq 1 ]; then
    effective_runs=$((parallel_runs + 1))
else
    effective_runs=$parallel_runs
fi

# --- Helper: extract ECP from a run's log ---
# Looks for "clock period_min = <value>" in the Report metrics stage 6 section,
# or computes ECP = clock_period - worst_slack as fallback.
extract_ecp_from_log() {
    local log_file=$1
    # Try direct ECP (clock_period_min) from stage 6
    local ecp
    ecp=$(grep -oP 'cl(ock)?\s*period_min\s*=\s*\K[-\d.]+' "$log_file" | tail -1)
    if [ -n "$ecp" ]; then
        echo "$ecp"
        return
    fi
    # Fallback: clock_period - worst_slack
    local period wns
    period=$(grep -oP 'clock period to\s*\K[\d.]+' "$log_file" | tail -1)
    wns=$(grep -oP 'wns max\s+\K[-\d.]+' "$log_file" | tail -1)
    if [ -n "$period" ] && [ -n "$wns" ]; then
        echo "$period $wns" | awk '{printf "%.4f", $1 - $2}'
        return
    fi
    echo ""
}

# ============================================================
# Step 1: Compute CP_0 on iteration 1
# ============================================================
if [ "$iteration" -eq 1 ]; then
    default_task_id=$((parallel_runs + 1))
    default_log="${SCRIPT_DIR}/logs/${platform}_${design}_run${default_task_id}.log"

    if [ ! -f "$default_log" ]; then
        echo "[eval_tns.sh] ERROR: default run log not found: $default_log"
        exit 1
    fi

    default_ecp=$(extract_ecp_from_log "$default_log")
    if [ -z "$default_ecp" ]; then
        echo "[eval_tns.sh] ERROR: could not extract ECP from default run log: $default_log"
        exit 1
    fi

    CP_0=$(echo "$default_ecp $CP0_FACTOR" | awk '{printf "%.4f", $1 * $2}')
    echo "$CP_0" > "$CP0_FILE"
    echo "[eval_tns.sh] Default ECP = $default_ecp, CP_0 = ECP * $CP0_FACTOR = $CP_0 (saved to $CP0_FILE)"
else
    if [ ! -f "$CP0_FILE" ]; then
        echo "[eval_tns.sh] ERROR: CP_0 file not found: $CP0_FILE (was iteration 1 run?)"
        exit 1
    fi
    CP_0=$(cat "$CP0_FILE")
    echo "[eval_tns.sh] Loaded CP_0 = $CP_0 from $CP0_FILE"
fi

# ============================================================
# Step 2: Create temporary SDC with CP_0
# ============================================================
CP0_SDC="${SCRIPT_DIR}/designs/${platform}/${design}/constraint_cp0.sdc"
cp "$BASE_SDC" "$CP0_SDC"
# Remove existing clk_period line and prepend CP_0
sed -i '/set clk_period/d' "$CP0_SDC"
sed -i "1i set clk_period $CP_0" "$CP0_SDC"
tr -d '\r' < "$CP0_SDC" > "${CP0_SDC}.tmp" && mv "${CP0_SDC}.tmp" "$CP0_SDC"
export SDC_PATH="$CP0_SDC"

echo "[eval_tns.sh] CP_0 SDC created at $CP0_SDC (clk_period = $CP_0)"

# ============================================================
# Step 3: Evaluate TNS for each run
# ============================================================
echo "[eval_tns.sh] Evaluating TNS for $effective_runs runs under CP_0 = $CP_0 ..."

for ((i=1; i<=effective_runs; i++)); do
    odb_path="${SCRIPT_DIR}/results/${platform}/${design}/base_${i}/6_final.odb"
    spef_path="${SCRIPT_DIR}/results/${platform}/${design}/base_${i}/6_final.spef"
    run_log="${SCRIPT_DIR}/logs/${platform}_${design}_run${i}.log"

    if [ ! -f "$odb_path" ]; then
        echo "[eval_tns.sh] SKIP run $i: ODB not found at $odb_path"
        echo "[TNS_EVAL] tns_eval = N/A (odb_missing)" >> "$run_log"
        continue
    fi

    export ODB_PATH="$odb_path"
    export SPEF_PATH="$spef_path"

    echo "[eval_tns.sh] Evaluating run $i ..."
    eval_output=$($OPENROAD_EXE -exit "${SCRIPT_DIR}/eval_tns.tcl" 2>&1) || true

    # Extract tns_eval value from OpenROAD output
    tns_val=$(echo "$eval_output" | grep -oP 'tns_eval = \K[-\d.eE+]+' | tail -1)

    if [ -n "$tns_val" ]; then
        echo "[TNS_EVAL] tns_eval = $tns_val" >> "$run_log"
        echo "  run $i: tns_eval = $tns_val"
    else
        echo "[TNS_EVAL] tns_eval = N/A (extraction_failed)" >> "$run_log"
        echo "  run $i: tns_eval extraction failed"
        # Dump last 20 lines for debugging
        echo "$eval_output" | tail -20
    fi
done

# Clean up temp SDC
rm -f "$CP0_SDC"

echo "[eval_tns.sh] TNS evaluation complete for iteration $iteration."
