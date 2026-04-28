#!/usr/bin/env bash
#
# Run SCL experiments with configurable parallelism.
#
# Usage:
#   ./run_experiments.sh --experiments 55 56 57 58 --max-parallel 2
#   ./run_experiments.sh --range 55 67 --max-parallel 3
#   ./run_experiments.sh --range 55 67 --max-parallel 1 --dry-run
#
# Logs go to logs/scl_exp_<N>.log
# PID files go to logs/scl_exp_<N>.pid (removed on completion)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="vh-crl"
LOG_DIR="$SCRIPT_DIR/logs"
MAX_PARALLEL=1
DRY_RUN=false
EXPERIMENTS=()
EXP_TYPE="scl"  # "scl" or "mem_scl"

# ── Argument parsing ──────────────────────────────────────────────
usage() {
    echo "Usage: $0 [--type scl|mem_scl] [--experiments N N N...] [--range START END] [--max-parallel M] [--dry-run]"
    echo ""
    echo "Options:"
    echo "  --type TYPE             Experiment type: 'scl' or 'mem_scl' (default: scl)"
    echo "  --experiments N N N...  List of experiment numbers to run"
    echo "  --range START END       Run experiments from START to END (inclusive)"
    echo "  --max-parallel M        Max experiments to run simultaneously (default: 1)"
    echo "  --dry-run               Print what would run without executing"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiments)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXPERIMENTS+=("$1")
                shift
            done
            ;;
        --type)
            shift
            EXP_TYPE="$1"; shift
            ;;
        --range)
            shift
            START="$1"; shift
            END="$1"; shift
            for ((i=START; i<=END; i++)); do
                EXPERIMENTS+=("$i")
            done
            ;;
        --max-parallel)
            shift
            MAX_PARALLEL="$1"; shift
            ;;
        --dry-run)
            DRY_RUN=true; shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "Error: No experiments specified."
    usage
fi

# Resolve config dir and naming prefix based on experiment type
case "$EXP_TYPE" in
    scl)
        CONFIG_DIR="$SCRIPT_DIR/configs/SubspaceConceptLattice"
        EXP_PREFIX="scl_exp"
        ;;
    mem_scl)
        CONFIG_DIR="$SCRIPT_DIR/configs/MemorySubspaceConceptLattice"
        EXP_PREFIX="mem_scl_exp"
        ;;
    mem_scl_obj)
        CONFIG_DIR="$SCRIPT_DIR/configs/MemorySubspaceConceptLattice"
        EXP_PREFIX="mem_scl_obj_exp"
        ;;
    mem_scl_v2)
        CONFIG_DIR="$SCRIPT_DIR/configs/MemorySubspaceConceptLattice"
        EXP_PREFIX="mem_scl_v2_exp"
        ;;
    scl_v2)
        CONFIG_DIR="$SCRIPT_DIR/configs/SubspaceConceptLattice"
        EXP_PREFIX="scl_exp_v2"
        ;;
    *)
        echo "Error: Unknown experiment type '$EXP_TYPE'. Use 'scl', 'scl_v2', 'mem_scl', 'mem_scl_obj', or 'mem_scl_v2'."
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR"

# ── Helper functions ──────────────────────────────────────────────

running_jobs() {
    # Count how many experiment PIDs of the CURRENT type are still alive
    local count=0
    for pidfile in "$LOG_DIR"/${EXP_PREFIX}_*.pid; do
        [[ -f "$pidfile" ]] || continue
        local pid
        pid=$(<"$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        else
            # Process finished — clean up PID file
            rm -f "$pidfile"
        fi
    done
    echo "$count"
}

wait_for_slot() {
    while [[ $(running_jobs) -ge $MAX_PARALLEL ]]; do
        sleep 5
    done
}

launch_experiment() {
    local exp_num="$1"
    local config_path="$CONFIG_DIR/${EXP_PREFIX}_${exp_num}.yaml"
    local log_file="$LOG_DIR/${EXP_PREFIX}_${exp_num}.log"
    local pid_file="$LOG_DIR/${EXP_PREFIX}_${exp_num}.pid"

    if [[ ! -f "$config_path" ]]; then
        echo "  [SKIP] ${EXP_PREFIX}_${exp_num}: config not found at $config_path"
        return
    fi

    if [[ -f "$pid_file" ]]; then
        local existing_pid
        existing_pid=$(<"$pid_file")
        if kill -0 "$existing_pid" 2>/dev/null; then
            echo "  [SKIP] ${EXP_PREFIX}_${exp_num}: already running (PID $existing_pid)"
            return
        fi
        rm -f "$pid_file"
    fi

    echo "  [START] ${EXP_PREFIX}_${exp_num} → $log_file"

    # Launch in background, writing PID file
    (
        eval "$(conda shell.bash hook 2>/dev/null)" && \
        conda activate "$CONDA_ENV" && \
        PYTHONUNBUFFERED=1 python train.py --config_filepath "$config_path" \
            > "$log_file" 2>&1
        EXIT_CODE=$?
        rm -f "$pid_file"
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "[$(date '+%H:%M:%S')] ${EXP_PREFIX}_${exp_num} FINISHED (exit 0)" >> "$LOG_DIR/scheduler.log"
        else
            echo "[$(date '+%H:%M:%S')] ${EXP_PREFIX}_${exp_num} FAILED (exit $EXIT_CODE)" >> "$LOG_DIR/scheduler.log"
        fi
    ) &

    local bg_pid=$!
    echo "$bg_pid" > "$pid_file"
}

# ── Main ──────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════"
echo "  Experiment Runner (type: $EXP_TYPE)"
echo "  Experiments: ${EXPERIMENTS[*]}"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Config dir: $CONFIG_DIR"
echo "  Log dir: $LOG_DIR"
echo "════════════════════════════════════════════════════════"

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Would launch the following:"
    for exp_num in "${EXPERIMENTS[@]}"; do
        config_path="$CONFIG_DIR/${EXP_PREFIX}_${exp_num}.yaml"
        if [[ -f "$config_path" ]]; then
            echo "  conda run -n $CONDA_ENV python train.py --config_filepath $config_path"
        else
            echo "  [MISSING] $config_path"
        fi
    done
    exit 0
fi

echo "" >> "$LOG_DIR/scheduler.log"
echo "[$(date '+%H:%M:%S')] === New run ($EXP_TYPE): experiments ${EXPERIMENTS[*]}, max_parallel=$MAX_PARALLEL ===" >> "$LOG_DIR/scheduler.log"

for exp_num in "${EXPERIMENTS[@]}"; do
    wait_for_slot
    launch_experiment "$exp_num"
done

echo ""
echo "All experiments launched. Waiting for completion..."
echo "  Monitor with: ./monitor_experiments.sh"
echo "  Tail a log:   tail -f logs/${EXP_PREFIX}_<N>.log"

# Wait for all background jobs to finish
wait

echo ""
echo "All experiments complete. Check $LOG_DIR/scheduler.log for summary."
