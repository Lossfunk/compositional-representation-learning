#!/usr/bin/env bash
#
# Monitor running SCL experiments.
#
# Usage:
#   ./monitor_experiments.sh              # One-shot status
#   ./monitor_experiments.sh --watch      # Auto-refresh every 10s
#   ./monitor_experiments.sh --watch 5    # Auto-refresh every 5s

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

WATCH=false
INTERVAL=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch|-w)
            WATCH=true; shift
            if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
                INTERVAL="$1"; shift
            fi
            ;;
        *) shift ;;
    esac
done

print_status() {
    clear 2>/dev/null || true
    echo "════════════════════════════════════════════════════════════════════"
    echo "  SCL Experiment Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""

    local running=0 finished=0 failed=0 pending=0

    printf "  %-14s %-10s %-8s %-10s %s\n" "EXPERIMENT" "STATUS" "EPOCH" "DURATION" "LAST LOG LINE"
    printf "  %-14s %-10s %-8s %-10s %s\n" "──────────" "──────" "─────" "────────" "─────────────"

    for log_file in "$LOG_DIR"/scl_exp_*.log; do
        [[ -f "$log_file" ]] || continue

        local exp_name
        exp_name=$(basename "$log_file" .log)
        local pid_file="$LOG_DIR/${exp_name}.pid"

        # Determine status
        local status="UNKNOWN"
        if [[ -f "$pid_file" ]]; then
            local pid
            pid=$(<"$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                status="RUNNING"
                running=$((running + 1))
            else
                # PID file exists but process dead — check exit
                rm -f "$pid_file"
                if grep -q "Error\|Traceback\|FAILED\|Exception" "$log_file" 2>/dev/null; then
                    status="FAILED"
                    failed=$((failed + 1))
                else
                    status="FINISHED"
                    finished=$((finished + 1))
                fi
            fi
        else
            # No PID file — already completed
            if grep -q "Error\|Traceback\|FAILED\|Exception" "$log_file" 2>/dev/null; then
                status="FAILED"
                failed=$((failed + 1))
            else
                status="FINISHED"
                finished=$((finished + 1))
            fi
        fi

        # Extract epoch info from log (look for Lightning epoch progress)
        local epoch="—"
        local epoch_line
        epoch_line=$(grep -oP "Epoch \K[0-9]+" "$log_file" 2>/dev/null | tail -1 || true)
        if [[ -n "$epoch_line" ]]; then
            epoch="${epoch_line}/100"
        fi

        # Get file age (duration since creation)
        local duration="—"
        if [[ -f "$log_file" ]]; then
            local created modified now elapsed
            created=$(stat -c %Y "$log_file" 2>/dev/null || echo 0)
            modified=$(stat -c %Y "$log_file" 2>/dev/null || echo 0)
            # Use modification time - creation time isn't reliable, use first/last modify
            if [[ "$created" -gt 0 ]]; then
                now=$(date +%s)
                elapsed=$((now - created))
                local h=$((elapsed / 3600))
                local m=$(( (elapsed % 3600) / 60 ))
                duration="${h}h${m}m"
            fi
        fi

        # Last meaningful log line
        local last_line
        last_line=$(tail -1 "$log_file" 2>/dev/null | head -c 50 || echo "—")

        # Color based on status
        local color="\033[0m"
        case "$status" in
            RUNNING)  color="\033[1;33m" ;;  # Yellow
            FINISHED) color="\033[1;32m" ;;  # Green
            FAILED)   color="\033[1;31m" ;;  # Red
        esac

        printf "  %-14s ${color}%-10s\033[0m %-8s %-10s %s\n" \
            "$exp_name" "$status" "$epoch" "$duration" "$last_line"
    done

    echo ""
    echo "  Summary: $running running, $finished finished, $failed failed"

    # Show scheduler log tail
    if [[ -f "$LOG_DIR/scheduler.log" ]]; then
        echo ""
        echo "  Recent scheduler events:"
        tail -5 "$LOG_DIR/scheduler.log" 2>/dev/null | sed 's/^/    /'
    fi

    # Show GPU utilization
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "—,—,—")
        echo "  GPU: ${gpu_info}% util, ${gpu_info##*,} MB total"
    fi
}

if $WATCH; then
    while true; do
        print_status
        sleep "$INTERVAL"
    done
else
    print_status
fi
