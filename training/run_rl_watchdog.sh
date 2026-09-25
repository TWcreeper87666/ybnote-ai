#!/bin/bash
# Self-healing launcher for train_rl.py. Some runs on this machine
# inexplicably stall right after start (near-zero CPU, zero output for
# 10+ minutes) with no reproducible cause found (not disk I/O, not thread
# contention -- torch.set_num_threads(1) already applied in train_rl.py).
# This retries the whole process from scratch if no log growth appears
# within GRACE_SECONDS, up to MAX_ATTEMPTS times, then just tails the
# eventually-healthy run to completion.
set -u
cd "$(dirname "$0")"

LOG=train_rl.log
GRACE_SECONDS=180
MAX_ATTEMPTS=8
ARGS="--charts-dir ../output --holdout 5 --iterations 800 --rollout-steps 4096 --stage-b-after 500 --eval-every 20 --save rl_policy.pt"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[watchdog] attempt $attempt/$MAX_ATTEMPTS: launching train_rl.py"
    : > "$LOG"
    python -u train_rl.py $ARGS > "$LOG" 2>&1 &
    pid=$!

    waited=0
    healthy=0
    while [ "$waited" -lt "$GRACE_SECONDS" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[watchdog] process exited early (attempt $attempt) -- see $LOG"
            break
        fi
        if [ -s "$LOG" ] && grep -q "iter" "$LOG"; then
            echo "[watchdog] attempt $attempt is healthy after ${waited}s -- letting it run"
            healthy=1
            break
        fi
        sleep 5
        waited=$((waited + 5))
    done

    if [ "$healthy" -eq 1 ]; then
        wait "$pid"
        echo "[watchdog] training process exited with code $?"
        exit 0
    fi

    echo "[watchdog] attempt $attempt stalled (no output after ${GRACE_SECONDS}s) -- killing and retrying"
    kill -9 "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    sleep 3
done

echo "[watchdog] gave up after $MAX_ATTEMPTS attempts"
exit 1
