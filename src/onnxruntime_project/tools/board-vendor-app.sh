#!/bin/sh
# Board side: run the vendor's own multi-stream application and collect its numbers.
#
# `yolo-demo` (deb `yolo-demo` 1.0.2 from the Bianbu archive) is the application behind the
# published 12-stream table (`multi_stream_vision.md`: yolov8n 134.29 FPS total, yolov8s 77.74, ...
# with VPU decode, GPU OpenCL pre-processing and a Qt UI). Its source repository 404s
# (measured 2026-09-22: `github.com/spacemit-com/multi-stream-vision` does not resolve while
# `spacemit-com/docs-ai` does), so the packaged application is the path to a like-for-like number.
#
# It runs without a display: `QT_QPA_PLATFORM=offscreen` plus `display.mode: stats` in the config,
# and it prints per-channel `INF PERF ... fps=` lines to stdout, which is what `-l 12` needs.
#
# Process discipline, learned the hard way on 2026-09-22:
#   * `yolo-demo` may outlive a plain SIGTERM; SIGKILL leaves **all eight TCM blocks allocated to
#     threads that no longer exist** (stale entries), and every later run then fails with
#     `FileSource[0] Failed to init MPP pipeline`;
#   * the vendor's own documentation gives the remedy, `spacemit-tcm-smi -c`, and only for a board
#     with no inference running;
#   * so this script refuses to start unless no `yolo-demo` runs and all eight blocks are free,
#     and it clears stale TCM (reporting it) instead of silently measuring on a dirty board.
#
#   board-vendor-app.sh <tag> <config.yaml> <layout> <model-id> [seconds]
#
# Env: OUT (output directory), SETTLE (seconds before sampling starts), SAMPLES (TCM samples).
set -eu
OUT=${OUT:-/tmp/vendor-app}
SETTLE=${SETTLE:-15}
SAMPLES=${SAMPLES:-80}
tag=$1
config=$2
layout=$3
model=$4
seconds=${5:-120}
mkdir -p "$OUT"

free_blocks=$(spacemit-tcm-smi | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')
if pgrep -x yolo-demo > /dev/null 2>&1; then
    echo "refusing to start: a yolo-demo process is already running" >&2
    exit 2
fi
if [ "${free_blocks:-0}" != "8" ]; then
    echo "board is not clean: available_blocks=$free_blocks/8 - clearing stale TCM with the documented -c"
    spacemit-tcm-smi -c > "$OUT/${tag}-tcm-clear.txt" 2>&1 || true
    sleep 1
    free_blocks=$(spacemit-tcm-smi | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')
    [ "$free_blocks" = "8" ] || { echo "still not clean: $free_blocks/8" >&2; exit 2; }
fi

printf '%-16s state before: temps=%s clock_cpu8=%s\n' "$tag" \
    "$(cat /sys/class/thermal/thermal_zone0/temp)" \
    "$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq)"

log="$OUT/${tag}.log"
QT_QPA_PLATFORM=offscreen yolo-demo -c "$config" -l "$layout" -m "$model" > "$log" 2>&1 &
pid=$!
printf '%-16s started pid=%s layout=%s model=%s config=%s\n' "$tag" "$pid" "$layout" "$model" "$config"

i=0
busy_max=0
census_done=0
sleep "$SETTLE"
while [ "$i" -lt "$SAMPLES" ] && kill -0 "$pid" 2>/dev/null; do
    busy=$(spacemit-tcm-smi 2>/dev/null | grep -c busy || true)
    [ "$busy" -gt "$busy_max" ] && busy_max=$busy
    if [ "$census_done" = 0 ] && [ -d "/proc/$pid/task" ]; then
        ls /proc/"$pid"/task > "$OUT/${tag}-threads.txt" 2>/dev/null || true
        census_done=1
    fi
    sleep 1
    i=$((i + 1))
done

printf '%-16s tcm_busy_max=%s/8 samples=%s\n' "$tag" "$busy_max" "$i"
kill -TERM "$pid" 2>/dev/null || true
j=0
while kill -0 "$pid" 2>/dev/null && [ "$j" -lt 20 ]; do sleep 1; j=$((j + 1)); done
if kill -0 "$pid" 2>/dev/null; then
    printf '%-16s still alive after %ss of TERM - KILL (TCM will be cleared)\n' "$tag" "$j"
    kill -KILL "$pid" 2>/dev/null || true
    sleep 2
    spacemit-tcm-smi -c > "$OUT/${tag}-tcm-clear.txt" 2>&1 || true
fi
printf '%-16s terminated after %ss; blocks now %s/8; temps=%s\n' "$tag" "$j" \
    "$(spacemit-tcm-smi | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')" \
    "$(cat /sys/class/thermal/thermal_zone0/temp)"
printf '%-16s log lines=%s perf lines=%s\n' "$tag" \
    "$(wc -l < "$log")" "$(grep -c 'INF PERF' "$log" || true)"
