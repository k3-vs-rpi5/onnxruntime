#!/bin/sh
# Board side: does the multi-session throughput rule survive a bigger model?
#
# Round 14.1 measured, on `yolov8n`, that one process running N identical sessions of 8/N AI threads
# each beats one fat 8-thread session by 42 % in aggregate (8x1 = 151.50 inf/s against 1x8 = 106.27)
# and that this matches the vendor's own application, which documents "eight parallel ONNX Runtime
# sessions by default" and distributes work over them round-robin. The rule was established on one
# small model; the vendor's application runs it for `yolov8s`/`yolov8m` too, and this round tests
# whether it holds there.
#
# Instrument: `ort-two-session-pipeline`'s `same` mode (one process, N sessions, 8/N threads each),
# the same harness section 14.1 used, so the numbers are the same kind. Each shape is run `ROUNDS`
# times; the AI-cluster claim for the first round of each shape carries an in-flight
# `spacemit-tcm-smi` sample and the process's thread set.
#
#   board-multisession-scaling.sh
#
# Env: ORT (package root), PIPE (harness binary), MODELS (space-separated), OUT, ROUNDS, ITERS.
#
# The harness CLI is `pipeline <modelA> <modelB> <serial|overlap|same> ...`; `same` mode ignores
# `<modelB>`, so the model path is passed twice. (The header of `ort-two-session-pipeline.cpp` shows a
# shorter form that the binary does not accept - measured 2026-09-22: it fails with
# `Load model same failed` - and the comment is corrected there.)
set -eu
ORT=${ORT:-/tmp/ort206}
PIPE=${PIPE:-/root/pipeline}
MODELS=${MODELS:-/root/onnx-zoo/yolov8n.q.onnx /root/onnx-zoo/yolov8s.q.onnx}
OUT=${OUT:-/tmp/multisession}
ROUNDS=${ROUNDS:-3}
ITERS=${ITERS:-10}
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$ORT/lib"

# `spacemit-tcm-smi`'s PID column names whatever id the AI thread registered with the driver - the
# process itself on some paths, a thread id on others (measured 2026-09-22: `perf_test` reported the
# process pid, this harness reported one of its tids, 39 apart). Match the workload by *any* of its
# thread ids, and recompute the set each sample because the pool grows during start-up.
tcm_lines() {
    pattern=$(ls /proc/"$1"/task 2>/dev/null | tr '\n' '|' | sed 's/|$//')
    [ -n "$pattern" ] || return 0
    spacemit-tcm-smi 2>/dev/null | grep -E " ($pattern) "
}

printf '== board state\n'
for c in 0 8; do
    printf '   cpu%-2s governor=%s cur_khz=%s max_khz=%s\n' "$c" \
        "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)" \
        "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq)" \
        "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq)"
done
printf '   temps_c: %s\n' "$(cat /sys/class/thermal/thermal_zone0/temp \
    /sys/class/thermal/thermal_zone3/temp 2>/dev/null | tr '\n' ' ')"
printf '   models:\n'
for m in $MODELS; do
    printf '     %s %s\n' "$(sha256sum "$m" | cut -d' ' -f1)" "$m"
done

row() {
    tag=$1
    model=$2
    sessions=$3
    threads=$4
    rounds=$5
    base=$(basename "$model" .onnx)
    log="$OUT/${base}-${sessions}x${threads}-r1.log"
    if ! "$PIPE" "$model" "$model" same "$sessions" "$ITERS" "$threads" > "$log" 2>&1; then
        printf '%-26s r1 FAILED: %s\n' "$tag" "$(tail -1 "$log")"
        return 0
    fi
    slowest=$(sed -n 's/^slowest_ms=\([0-9.]*\).*$/\1/p' "$log" | tail -1)
    printf '%-26s r1 %s\n' "$tag" "$(grep -h 'aggregate_per_s' "$log" | tail -1)"

    # A second, longer run of the same shape, sampled while it is in flight: the timing rounds are
    # ~0.1-2 s each and can end between two TCM samples.
    engage=$(awk -v s="$slowest" 'BEGIN { v = int(4000 / s) + 1; if (v < 20) v = 20; print v }')
    # No `setsid` here: it may fork, and then `$!` is the wrapper's pid rather than the workload's,
    # which is exactly how the first attempt at this sampling reported zero busy blocks while the
    # AI cluster was in fact fully engaged (measured 2026-09-22 on this board).
    "$PIPE" "$model" "$model" same "$sessions" "$engage" "$threads" \
        > "$OUT/${base}-${sessions}x${threads}-engage.log" 2>&1 &
    pid=$!
    i=0
    max_threads=0
    : > "$OUT/${base}-${sessions}x${threads}-tcm.txt"
    : > "$OUT/${base}-${sessions}x${threads}-threads.txt"
    while [ "$i" -lt 60 ]; do
        tcm_lines "$pid" >> "$OUT/${base}-${sessions}x${threads}-tcm.txt" || true
        kill -0 "$pid" 2>/dev/null || break
        if [ -d "/proc/$pid/task" ]; then
            n=$(ls /proc/"$pid"/task | wc -l)
            if [ "$n" -gt "$max_threads" ]; then
                max_threads=$n
                ps -L -o tid,comm,psr -p "$pid" \
                    > "$OUT/${base}-${sessions}x${threads}-threads.txt" 2>&1 || true
            fi
        fi
        sleep 0.2
        i=$((i + 1))
    done
    wait "$pid" || true
    busy=$(grep -c busy "$OUT/${base}-${sessions}x${threads}-tcm.txt" || true)
    declare_blocks=$(awk '{print $1}' "$OUT/${base}-${sessions}x${threads}-tcm.txt" \
        | sort -u | tr '\n' ' ')
    printf '%-26s    engage iters=%-5s tcm_busy_lines=%-4s blocks=[%s] max_threads=%s\n' \
        "" "$engage" "$busy" "$declare_blocks" "$max_threads"

    r=2
    while [ "$r" -le "$rounds" ]; do
        log="$OUT/${base}-${sessions}x${threads}-r${r}.log"
        if "$PIPE" "$model" "$model" same "$sessions" "$ITERS" "$threads" > "$log" 2>&1; then
            printf '%-26s r%s %s\n' "$tag" "$r" "$(grep -h 'aggregate_per_s' "$log" | tail -1)"
        else
            # Kept, not retried silently: the harness failed at session creation on 2026-09-22 with
            # "SpacemiT EP init: Provider options key/value cannot be empty" after two clean rounds of
            # the same shape, which is a run result too.
            printf '%-26s r%s FAILED: %s\n' "$tag" "$r" "$(tail -1 "$log")"
        fi
        r=$((r + 1))
    done
}

for m in $MODELS; do
    base=$(basename "$m" .onnx)
    printf '\n== %s: one process, N sessions x 8/N threads\n' "$base"
    for shape in "1 8" "2 4" "4 2" "8 1"; do
        set -- $shape
        row "${base} ${1}x${2}" "$m" "$1" "$2" "$ROUNDS"
    done
done

printf '\n== board state after\n'
printf '   temps_c: %s\n' "$(cat /sys/class/thermal/thermal_zone0/temp \
    /sys/class/thermal/thermal_zone3/temp 2>/dev/null | tr '\n' ' ')"
ls -l "$OUT" | tail -12
