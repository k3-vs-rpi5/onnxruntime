#!/bin/sh
# AI-cluster engagement evidence for one provider run (board side).
#
# The workspace contract forbids per-CPU counters and per-process CPU time as evidence of AI-core
# work, because on this board those instruments do not see the cluster. What is allowed: the
# runtime's own thread registration, the driver's TCM occupancy naming the workload's pid, and the
# throughput arithmetic. This script samples the middle one while a long provider run is in flight
# and records the process's thread set (count, affinity, names).
#
#   evidence.sh <tag> <model> <intra_threads> <repeats>
set -eu
tag=$1
model=$2
threads=$3
reps=$4
ORT=${ORT:-/tmp/ort206}
OUT=${OUT:-/tmp/onnx-baseline}
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$ORT/lib"

setsid nohup "$ORT/bin/onnxruntime_perf_test" "$model" -e spacemit -r "$reps" -x 1 -S 1 -s -c 1 -I \
    -i "SPACEMIT_EP_INTRA_THREAD_NUM|$threads" > "$OUT/ev-$tag.log" 2>&1 &
sleep 4
pid=$(pgrep -f "onnxruntime_perf_test .*$model" | head -1)
{
    echo "# $tag: pid=$pid model=$model SPACEMIT_EP_INTRA_THREAD_NUM=$threads repeats=$reps"
    echo "# date: $(date -Ins)"
    echo
    echo "## spacemit-tcm-smi while the run is in flight"
    spacemit-tcm-smi
    echo
    echo "## thread summary (count, then per-thread cpu/comm)"
    echo "thread_count=$(ls /proc/$pid/task | wc -l)"
    ps -o pid,tid,psr,pcpu,comm -L -p "$pid" 2>/dev/null | head -30
    echo
    echo "## per-thread affinity (cpus_allowed_list), one line per tid"
    for t in $(ls /proc/$pid/task | head -40); do
        printf '%s %s\n' "$t" "$(cat /proc/$pid/task/$t/status 2>/dev/null | sed -n 's/^Cpus_allowed_list:\s*//p')"
    done
} > "$OUT/ev-$tag.txt" 2>&1
sleep 2
{
    echo
    echo "## spacemit-tcm-smi after the run"
    spacemit-tcm-smi
    echo
    echo "## run result"
    grep -E "Average inference time cost|Total inference time|P90|First inference" "$OUT/ev-$tag.log" || true
} >> "$OUT/ev-$tag.txt" 2>&1
wait || true
cat "$OUT/ev-$tag.txt"
