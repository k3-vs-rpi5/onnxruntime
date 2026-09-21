#!/bin/sh
# Board side: locate the session-shape effect that ONNX Runtime's own intra-op pool has on a graph
# whose EP claims only part of its nodes (round 6, `OPTIMISATION2-20260921.md`).
#
# The vendor matrix command pins that pool at one thread (`-x 1`). Everything else is held fixed at
# the vendor's own option set, so the pool width is the only variable:
#
#   * the EP keeps its own width in both runs (`SPACEMIT_EP_INTRA_THREAD_NUM|8`);
#   * `spacemit-tcm-smi` is sampled while each run is in flight - the AI side must be the same shape
#     (8 blocks busy, naming that run's pid) in both configurations, or the comparison is void;
#   * `SPACEMIT_EP_DEBUG_PROFILE` writes the provider's own Chrome trace; the sum of its
#     `cat=Session` spans is the EP's own execution time. If that is flat while the frame moves, the
#     difference is work outside the EP;
#   * `-p` writes ONNX Runtime's per-node profile, which names the provider of every node. It is a
#     naming instrument only: with a one-thread pool it inflates itself (measured 2.6x), so it is
#     never used for a quantity.
#
#   board-session-shape.sh            # both configurations of $M
#
# Env: ORT (package root), M (model), OUT (output directory). The host pushes this script and pulls
# `$OUT/{x,ortprof}-x{1,8}.log`, `$OUT/trace-ppl-x{1,8}_*.json`, `$OUT/ortprof.json_*.json`.
set -eu
ORT=${ORT:-/tmp/ort206}
M=${M:-/root/onnx-zoo2/pp_liteseg.q.onnx}
OUT=${OUT:-/tmp/onnx-baseline}
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$ORT/lib"
perf="$ORT/bin/onnxruntime_perf_test"

run_pair() {
    label=$1
    shift
    for x in 1 8; do
        rm -f "$OUT/trace-ppl-x$x"_*.json
        setsid nohup "$perf" "$M" -e spacemit -r 10 -x "$x" -S 1 -s -c 1 -I \
            -i "SPACEMIT_EP_INTRA_THREAD_NUM|8 SPACEMIT_EP_DEBUG_PROFILE|$OUT/trace-ppl-x$x" \
            "$@" > "$OUT/$label-x$x.log" 2>&1 &
        pid=$!
        printf '== %s -x%s (pid %s): tcm samples naming the pid\n' "$label" "$x" "$pid"
        i=0
        while [ "$i" -lt 25 ]; do
            spacemit-tcm-smi 2>/dev/null | grep " $pid " | sed 's/^/   /'
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
            i=$((i + 1))
        done
        wait "$pid" || true
        printf '   %s -x%s avg: %s\n' "$label" "$x" \
            "$(sed -n 's/^Average inference time cost total: \(.*\)$/\1/p' "$OUT/$label-x$x.log" | tail -1)"
    done
}

run_pair x
run_pair ortprof -p "$OUT/ortprof.json"

printf '\n== artifacts\n'
ls -l "$OUT"/trace-ppl-x*.json "$OUT"/ortprof.json_*.json 2>&1 | head -20
