#!/bin/sh
# Board side: the execution provider's own options, which the session-level sweep never touched.
#
# The YOLO family is EP-complete (round 6 §3), so no ONNX Runtime session setting can move it - and
# the published rows were measured with the provider's defaults. What is left is the provider itself:
# which cores its threads take (`SPACEMIT_EP_INTRA_THREAD_AFFINITY`), and how the eight AI cores are
# split between parallel threads and simultaneous streams (`INTER x INTRA`).
#
#   board-ep-options.sh <model>
#
# Env: ORT, OUT, REPS.
set -eu
ORT=${ORT:-/tmp/ort206}
OUT=${OUT:-/tmp/onnx-ep-options}
REPS=${REPS:-10}
model=$1
tag=$(basename "$model" .onnx)
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$ORT/lib"
perf="$ORT/bin/onnxruntime_perf_test"

row() {
    name=$1
    parallel=$2
    opts=$3
    if "$perf" "$model" -e spacemit -r "$REPS" -x 8 -S 1 -s -c "$parallel" -I \
        -i "$opts" > "$OUT/$tag-$name.log" 2>&1; then
        avg=$(sed -n 's/^Average inference time cost total: \(.*\)$/\1/p' "$OUT/$tag-$name.log" | tail -1)
        per=$(sed -n 's/^Number of inferences per second: \(.*\)$/\1/p' "$OUT/$tag-$name.log" | tail -1)
        printf '%-26s -c %-2s avg=%-14s inf/s=%s\n' "$name" "$parallel" "$avg" "$per"
    else
        printf '%-26s -c %-2s FAILED: %s\n' "$name" "$parallel" "$(tail -2 "$OUT/$tag-$name.log" | tr '\n' ' ')"
    fi
}

echo "== $tag : which cores the provider's threads take (one stream)"
row intra8-default 1 "SPACEMIT_EP_INTRA_THREAD_NUM|8"
row intra4-default 1 "SPACEMIT_EP_INTRA_THREAD_NUM|4"
row intra4-contig 1 "SPACEMIT_EP_INTRA_THREAD_NUM|4 SPACEMIT_EP_INTRA_THREAD_AFFINITY|8;9;10;11"
row intra4-pairs 1 "SPACEMIT_EP_INTRA_THREAD_NUM|4 SPACEMIT_EP_INTRA_THREAD_AFFINITY|8;10;12;14"
row intra4-odd 1 "SPACEMIT_EP_INTRA_THREAD_NUM|4 SPACEMIT_EP_INTRA_THREAD_AFFINITY|9;11;13;15"
row intra2-pairs 1 "SPACEMIT_EP_INTRA_THREAD_NUM|2 SPACEMIT_EP_INTRA_THREAD_AFFINITY|8;12"
row intra1 1 "SPACEMIT_EP_INTRA_THREAD_NUM|1"

echo "== $tag : eight cores as threads x streams"
row x8-1stream 1 "SPACEMIT_EP_INTRA_THREAD_NUM|8 SPACEMIT_EP_INTER_THREAD_NUM|1"
row x8-2streams 2 "SPACEMIT_EP_INTRA_THREAD_NUM|4 SPACEMIT_EP_INTER_THREAD_NUM|2"
row x8-4streams 4 "SPACEMIT_EP_INTRA_THREAD_NUM|2 SPACEMIT_EP_INTER_THREAD_NUM|4"
row x8-8streams 8 "SPACEMIT_EP_INTRA_THREAD_NUM|1 SPACEMIT_EP_INTER_THREAD_NUM|8"
