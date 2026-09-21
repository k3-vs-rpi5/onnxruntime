#!/bin/sh
# Board side: one per-node profile per model, for the provider-coverage census.
#
# The census answers one deployment question - how much of a frame does the SpacemiT EP hand back to
# the CPU EP - and it answers it in the shape a deployment runs: EP width explicit, ONNX Runtime's
# own intra-op pool at the same width (round 6 measured that leaving it at one costs 2.32x on a
# graph with a fallback).
#
#   board-coverage-census.sh <model> [<model> ...]
#
# Env: ORT (package root), OUT (output directory), REPS. The profile lands as $OUT/<tag>.json and the
# run log as $OUT/<tag>.log; the host parses them with `ort-provider-census.py`.
set -eu
ORT=${ORT:-/tmp/ort206}
OUT=${OUT:-/tmp/onnx-census}
REPS=${REPS:-5}
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$ORT/lib"
perf="$ORT/bin/onnxruntime_perf_test"

for model in "$@"; do
    tag=$(basename "$model" .onnx)
    rm -f "$OUT/$tag.json"_*.json
    if "$perf" "$model" -e spacemit -r "$REPS" -x 8 -S 1 -s -c 1 -I \
        -i "SPACEMIT_EP_INTRA_THREAD_NUM|8" -p "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1; then
        printf '%-28s avg=%s\n' "$tag" \
            "$(sed -n 's/^Average inference time cost total: \(.*\)$/\1/p' "$OUT/$tag.log" | tail -1)"
    else
        printf '%-28s FAILED: %s\n' "$tag" "$(tail -2 "$OUT/$tag.log" | tr '\n' ' ')"
    fi
done

ls -l "$OUT"/*.json_*.json 2>/dev/null | tail -20
