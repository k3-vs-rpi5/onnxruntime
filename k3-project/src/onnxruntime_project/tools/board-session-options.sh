#!/bin/sh
# Board side: session-level option sweep for one model, in the deployment shape.
#
# `onnxruntime_perf_test` defaults to graph optimisation level 1 (basic), while an ONNX Runtime
# application gets ORT_ENABLE_ALL by default - so the published matrix, which passes no `-o`, is
# measured in a shape a deployment does not use. This sweeps the levels and the session knobs that
# are actually reachable from the vendor's own tool.
#
#   board-session-options.sh <model>
#
# Env: ORT, OUT, REPS.
set -eu
ORT=${ORT:-/tmp/ort206}
OUT=${OUT:-/tmp/onnx-session}
REPS=${REPS:-10}
model=$1
tag=$(basename "$model" .onnx)
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$ORT/lib"
perf="$ORT/bin/onnxruntime_perf_test"
common="-e spacemit -r $REPS -x 8 -S 1 -s -c 1 -I -i SPACEMIT_EP_INTRA_THREAD_NUM|8"

row() {
    name=$1
    shift
    # shellcheck disable=SC2086
    if "$perf" "$model" $common "$@" > "$OUT/$tag-$name.log" 2>&1; then
        printf '%-24s avg=%s\n' "$name" \
            "$(sed -n 's/^Average inference time cost total: \(.*\)$/\1/p' "$OUT/$tag-$name.log" | tail -1)"
    else
        printf '%-24s FAILED: %s\n' "$name" "$(tail -2 "$OUT/$tag-$name.log" | tr '\n' ' ')"
    fi
}

echo "== $tag (EP 8 threads, ORT intra-op 8)"
row published-default
row o99 -o 99
row o2 -o 2
row o0 -o 0
row o99-parallel -o 99 -P
row o99-noarena -o 99 -A -M
row o99-nospin -o 99 -C "session.intra_op.allow_spinning|0"
row o99-envallocs -o 99 -C "session.use_env_allocators|1"
