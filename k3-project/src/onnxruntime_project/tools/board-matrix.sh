#!/bin/sh
# ONNX provider baseline matrix, board side.
#
# Instrument: the vendor release package's own onnxruntime_perf_test, invoked with the exact
# option set the published model-zoo tables used
# (modelzoo.md: -e spacemit -r 10 -x 1 -S 1 -s -c 1 -i "SPACEMIT_EP_INTRA_THREAD_NUM|N" -I),
# so the local numbers are comparable with the vendor's K3 table rather than with a re-derived
# harness.
#
#   matrix.sh            run the whole matrix
#
# Output: one log per row in $OUT plus a compact table on stdout. Nothing is written to the
# project tree; evidence is pulled by the host.
set -eu
ORT=${ORT:-/tmp/ort206}
ZOO=${ZOO:-/root/onnx-zoo}
OUT=${OUT:-/tmp/onnx-baseline}
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$ORT/lib"
perf="$ORT/bin/onnxruntime_perf_test"
common="-r 10 -x 1 -S 1 -s -c 1 -I"

row() {
    tag=$1
    shift
    if "$perf" "$@" > "$OUT/$tag.log" 2>&1; then
        avg=$(sed -n 's/^Average inference time cost: \(.*\)$/\1/p' "$OUT/$tag.log" | tail -1)
        first=$(sed -n 's/^First inference time cost: \(.*\)$/\1/p' "$OUT/$tag.log" | tail -1)
        sess=$(sed -n 's/^Session creation time cost: \(.*\)$/\1/p' "$OUT/$tag.log" | tail -1)
        p75=$(sed -n 's/^P75 inference time cost: \(.*\)$/\1/p' "$OUT/$tag.log" | tail -1)
        p90=$(sed -n 's/^P90 inference time cost: \(.*\)$/\1/p' "$OUT/$tag.log" | tail -1)
        printf '%-34s avg=%-10s p75=%-10s p90=%-10s first=%-10s sess=%s\n' "$tag" "$avg" "$p75" "$p90" "$first" "$sess"
    else
        printf '%-34s FAILED: %s\n' "$tag" "$(tail -2 "$OUT/$tag.log" | tr '\n' ' ')"
    fi
}

echo "== provider on the AI cluster, vendor thread option 1/2/4/8"
for n in 1 2 4 8; do
    for m in resnet18.q resnet50.q yolov8n.q yolov8s.q yolov8n-seg.q vit_b_16.q vit_b_16.fp16; do
        row "sm$n-${m%.onnx}" "$ZOO/$m.onnx" -e spacemit $common -i "SPACEMIT_EP_INTRA_THREAD_NUM|$n"
    done
done
for n in 4 8; do
    row "sm$n-fastvlm_vision.f16" "$ZOO/fastvlm_vision.f16.onnx" -e spacemit $common -i "SPACEMIT_EP_INTRA_THREAD_NUM|$n"
done

echo
echo "== CPU EP baseline on the X100 (ORT intra-op threads, no provider options)"
for n in 4 8; do
    for m in resnet50.q yolov8n.q vit_b_16.q; do
        row "cpu$n-${m%.onnx}" "$ZOO/$m.onnx" -e cpu -r 10 -S 1 -s -c 1 -I -x "$n"
    done
done

echo
echo "== provider coverage: CPU EP fallback disabled (a failure names an unclaimed op)"
for m in resnet50.q yolov8n.q vit_b_16.q fastvlm_vision.f16; do
    row "cover-${m%.onnx}" "$ZOO/$m.onnx" -e spacemit $common -i "SPACEMIT_EP_INTRA_THREAD_NUM|8" -C "session.disable_cpu_ep_fallback|1"
done

echo
echo "== subgraph dumps and one profile trace per architecture"
mkdir -p "$OUT/subgraph_vit" "$OUT/subgraph_yolo"
row "dump-vit" "$ZOO/vit_b_16.q.onnx" -e spacemit -r 1 -x 1 -S 1 -I -i "SPACEMIT_EP_INTRA_THREAD_NUM|8 SPACEMIT_EP_DUMP_SUBGRAPHS|$OUT/subgraph_vit"
row "dump-yolo" "$ZOO/yolov8n.q.onnx" -e spacemit -r 1 -x 1 -S 1 -I -i "SPACEMIT_EP_INTRA_THREAD_NUM|8 SPACEMIT_EP_DUMP_SUBGRAPHS|$OUT/subgraph_yolo"
row "trace-vit" "$ZOO/vit_b_16.q.onnx" -e spacemit -r 2 -x 1 -S 1 -I -i "SPACEMIT_EP_INTRA_THREAD_NUM|8 SPACEMIT_EP_DEBUG_PROFILE|$OUT/trace_vit"
row "trace-yolo" "$ZOO/yolov8n.q.onnx" -e spacemit -r 2 -x 1 -S 1 -I -i "SPACEMIT_EP_INTRA_THREAD_NUM|8 SPACEMIT_EP_DEBUG_PROFILE|$OUT/trace_yolo"

echo
echo "== dumps written"
ls -la "$OUT" | tail -20
