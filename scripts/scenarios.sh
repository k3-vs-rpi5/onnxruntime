#!/bin/sh
# The frozen scenario suite: one command that re-runs the self-contained scenarios against their
# acceptance bands and writes a fresh evidence package.
#
#   scripts/scenarios.sh list       show the suite and each scenario's band
#   scripts/scenarios.sh run        verify materials + build, run the suite on the board, seal results
#
# Scenarios in the suite are the ones that run unattended: S10 (decision layer, board, numpy),
# S12 (OCR chain, board, provider) and S11's metric step (host, from frozen embeddings). S9/S9.3 need
# long-lived model servers and stay as documented recipes in REPRODUCE.md, so they are listed here
# with their bands but not executed by this command.
set -eu

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(CDPATH='' cd -- "$ROOT/../.." && pwd)
BOARD=$WORKSPACE/hardware/k3-monitor/experiments/board.sh
TOOLS=$ROOT/src/onnxruntime_project/tools
M=${ONNX_MATERIALS:-$WORKSPACE/.local/materials}
EVIDENCE=${ONNX_EVIDENCE:-$WORKSPACE/../k3-evidence/onnxruntime}

list() {
    cat <<'EOF'
scenario  what runs                                     band
S10       s10-cv-decision.py on 100 scenes (384x384)    accuracy >= 95/100, latency <= 160 ms
S12       ocr-chain over the 30-page fixture            printed CER <= 0.01; tables <= 1.5x tesseract
S11*      s11-metrics.py over frozen embeddings         NMI >= 0.80; exact-hash duplicates P/R = 1.000
S9        rag-eval.py / rag-answer.py (needs servers)   recall@1 >= 0.98; TTFT < 3 s
S9.3      s9-vendor.py against llm-sdk (needs setup)    grounding on numeric questions (measured 0/5)
S14       decode-headroom study (operator's next ask)   to be defined before it runs
EOF
}

run() {
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    out="$EVIDENCE/suite-$stamp"
    mkdir -p "$out"
    echo "== 1/4 materials"; sh "$SCRIPT_DIR/materials.sh" verify | tee "$out/materials.txt"
    echo "== 2/4 build"; ( cd "$ROOT" && sh scripts/build.sh ) > "$out/build.txt" 2>&1
    grep -E 'bin/' "$out/build.txt"
    echo "== 3/4 board fixtures"
    "$BOARD" push "$EVIDENCE/s10-embodied-20260923/v2" /root/suite-s10 >/dev/null
    "$BOARD" push "$ROOT/build/bin/ocr-chain" /root/suite-ocr-chain >/dev/null
    "$BOARD" run 'ls /root/suite-s10/scenes/*.png | wc -l; spacemit-tcm-smi | sed -n 2p' | tee "$out/board-state.txt"
    echo "== 4/4 scenarios"
    # Every scenario runs REPS times with the board monitor alongside; the verdict uses the median and
    # the spread, and a monitor excursion marks the row incomparable - the fluctuation rule, applied
    # mechanically rather than remembered.
    REPS=${REPS:-3}
    # The two sessions' AI threads. Four and four is the accepted shape on a board whose model path is
    # sound at any thread count; a board that corrupts above four AI threads per process
    # (k3-evidence/onnxruntime/board-215-ai-thread-cap-20260925) declares two and two here instead,
    # and the row it produces is then a declared deviation, not the accepted shape.
    DET_THREADS=${ONNX_DET_THREADS:-4}
    REC_THREADS=${ONNX_REC_THREADS:-4}
    MON=$TOOLS/board-monitor.sh
    "$BOARD" push "$MON" /root/board-monitor.sh >/dev/null
    # S10: decision layer at the declared operating point
    # The monitor writes on the board; a host path here would be expanded into a board command, the
    # write would fail inside the backgrounded subshell and the pull below would silently adopt
    # whatever /tmp/mon-s10.csv last held - a stale window reads as a clean one.
    "$BOARD" run "chmod +x /root/board-monitor.sh; rm -f /tmp/mon-s10.csv; \
(/root/board-monitor.sh 90 2 /tmp/mon-s10.csv s10 &); \
cd /root && for r in \$(seq 1 $REPS); do python3 /root/s10-cv-decision.py /root/suite-s10 100 384 2>&1 | tail -1; done; sleep 3" \
        > "$out/s10.txt" 2>&1 || true
    "$BOARD" pull /tmp/mon-s10.csv "$out/mon-s10.csv" >/dev/null 2>&1 || true
    # S12: OCR chain over the frozen page fixture (PPM pages already on the board from the S12 work)
    "$BOARD" run "cd /root && mkdir -p suite-ocr-out && rm -f suite-ocr-out/*.txt && LD_LIBRARY_PATH=/tmp/ort206/lib ./suite-ocr-chain /root/alt-models/PP-OCRv6_tiny_det_640x640.fp16.onnx /root/alt-models/PP-OCRv6_tiny_rec_48x320.dynq.onnx /root/ppocrv6_tiny_dict.txt /root/s12-ppm/*.ppm -o suite-ocr-out --threads $DET_THREADS $REC_THREADS 2>&1 | tail -1" \
        > "$out/s12.txt" 2>&1 || true
    "$BOARD" pull /root/suite-ocr-out "$out/ocr-out" >/dev/null 2>&1 || true
    for d in "$out/ocr-out" "$out/ocr-out/suite-ocr-out"; do
        [ -d "$d" ] && [ "$(ls "$d" | wc -l)" -gt 5 ] && python3 "$TOOLS/cer-wer.py" \
            "$EVIDENCE/s12-fixture-20260922" "$d" suite > "$out/s12-rates.txt" 2>&1 && break
    done
    # verdicts
    {
        echo "scenario  result"
        acc=$(grep -o 'accuracy=[0-9]*/[0-9]*' "$out/s10.txt" 2>/dev/null | head -1 || echo 'accuracy=?')
        lat=$(grep -o 'latency_median_ms=[0-9.]*' "$out/s10.txt" 2>/dev/null | head -1 || echo 'lat=?')
        echo "S10       $acc $lat"
        if [ -f "$out/mon-s10.csv" ]; then
            vals=$(grep -o 'accuracy=[0-9]*' "$out/s10.txt" | cut -d= -f2 | sed 's/^/s10=/')
            # shellcheck disable=SC2086
            python3 "$TOOLS/variance.py" $vals --band 95:100 --monitor "$out/mon-s10.csv" \
                > "$out/s10-variance.txt" 2>&1 || true
            sed 's/^/S10 var   /' "$out/s10-variance.txt"
        fi
        grep -E '^(printed|scanned|tables)' "$out/s12-rates.txt" 2>/dev/null | sed 's/^/S12       /' || echo "S12       (rates unavailable)"
    } | tee "$out/verdicts.txt"
    echo "evidence: $out"
}

case "${1:-}" in
    list) list ;;
    run) run ;;
    *) echo "usage: scenarios.sh list|run" >&2; exit 2 ;;
esac
