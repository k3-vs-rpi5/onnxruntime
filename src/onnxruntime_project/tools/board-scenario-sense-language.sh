#!/bin/sh
# Board side: one cell of the "perception + language" scenario - a detector running beside the VLM
# service - with the same lifecycle discipline the service-shape round established.
#
# The question this scenario answers: the VLM service already keeps the encoder and the text model
# on the eight AI cores; can a perception stage (YOLOv8n) share the board with it, and what does each
# side pay? It is the deployment shape an embodied stack needs, and it is a scenario, not an
# optimisation: nothing here is tuned for a win.
#
#   board-scenario-sense-language.sh <tag> <vlm|detector|both> <config-dir> <concurrency> [det-threads]
#
# Env: OUT, PORT, IMG, TOKENS, M (VLM model dir), DET (detector model), DET_REPS, CLIENT_ROUNDS,
#      TCM_SAMPLES. Size DET_REPS so the detector's whole run overlaps the client rounds: the
#      detector's average is only a contention number if it is measured while the VLM is loaded.
set -eu
tag=$1
mode=$2
cfg=$3
conc=$4
det_threads=${5:-2}
OUT=${OUT:-/tmp/scenario}
PORT=${PORT:-8099}
IMG=${IMG:-/root/pic/frame-a.png}
TOKENS=${TOKENS:-32}
DET=${DET:-/root/onnx-zoo/yolov8n.q.onnx}
DET_REPS=${DET_REPS:-4000}
TCM_SAMPLES=${TCM_SAMPLES:-40}
CLIENT_ROUNDS=${CLIENT_ROUNDS:-1}
CLIENT=${CLIENT:-/root/vlm-service-shape-client.py}
ORT=${ORT:-/tmp/ort206}
export LD_LIBRARY_PATH="$ORT/lib"
mkdir -p "$OUT"

stop_servers() {
    pkill -f "llama-server -m" 2>/dev/null || true
    i=0
    while [ "$i" -lt 20 ] && pgrep -f "llama-server -m" >/dev/null 2>&1; do sleep 0.5; i=$((i+1)); done
    pgrep -f "llama-server -m" >/dev/null 2>&1 && { pkill -9 -f "llama-server -m" || true; sleep 1; }
    spacemit-tcm-smi -c >/dev/null 2>&1 || true
}

clean_board() {
    free_blocks=$(spacemit-tcm-smi | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')
    if [ "${free_blocks:-0}" != "8" ]; then
        spacemit-tcm-smi -c > "$OUT/$tag-tcm-clear.txt" 2>&1 || true
        sleep 1
    fi
    free_blocks=$(spacemit-tcm-smi | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')
    [ "$free_blocks" = "8" ] || { echo "FATAL: $free_blocks/8 blocks free" >&2; exit 2; }
}

printf '== cell %s (mode=%s cfg=%s conc=%s det_threads=%s) temps=%s\n' \
    "$tag" "$mode" "$cfg" "$conc" "$det_threads" "$(cat /sys/class/thermal/thermal_zone0/temp)"

if [ "$mode" = "detector" ]; then
    clean_board
    log="$OUT/$tag.log"
    "$ORT/bin/onnxruntime_perf_test" "$DET" -e spacemit -r "$DET_REPS" -x 1 -S 1 -c 1 -I \
        -i "SPACEMIT_EP_INTRA_THREAD_NUM|$det_threads" > "$log" 2>&1
    printf 'detector alone t%s: %s\n' "$det_threads" \
        "$(sed -n 's/^Average inference time cost total: \(.*\)$/\1/p' "$log" | tail -1)"
    printf 'tcm busy: %s/8\n' "$(spacemit-tcm-smi | grep -c busy || true)"
    exit 0
fi

stop_servers
sleep 1
clean_board
gguf=$(ls "${M:-$(dirname "$cfg")}"/*.gguf 2>/dev/null | head -1)
[ -n "$gguf" ] || { echo "FATAL: no text model next to $cfg" >&2; exit 1; }

server_log="$OUT/$tag.server.log"
setsid nohup llama-server -m "$gguf" --media-backend smt --smt-config-dir "$cfg" \
    -ctk f16 -ctv f16 -t 8 -c 1024 --host 127.0.0.1 --port "$PORT" \
    ${EXTRA_ARGS:-} > "$server_log" 2>&1 &
for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done
grep -a -m1 "vision. Spacemit EP enabled" "$server_log" | sed 's/^/banner: /'
[ "$(pgrep -f 'llama-server -m' | wc -l)" = "1" ] || { echo "FATAL: server count != 1" >&2; exit 1; }

det_log="$OUT/$tag.detector.log"
det_pid=
if [ "$mode" = "both" ]; then
    "$ORT/bin/onnxruntime_perf_test" "$DET" -e spacemit -r "$DET_REPS" -x 1 -S 1 -c 1 -I \
        -i "SPACEMIT_EP_INTRA_THREAD_NUM|$det_threads" > "$det_log" 2>&1 &
    det_pid=$!
    sleep 3
fi

: > "$OUT/$tag-tcm.txt"
( i=0; while [ "$i" -lt "$TCM_SAMPLES" ]; do
      spacemit-tcm-smi 2>/dev/null | grep busy >> "$OUT/$tag-tcm.txt" || true
      sleep 1; i=$((i+1))
  done ) &
sampler=$!

r=1
while [ "$r" -le "$CLIENT_ROUNDS" ]; do
    printf -- '-- client round %s/%s\n' "$r" "$CLIENT_ROUNDS"
    python3 "$CLIENT" "$PORT" "$IMG" "$conc" "$TOKENS" 2>&1 | tee -a "$OUT/$tag.client.txt"
    r=$((r + 1))
done

kill "$sampler" 2>/dev/null || true
if [ -n "$det_pid" ]; then
    # Wait for the detector's own summary: its average is the contention number, so it must finish
    # while the VLM load is still on (size DET_REPS to the client window, not larger).
    w=0
    while kill -0 "$det_pid" 2>/dev/null && [ "$w" -lt 40 ]; do sleep 1; w=$((w + 1)); done
    if kill -0 "$det_pid" 2>/dev/null; then
        kill -TERM "$det_pid" 2>/dev/null || true; sleep 2; kill -KILL "$det_pid" 2>/dev/null || true
        printf 'detector alongside: INCOMPLETE (killed after %ss; DET_REPS too large for the window)\n' "$w"
    else
        printf 'detector alongside: %s (finished after %ss inside the client window)\n' \
            "$(sed -n 's/^Average inference time cost total: \(.*\)$/\1/p' "$det_log" | tail -1)" "$w"
    fi
fi
grep -a -E "image prefill|prompt eval time|eval time" "$server_log" | tail -4
printf 'tcm busy lines sampled: %s (blocks seen: %s)\n' \
    "$(wc -l < "$OUT/$tag-tcm.txt")" \
    "$(awk '{print $1}' "$OUT/$tag-tcm.txt" | sort -u | tr '\n' ' ')"
stop_servers
printf '== cell %s done; blocks %s/8; temps=%s\n' "$tag" \
    "$(spacemit-tcm-smi | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')" \
    "$(cat /sys/class/thermal/thermal_zone0/temp)"
