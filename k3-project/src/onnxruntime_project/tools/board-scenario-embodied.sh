#!/bin/sh
# Board side: scenario S8, the embodied perception stack - a detector feeding a depth model in one
# process - on its own, and then with the VLM language service running beside it.
#
#   board-scenario-embodied.sh <tag> <pipeline|both> <vlm-config-dir> <concurrency> [tA] [tB] [iters]
#
# Env: OUT, PORT, IMG, TOKENS, CLIENT_ROUNDS, A (detector), B (depth), PIPE, ORT.
set -eu
tag=$1
mode=$2
cfg=$3
conc=$4
tA=${5:-4}
tB=${6:-4}
iters=${7:-60}
OUT=${OUT:-/tmp/scenario-embodied}
PORT=${PORT:-8099}
IMG=${IMG:-/root/pic/frame-a.png}
TOKENS=${TOKENS:-32}
CLIENT_ROUNDS=${CLIENT_ROUNDS:-2}
CLIENT=${CLIENT:-/root/vlm-service-shape-client.py}
A=${A:-/root/onnx-zoo/yolov8n.q.onnx}
B=${B:-/root/alt-models/yolo26n-depth.fp16.onnx}
PIPE=${PIPE:-/root/pipeline}
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

printf '== cell %s (mode=%s A=%s B=%s tA=%s tB=%s iters=%s) temps=%s\n' \
    "$tag" "$mode" "$(basename "$A")" "$(basename "$B")" "$tA" "$tB" "$iters" \
    "$(cat /sys/class/thermal/thermal_zone0/temp)"

free_blocks=$(spacemit-tcm-smi | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')
if [ "$free_blocks" != "8" ]; then
    spacemit-tcm-smi -c > "$OUT/$tag-tcm-clear.txt" 2>&1 || true
    sleep 1
fi

run_pipeline() {
    label=$1
    shift
    : > "$OUT/$tag-$label-tcm.txt"
    "$PIPE" "$A" "$B" "$@" > "$OUT/$tag-$label.log" 2>&1 &
    pid=$!
    i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 60 ]; do
        pattern=$(ls /proc/"$pid"/task 2>/dev/null | tr '\n' '|' | sed 's/|$//')
        [ -n "$pattern" ] && spacemit-tcm-smi 2>/dev/null | grep -E " ($pattern) " >> "$OUT/$tag-$label-tcm.txt" || true
        sleep 0.2
        i=$((i + 1))
    done
    wait "$pid" || true
    printf '%-24s %s | tcm lines=%s blocks=[%s]\n' "$tag-$label" \
        "$(grep -hE 'A_ms|frame_ms|frames_per_s' "$OUT/$tag-$label.log" | tail -1)" \
        "$(wc -l < "$OUT/$tag-$label-tcm.txt")" \
        "$(awk '{print $1}' "$OUT/$tag-$label-tcm.txt" | sort -u | tr '\n' ' ')"
}

if [ "$mode" = "pipeline" ]; then
    run_pipeline serial serial "$iters" "$tA" "$tB"
    run_pipeline overlap overlap "$iters" "$tA" "$tB"
    exit 0
fi

stop_servers
sleep 1
gguf=$(ls "$(dirname "$cfg")"/*.gguf 2>/dev/null | head -1)
[ -n "$gguf" ] || { echo "FATAL: no text model next to $cfg" >&2; exit 1; }
server_log="$OUT/$tag.server.log"
setsid nohup llama-server -m "$gguf" --media-backend smt --smt-config-dir "$cfg" \
    -ctk f16 -ctv f16 -t 8 -c 1024 --host 127.0.0.1 --port "$PORT" \
    ${EXTRA_ARGS:-} > "$server_log" 2>&1 &
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
grep -a -m1 "vision. Spacemit EP enabled" "$server_log" | sed 's/^/banner: /'
[ "$(pgrep -f 'llama-server -m' | wc -l)" = "1" ] || { echo "FATAL: server count != 1" >&2; exit 1; }

# The pipeline must run *while* the client rounds run: a synchronous pipeline followed by the client
# is a sequence, not a shared board, and the first version of this cell made exactly that mistake.
: > "$OUT/$tag-both-overlap-tcm.txt"
"$PIPE" "$A" "$B" overlap "$iters" "$tA" "$tB" > "$OUT/$tag-both-overlap.log" 2>&1 &
pipe_pid=$!
(
    i=0
    while kill -0 "$pipe_pid" 2>/dev/null && [ "$i" -lt 200 ]; do
        pattern=$(ls /proc/"$pipe_pid"/task 2>/dev/null | tr '\n' '|' | sed 's/|$//')
        [ -n "$pattern" ] && spacemit-tcm-smi 2>/dev/null | grep -E " ($pattern) " \
            >> "$OUT/$tag-both-overlap-tcm.txt" || true
        sleep 0.2
        i=$((i + 1))
    done
) &
sampler_pid=$!
r=1
while [ "$r" -le "$CLIENT_ROUNDS" ]; do
    printf -- '-- client round %s/%s\n' "$r" "$CLIENT_ROUNDS"
    python3 "$CLIENT" "$PORT" "$IMG" "$conc" "$TOKENS" 2>&1 | tee -a "$OUT/$tag.client.txt"
    r=$((r + 1))
done
wait "$pipe_pid" || true
kill "$sampler_pid" 2>/dev/null || true
printf '%-24s %s | tcm lines=%s blocks=[%s]\n' "$tag-both-overlap" \
    "$(grep -hE 'A_ms|frame_ms|frames_per_s' "$OUT/$tag-both-overlap.log" | tail -1)" \
    "$(wc -l < "$OUT/$tag-both-overlap-tcm.txt")" \
    "$(awk '{print $1}' "$OUT/$tag-both-overlap-tcm.txt" | sort -u | tr '\n' ' ')"
stop_servers
printf '== cell %s done; blocks %s/8; temps=%s\n' "$tag" \
    "$(spacemit-tcm-smi | sed -n 's/.*available_blocks=\([0-9]*\)\/8.*/\1/p')" \
    "$(cat /sys/class/thermal/thermal_zone0/temp)"
