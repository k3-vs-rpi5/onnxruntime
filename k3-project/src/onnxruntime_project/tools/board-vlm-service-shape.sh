#!/bin/sh
# Board side: one VLM service-shape cell, with a server lifecycle that cannot lie.
#
# The measurement is "one server, N simultaneous image requests", and its whole validity rests on the
# server that answers being the one this cell started. That is not automatic: llama-server does not
# always leave on SIGTERM, and a surviving server holds the port, so the next cell's server fails to
# bind and the client silently measures the *previous* configuration. This script therefore
#
#   1. stops any existing server, waits, escalates to SIGKILL, and refuses to continue if one survives;
#   2. refuses to start if the port is still answered;
#   3. prints the banner line from the new server's own log, so the cell carries the configuration
#      that actually served it;
#   4. asserts exactly one server process is alive before the client runs.
#
#   board-vlm-service-shape.sh <config-dir> <concurrency> <tag>
#
# Env: M (model dir), GGUF (text model; defaults to the single *.gguf in the config dir's parent),
#      PORT, IMG, TOKENS, STAGGER_MS, OUT.
set -eu
# The config directory is the argument; the model directory is its parent unless M says otherwise, so
# that a cell for a second model cannot silently load the first model's text weights.
CFG_ARG=$1
M=${M:-$(dirname "$CFG_ARG")}
PORT=${PORT:-8099}
IMG=${IMG:-/root/pic/frame-a.png}
TOKENS=${TOKENS:-32}
CLIENT=${CLIENT:-/root/vlm-service-shape-client.py}
OUT=${OUT:-/tmp}
CFG=$CFG_ARG
CONC=$2
TAG=$3

# The text model is the .gguf next to the config directory, found rather than hard-coded so the same
# script serves every model in the vendor's VLM family.
if [ -z "${GGUF:-}" ]; then
    GGUF=$(ls "$M"/*.gguf 2>/dev/null | head -1)
fi
[ -n "$GGUF" ] && [ -f "$GGUF" ] || { echo "FATAL: no text model found under $M" >&2; exit 1; }

stop_servers() {
    pkill -f "llama-server -m" 2>/dev/null || true
    i=0
    while [ "$i" -lt 20 ] && pgrep -f "llama-server -m" >/dev/null 2>&1; do
        sleep 0.5
        i=$((i + 1))
    done
    if pgrep -f "llama-server -m" >/dev/null 2>&1; then
        pkill -9 -f "llama-server -m" 2>/dev/null || true
        sleep 1
    fi
    if pgrep -f "llama-server -m" >/dev/null 2>&1; then
        echo "FATAL: a llama-server survived SIGKILL; refusing to measure" >&2
        exit 1
    fi
    spacemit-tcm-smi -c >/dev/null 2>&1 || true
}

stop_servers
sleep 1
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "FATAL: something still answers on port $PORT" >&2
    exit 1
fi

log="$OUT/vlm-cell-$TAG.server.log"
# EXTRA_ARGS carries the per-model flags the vendor's own documentation uses (for the Qwen3.5 family
# that is `--reasoning-budget 0 --reasoning off`, without which the answers come back empty).
setsid nohup llama-server -m "$GGUF" --media-backend smt \
    --smt-config-dir "$CFG" -ctk f16 -ctv f16 -t 8 -c 1024 \
    --host 127.0.0.1 --port "$PORT" ${EXTRA_ARGS:-} > "$log" 2>&1 &
for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done

echo "==== cell $TAG (config $CFG, concurrency $CONC, stagger ${STAGGER_MS:-0} ms)"
grep -a -m1 "vision. Spacemit EP enabled" "$log" | sed 's/^/banner: /'
echo "servers alive: $(pgrep -f 'llama-server -m' | wc -l)"
if [ "$(pgrep -f 'llama-server -m' | wc -l)" != "1" ]; then
    echo "FATAL: expected exactly one server" >&2
    exit 1
fi
grep -a -i "address already in use\|failed to bind" "$log" && {
    echo "FATAL: the log reports a bind failure" >&2
    exit 1
}

# MODE=image (default) drives N simultaneous image requests; MODE=decode drives the text-only
# `/completion` sweep with ignore_eos, which is the decode-throughput measurement.
if [ "${MODE:-image}" = "decode" ]; then
    python3 "$CLIENT" "$PORT" "$CONC" "$TOKENS" 2>&1
else
    python3 "$CLIENT" "$PORT" "$IMG" "$CONC" "$TOKENS" 2>&1
fi
grep -a -E "image prefill|prompt eval time|eval time" "$log" | tail -4
echo "tcm busy at the end: $(spacemit-tcm-smi | grep -c busy)"
stop_servers
