#!/bin/sh
# Deliver this project's sealed build inputs to the board, run there, and keep the raw output as a
# new evidence directory. Board credentials come from the workspace record .ssh-k3 (read inside this
# process only, never printed); the board never selects or builds its own code.
#
#   scripts/run.sh verify
#       offline check of build/SHA256SUMS against the files on disk
#   scripts/run.sh smoke
#       the built harness delivered and run on the board (one 8-thread session, yolov8n)
#   scripts/run.sh ime-probe
#       the AI-cluster ceiling instrument (three probes, 1/2/4/8 threads, three rounds each)
#   scripts/run.sh instrument <board-script> [args...]
#       deliver and run any instrument from src/onnxruntime_project/tools and pull its output
#
# Evidence lands in $ONNX_EVIDENCE/<workflow>-<UTC>/ and is printed at the end. Nothing is written
# into the project tree by a run.
set -eu

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(CDPATH='' cd -- "$ROOT/../.." && pwd)
BUILD=${ONNX_BUILD:-$ROOT/build}
TOOLS=$ROOT/src/onnxruntime_project/tools
CREDENTIALS=${ONNX_CREDENTIALS:-"$WORKSPACE/.ssh-k3"}
EVIDENCE=$(CDPATH='' cd -- "${ONNX_EVIDENCE:-$WORKSPACE/../k3-evidence/onnxruntime}" 2>/dev/null \
    && pwd || echo "${ONNX_EVIDENCE:-$WORKSPACE/../k3-evidence/onnxruntime}")
ORT=${ONNX_ORT:-/tmp/ort206}
ZOO=${ONNX_ZOO:-/root/onnx-zoo}

command -v sshpass >/dev/null 2>&1 || { echo "run: sshpass is required" >&2; exit 1; }
[ -r "$CREDENTIALS" ] || { echo "run: cannot read credentials: $CREDENTIALS" >&2; exit 1; }
mode=$(stat -c '%a' "$CREDENTIALS")
[ "$mode" = "600" ] || { echo "run: credentials must be mode 0600 (found $mode)" >&2; exit 1; }

board_host=; board_user=; board_secret=
while read -r board_host_field board_user_field board_secret_field _rest; do
    board_host=$board_host_field; board_user=$board_user_field; board_secret=$board_secret_field
    break
done < "$CREDENTIALS"
[ -n "$board_host" ] && [ -n "$board_user" ] && [ -n "$board_secret" ] ||
    { echo "run: credentials record must contain: host account secret" >&2; exit 1; }

SSHPASS=$board_secret
export SSHPASS
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$WORKSPACE/.local/known_hosts \
-o ConnectTimeout=20 -o NumberOfPasswordPrompts=1 -o PreferredAuthentications=password,publickey"
DEST="$board_user@$board_host"
remote() { sshpass -e ssh $SSH_OPTS "$DEST" "$@"; }
push() { sshpass -e scp -q -r $SSH_OPTS "$1" "$DEST:$2"; }
pull() { sshpass -e scp -q -r $SSH_OPTS "$DEST:$1" "$2"; }

new_evidence() {
    dir="$EVIDENCE/$1-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$dir"
    printf '%s' "$dir"
}

case "${1:-}" in
    verify)
        ( cd "$BUILD" && sha256sum -c SHA256SUMS ) || exit 1
        echo "verify: build/ matches SHA256SUMS"
        ;;

    smoke)
        ev=$(new_evidence smoke)
        echo "== smoke: deliver $BUILD/bin/pipeline and run one 8-thread session"
        push "$BUILD/bin/pipeline" /root/pipeline-run
        remote "chmod +x /root/pipeline-run; LD_LIBRARY_PATH=$ORT/lib /root/pipeline-run \
                $ZOO/yolov8n.q.onnx $ZOO/yolov8n.q.onnx same 1 5 8" > "$ev/harness.txt" 2>&1
        remote "sha256sum /root/pipeline-run $(echo "$ZOO")/yolov8n.q.onnx; \
                spacemit-tcm-smi | sed -n 2p" > "$ev/identity.txt" 2>&1
        cat "$ev/harness.txt"
        echo "evidence: $ev"
        ;;

    ime-probe)
        ev=$(new_evidence ime-probe)
        echo "== ime-probe: ceiling instrument, three probes x 1/2/4/8 threads x 3 rounds"
        push "$TOOLS/board-ime-probe.sh" /root/board-ime-probe.sh
        push "$BUILD/probes/ime-probe-1024.onnx" /root/ime-probe/ime-probe-1024.onnx
        push "$BUILD/probes/ime-probe-2048.onnx" /root/ime-probe/ime-probe-2048.onnx
        push "$BUILD/probes/ime-probe-4096.onnx" /root/ime-probe/ime-probe-4096.onnx
        remote "PROBES='1024 2048 4096' OUT=/tmp/ime-probe-run ROUNDS=3 \
                sh /root/board-ime-probe.sh" > "$ev/session.txt" 2>&1
        pull /tmp/ime-probe-run "$ev/run"
        ( cd "$ROOT" && python3 src/onnxruntime_project/tools/ime-probe-report.py \
            "$ev/run" --probe p1024=1024,1024,1024 --probe p2048=2048,2048,2048 \
            --probe p4096=4096,4096,4096 ) > "$ev/ceiling-table.txt"
        tail -12 "$ev/ceiling-table.txt"
        echo "evidence: $ev"
        ;;

    instrument)
        shift
        [ $# -ge 1 ] || { echo "usage: scripts/run.sh instrument <board-script> [args...]" >&2; exit 2; }
        name=$1
        shift
        script=$TOOLS/$name
        [ -f "$script" ] || { echo "run: no such instrument: $script" >&2; exit 2; }
        tag=$(echo "$name" | sed 's/\.sh$//')
        ev=$(new_evidence "$tag")
        echo "== instrument $name $*"
        push "$script" "/root/$name"
        remote "OUT=/tmp/instrument-run sh /root/$name $*" > "$ev/session.txt" 2>&1
        pull /tmp/instrument-run "$ev/run"
        cat "$ev/session.txt" | tail -20
        echo "evidence: $ev"
        ;;

    *)
        echo "usage: scripts/run.sh verify|smoke|ime-probe|instrument <board-script> [args...]" >&2
        exit 2
        ;;
esac
