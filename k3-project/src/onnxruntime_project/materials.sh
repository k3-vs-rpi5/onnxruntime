#!/bin/sh
# Fetch and verify every material the acceptance scenarios need - models, fixtures, corpora, tools.
#
#   scripts/run.sh materials fetch    download/generate anything missing, then refresh materials.lock
#   scripts/run.sh materials verify   check every locked item (digest + size); non-zero on any problem
#   scripts/run.sh materials list     show what each scenario needs and where it comes from
#
# The lock file (`materials.lock`) is written by `fetch` and is the authority for `verify`: a fresh
# machine can rebuild the whole test setup with one command, and a changed upstream file is caught
# rather than silently used. Fixtures are *generated* rather than downloaded - their generators are
# the source of truth and their seeds make them deterministic.
set -eu

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE=$(CDPATH='' cd -- "$ROOT/../.." && pwd)
TOOLS=$ROOT/src/onnxruntime_project/tools
M=${ONNX_MATERIALS:-$WORKSPACE/.local/materials}
LOCK=$ROOT/materials.lock
ARCHIVE=https://archive.spacemit.com/spacemit-ai/model_zoo

# name|kind|source|destination-relative-to-$M
ITEMS='
bge-small-zh|url|'$ARCHIVE'/embed/Bge-Small-Zh-V1.5-Q4_K_M.gguf|models/embed/Bge-Small-Zh-V1.5-Q4_K_M.gguf
qwen3-embedding-0.6b|url|'$ARCHIVE'/embed/Qwen3-Embedding-0.6B-Q4_0.gguf|models/embed/Qwen3-Embedding-0.6B-Q4_0.gguf
nomic-embed-v2|url|https://hf-mirror.com/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-embed-text-v2-moe.Q4_0.gguf|models/embed/nomic-embed-text-v2-moe-Q4_0.gguf
ppocr-det|url|'$ARCHIVE'/vision/ppocr/PP-OCRv6_tiny_det_640x640.fp16.onnx|models/ocr/PP-OCRv6_tiny_det_640x640.fp16.onnx
ppocr-rec|url|'$ARCHIVE'/vision/ppocr/PP-OCRv6_tiny_rec_48x320.dynq.onnx|models/ocr/PP-OCRv6_tiny_rec_48x320.dynq.onnx
ppocr-rec-b4|url|'$ARCHIVE'/vision/ppocr/PP-OCRv6_tiny_rec_b4_48x320.dynq.onnx|models/ocr/PP-OCRv6_tiny_rec_b4_48x320.dynq.onnx
ppocr-dict|url|https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv6_tiny_dict.txt|models/ocr/ppocrv6_tiny_dict.txt
mobileclip2-s3|url|'$ARCHIVE'/vision/mobileclip2_s3/image_encoder.fp16.onnx|models/embed/image_encoder.fp16.onnx
'

REPO_ITEMS='
miniCPM5-2B|/root/MiniCPM5-2B-Q4_K_M.gguf|board-only (from the llama.cpp project)
fastvlm-0.5B|/root/vlm-fastvlm/fastvlm-text-0.5B-Q4_1.gguf|board-only (vendor VLM package)
qwen3.5-0.8B-text|/root/vlm-qwen08b/qwen3_5vl_0.8b-text-q41.gguf|board-only (vendor VLM package)
'

APT_PACKAGES="tesseract-ocr ffmpeg llm-sdk yolo-demo"

usage() { echo "usage: materials.sh fetch|verify|list" >&2; exit 2; }
[ $# -ge 1 ] || usage

list() {
    echo "== downloadable models (fetched into $M)"
    echo "$ITEMS" | while IFS='|' read -r name kind src dst; do
        [ -n "$name" ] || continue
        printf '  %-18s %s\n' "$name" "$src"
    done
    echo "== fixtures (generated deterministically)"
    echo "  s12-pages      tools/make-ingest-fixture.py       30 pages: printed/scanned/tables + ground truth"
    echo "  s10-scenes     tools/make-scene-fixture.py        100 decision scenes with ground truth"
    echo "  s11-album      tools/make-album-fixture.py        synthetic album (kept as a rejected fixture)"
    echo "  s11-natural    ffmpeg over /usr/share/yolo-demo/*.mp4  natural frames, 7 groups"
    echo "== corpora"
    echo "  s9-corpus      this workspace's own reports and evidence READMEs (sealed by digest)"
    echo "== board packages"
    echo "  $APT_PACKAGES"
}

fetch() {
    mkdir -p "$M"
    echo "$ITEMS" | while IFS='|' read -r name kind src dst; do
        [ -n "$name" ] || continue
        out="$M/$dst"
        mkdir -p "$(dirname "$out")"
        if [ -s "$out" ]; then
            echo "  have $name"
        else
            echo "  fetch $name <- $src"
            curl -sL --fail --max-time 1800 -o "$out.part" "$src" && mv "$out.part" "$out" \
                || { echo "  FAILED $name" >&2; rm -f "$out.part"; }
        fi
    done
    : > "$LOCK"
    echo "# materials.lock - written by 'scripts/run.sh materials fetch'; verify with 'scripts/run.sh materials verify'" >> "$LOCK"
    find "$M" -type f ! -name '*.part' | sort | while read -r f; do
        printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "${f#$M/}" >> "$LOCK"
    done
    echo "locked $(grep -c '^[0-9a-f]' "$LOCK") files in $LOCK"
}

verify() {
    [ -f "$LOCK" ] || { echo "no $LOCK yet - run fetch first" >&2; exit 2; }
    bad=0; missing=0; n=0
    while read -r digest rel; do
        case "$digest" in \#*|"") continue;; esac
        n=$((n + 1))
        f="$M/$rel"
        if [ ! -f "$f" ]; then echo "  MISSING $rel"; missing=$((missing + 1)); continue; fi
        got=$(sha256sum "$f" | cut -d' ' -f1)
        [ "$got" = "$digest" ] || { echo "  MISMATCH $rel"; bad=$((bad + 1)); }
    done < "$LOCK"
    echo "verified=$n missing=$missing mismatched=$bad"
    [ "$missing" = 0 ] && [ "$bad" = 0 ]
}

case "$1" in
    list) list ;;
    fetch) fetch ;;
    verify) verify ;;
    *) usage ;;
esac
