#!/bin/sh
# Cross-build this project's own artefacts for the K3 with the workspace toolchain.
#
# Binary-only delivery: the board receives the built harness and the instrument outputs; it never
# compiles anything. The vendor package and the models stay *inputs*, identified by digest, and are
# never rebuilt here (the provider ships stripped and its source is not published).
#
#   scripts/build.sh
#
# Env: ONNX_TOOLCHAIN, ONNX_PKG (sealed package root), ONNX_BUILD (output dir).
set -eu

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(CDPATH='' cd -- "$ROOT/../.." && pwd)
TOOLCHAIN=${ONNX_TOOLCHAIN:-$WORKSPACE/spacemit-toolchain-linux-glibc-x86_64-v1.2.4}
PKG=${ONNX_PKG:-$WORKSPACE/.local/k3-tools/riscv64}
BUILD=${ONNX_BUILD:-$ROOT/build}
FORK=$ROOT/src/onnxruntime
TOOLS=$ROOT/src/onnxruntime_project/tools

die() { echo "build: $*" >&2; exit 1; }

# ---------------------------------------------------------------- identity first
lock_commit=$(sed -n 's/^  commit: \(.*\)$/\1/p' "$ROOT/sources.lock.yaml" | head -1)
lock_sha=$(sed -n 's/^  package_sha256: \(.*\)$/\1/p' "$ROOT/sources.lock.yaml" | head -1)
[ -n "$lock_commit" ] && [ -n "$lock_sha" ] || die "sources.lock.yaml is missing the lock fields"

actual_commit=$(git -C "$FORK" rev-parse HEAD)
[ "$actual_commit" = "$lock_commit" ] || die "fork is at $actual_commit, lock says $lock_commit"
[ -z "$(git -C "$FORK" status --porcelain)" ] || die "the pinned fork has local changes"

tarball=$PKG/spacemit-ort.riscv64.2.0.6.tar.gz
[ -f "$tarball" ] || die "sealed package not found: $tarball"
actual_sha=$(sha256sum "$tarball" | cut -d' ' -f1)
[ "$actual_sha" = "$lock_sha" ] || die "package digest $actual_sha does not match the lock"

cxx=$TOOLCHAIN/bin/riscv64-unknown-linux-gnu-g++
[ -x "$cxx" ] || die "toolchain compiler not found: $cxx"
[ -f "$PKG/onnxrt/onnxruntime_cxx_api.h" ] || die "no ONNX Runtime headers under $PKG/onnxrt"

# ---------------------------------------------------------------- the artefacts
rm -rf "$BUILD"
mkdir -p "$BUILD/bin" "$BUILD/probes"

echo "== cross-compiling the two-session/scenario harness"
"$cxx" -O2 -std=c++17 "$TOOLS/ort-two-session-pipeline.cpp" -o "$BUILD/bin/pipeline" \
    -I"$PKG/onnxrt" -L"$PKG/onnxrt" -lonnxruntime -ldl -lpthread

echo "== cross-compiling the OCR chain (S12)"
"$cxx" -O2 -std=c++17 "$TOOLS/ort-ocr-chain.cpp" -o "$BUILD/bin/ocr-chain" \
    -I"$PKG/onnxrt" -L"$PKG/onnxrt" -lonnxruntime -ldl -lpthread

echo "== cross-compiling the image-embedding harness (S11)"
"$cxx" -O2 -std=c++17 "$TOOLS/ort-embed-image.cpp" -o "$BUILD/bin/embed-image" \
    -I"$PKG/onnxrt" -L"$PKG/onnxrt" -lonnxruntime -ldl -lpthread

echo "== cross-compiling the monitor (C, Rule 1 of TOOLING-RULES.md)"
"$cxx" -O2 -std=c++17 "$TOOLS/board-monitor.c" -o "$BUILD/bin/board-monitor"

echo "== generating the matrix probes (the instrument published in AI-CLUSTER-PEAK-20260922.md)"
for size in 1024 2048 4096; do
    python3 "$TOOLS/make-ime-probe.py" "$BUILD/probes/ime-probe-$size.onnx" \
        "$size" "$size" "$size"
done

echo "== sealing"
( cd "$BUILD" && find bin probes -type f | sort | xargs sha256sum > SHA256SUMS )
{
    echo "project: onnxruntime"
    echo "built_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "fork_commit: $actual_commit"
    echo "vendor_package: spacemit-ort.riscv64.2.0.6.tar.gz sha256 $actual_sha"
    echo "toolchain: $($cxx --version | head -1)"
    echo "compiler_sha256: $(sha256sum "$cxx" | cut -d' ' -f1)"
    echo "sysroot: $TOOLCHAIN/sysroot"
    echo "artefacts:"
    sed 's/^/  /' "$BUILD/SHA256SUMS"
} > "$BUILD/MANIFEST.txt"

cat "$BUILD/MANIFEST.txt"
