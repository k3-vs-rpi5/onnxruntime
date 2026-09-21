[中文](README.zh-CN.md)

# ONNX Runtime project (SpacemiT fork and vendor package)

The ONNX side of the AI work. Two artefacts belong together: the public fork that
carries the SpacemiT build files, and the closed provider package the board
actually runs.

## Pinned source

| field | value |
|---|---|
| upstream | `https://github.com/spacemit-com/onnxruntime.git` |
| ref | `upstream-1.24.2` |
| commit | `d76541ad26ca93d2d52e5d2db2aacaff862df5cf` |
| license | MIT |
| checkout | `src/onnxruntime` |
| vendor package | `spacemit-ort.riscv64.2.0.6.tar.gz`, package commit `5a5b5914` |

The provider itself is not published as source: the fork contains the toolchain
files and a provider test utility, and the release package ships
`libonnxruntime.so.1.24.2+spacemit.a1` with a stripped `libspacemit_ep.so.2.0.6`.
`sources.lock.yaml` therefore records both the fork's file digests and the sealed
package identity, and the documented entry points we are allowed to use.

## Status

- fork and package identity synced; provider internals stay closed by design.
- the A100 development rules are in the workspace contract, not here.
- automation (`src/onnxruntime_project`, `scripts/build.sh`, `scripts/run.sh`) is
  the next deliverable and is named as pending in `project.yaml`.
- planning round 2026-09-20: the release package is retrieved and identified (and
  its libraries differ from the board's deb pair under the same version number -
  `TECHNICAL-ANALYSIS.md` §2), the project layout and the measurement plan are in
  `src/onnxruntime_project/{PLAN.md,TECHNICAL-ANALYSIS.md}`, the ranked optimisation
  agenda built on the AI-cluster measurements is in `DEEP-ANALYSIS.md` (VLM and
  embodied-AI workloads in front). A first measurement round followed on 2026-09-21:
  `BASELINE-20260921.md` holds the latency/thread/TCM/coverage/operator/VLM tables,
  which reproduce the vendor's published K3 numbers within a couple of percent, and
  `BASELINE2-20260921.md` adds the numeric gate (60 intermediate tensors, bit-identical
  when nothing should change), the cold-start and per-new-frame costs, and the X100
  interference ladder.
  Nothing under `src/onnxruntime` has been modified.

## Workspace entry

```xml
<project name="onnxruntime.git" path="projects/onnxruntime" />
```
