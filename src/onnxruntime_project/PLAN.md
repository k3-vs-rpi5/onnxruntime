# ONNX Runtime project: the plan, as of 2026-09-22

Scope, unchanged since the planning round: measure and evidence what the SpacemiT ONNX execution
provider (`libspacemit_ep.so`) does on this board the way the llama.cpp project measures its vendor
backend - host-side tooling around a sealed vendor component, **no modification of the component**.
Nothing under `src/onnxruntime` is touched by any round.

This file replaces the 2026-09-20 version, whose three open questions are now closed by measurement
(D1-D3 in `NEXT-ROUND-ANALYSIS-20260922.md` §7). It is the live plan; the round reports are the
record.

## What is in place

| item | state |
|---|---|
| pinned fork `src/onnxruntime` | `d76541ad26ca93d2d52e5d2db2aacaff862df5cf` (ref `upstream-1.24.2`), clean |
| sealed release package | `spacemit-ort.riscv64.2.0.6.tar.gz`, sha256 checked against `sources.lock.yaml`; the board's deb pair differs under the same version - a declared deviation, measured behaviourally equivalent (60 tensors bit-identical, timings within 0.41 %) |
| board-side identity | ort `2.0.6` (deb runtime `076d9e83…`, provider `d8fa3633…`; tarball `93bb7560…`/`dcc95030…`), llama.cpp `0.1.7~a0+4` build `5a23f07a4` |
| measurement protocol | `METHOD-ALIGNMENT-20260921.md`: the vendor's methods verbatim, our reproduction, R1-R12 and the calibrated bands |
| tooling | `tools/` - the matrix, census, session, service-shape, pipeline, probe and report instruments, each documented in `tools/README.md` |
| evidence | `/root/k3-evidence/onnxruntime/{planning-20260920,baseline-20260921,peak-20260922,multisession-20260922}/`, every package with a per-file sha256 registry and a self-check |
| public entry points | **not built yet** - see S1 below; today's rounds run through `hardware/k3-monitor/experiments/board.sh`, which the workspace contract allows for the steps between delivered runs |

## What the rounds established (all with sealed evidence)

* the published **vision** rows are acceptance-grade here (six independent reproductions inside
  -1.4..+2.5 %, one +4.9 % outlier) and the vendor's own `4x2` application shape lands within 0.4 %
  of their published multi-stream totals;
* the **VLM encoder** beats the published row at both widths it defines (-5.2 % / -8.8 %), and the
  published VLM cell has no named statistic at all (`VLM-ACCEPTANCE-BASIS-20260922.md`);
* the published **LLM decode** rows are ~10 % above what the vendor's own binaries produce on this
  board; the prefill rows reproduce;
* the AI cluster's own matrix ceiling is **2754 GOPS at eight threads** (18.4 % of the quoted peak),
  one TCM block per compute thread (`AI-CLUSTER-PEAK-20260922.md`);
* the throughput rule is **N sessions x 8/N threads** (`8x1` = 146.62 / 86.14 inf/s on `yolov8n` /
  `yolov8s`, +9-11 % over the published totals), the latency rule is one 8-thread session, and the
  multi-slot VLM service keeps one server with four slots (`MULTI-SESSION-SCALING-20260922.md`,
  `MODEL-ALTERNATIVES-20260922.md`);
* configuration is exhausted on YOLO: every provider option, session option, core allocation, stream
  split and cluster choice is measured and none moves it (`YOLO-OPTIMISATION-20260921.md`).

## The current plan

Executed from `NEXT-ROUND-ANALYSIS-20260922.md`, in its order:

| # | item | state |
|---|---|---|
| G1 | the AI cluster's own int8 ceiling | **done** - `AI-CLUSTER-PEAK-20260922.md` |
| G2 | the multi-session rule on a second model | **done** - `MULTI-SESSION-SCALING-20260922.md` |
| G4 | the VLM acceptance basis | **done** - `VLM-ACCEPTANCE-BASIS-20260922.md` |
| G3 | make the multi-stream comparison like-for-like | open - three priced options, the strongest needs operator approval (D4) |
| G5 | one composed embodied/perception scenario under the three deployment rules | next, pending the scenario choice (D6) |
| G6 | the LLM decode rows: pursue or demote | proposal: demote to context (D8) |
| G7 | `/dev/tcm_sync_mem` | kept open: a board-provisioning question with a number attached |

## S1. The structural deliverable still pending: the two public scripts

`project.yaml` declares `public_delivery: pending-automation` with `scripts/build.sh` and
`scripts/run.sh` as the intended entry points, and the workspace contract makes those the only
public interface of a project. They do not exist yet; every round so far has used the monitor's
ad-hoc `board.sh`, which is allowed between delivered runs but is not the durable path.

What they should be, following the llama.cpp project's shape:

* `scripts/build.sh` - cross-build this project's own artefacts (the harness binaries and the probe
  generator's outputs) with the workspace toolchain
  `spacemit-toolchain-linux-glibc-x86_64-v1.2.4`, sealed with digests;
* `scripts/run.sh` - deliver those artefacts, the selected models and the instruments to the board,
  execute the measurement matrix there, and pull the raw output back into a new evidence package;
* the models and the vendor package stay inputs, identified by digest, never rebuilt;
* `project.yaml` moves off `pending-automation` once the two scripts exist and one round has run
  through them, and that state change is reported rather than silent.

## Decisions outstanding

`NEXT-ROUND-ANALYSIS-20260922.md` §7 lists them. The first three are proposals to close the
2026-09-20 questions and need only a nod; **D4** (may the vendor's multi-stream application be
fetched and run on the board) and **D6** (which composed scenario) materially change what the next
rounds do.

## Out of scope (unchanged)

* the provider's internals: it ships stripped and its source is not published; work uses documented
  interfaces only (`OrtSessionOptionsSpaceMITEnvInit`, the provider options, the dump/profile
  switches, the published plugin API);
* changing the model, its quantisation, its weight layout, or the board's packages (model
  substitution is a separate track, `MODEL-ALTERNATIVES-20260922.md`, and needs its own approval);
* claiming any number without the run's own counters and a physics check, and any AI-cluster claim
  without `spacemit-tcm-smi` naming the workload.
