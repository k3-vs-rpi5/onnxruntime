# The multi-session rule on a second model, and two things the harness got wrong

This is item **G2** of `NEXT-ROUND-ANALYSIS-20260922.md`. Section 14.1 established the throughput
rule on `yolov8n` alone - one process, **N sessions of 8/N AI threads each**, +42 % aggregate over
one fat session, with the vendor's own application documenting the same shape ("eight parallel ONNX
Runtime sessions by default", work distributed round-robin). A rule from one small model is a
hypothesis; this round tests it on `yolov8s`, whose published rows are 16.65 ms single-stream and
**77.74 FPS** for their 12-stream application.

Evidence: `/root/k3-evidence/onnxruntime/multisession-20260922/` (75 files, per-file sha256 registry,
self-check 0 mismatched / 0 missing / 0 unlisted). Instrument: `tools/board-multisession-scaling.sh`
driving `ort-two-session-pipeline`'s `same` mode, the same harness section 14.1 used.

## 1. The rule survives, and the curve has the same shape

Medians of three rounds per shape (one row has two clean rounds; see §3), 10 iterations per round,
unprofiled, with `yolov8n` re-measured as the control:

| shape | `yolov8n` slowest / aggregate | vs published 134.29 | `yolov8s` slowest / aggregate | vs published 77.74 |
|---|---|---|---|---|
| 1 x 8 | 9.489 ms / **105.38 inf/s** | -21.5 % | 16.520 ms / **60.53 inf/s** | -22.1 % |
| 2 x 4 | 15.741 ms / 127.06 | -5.4 % | 29.784 ms / 67.15 | -13.6 % |
| 4 x 2 | 29.668 ms / 134.82 | **+0.4 %** | 51.658 ms / 77.43 | **-0.4 %** |
| **8 x 1** | 54.561 ms / **146.62** | **+9.2 %** | 92.875 ms / **86.14** | **+10.8 %** |

Three readings:

1. **the rule generalises.** `8x1` beats `1x8` by +39 % on `yolov8n` and **+42 % on `yolov8s`** - the
   same order as the +42 % that established it - and both models are **above the published
   application total** at that shape (+9.2 %, +10.8 %);
2. **the vendor's own shape is the published number.** `4x2`, the decomposition their application
   uses, lands within **0.4 %** of their published total on both models, from either side. That is
   the strongest agreement this project has with an official *application* number, and it says the
   comparison is now method-aligned in shape even though their number still includes decode, GPU
   pre-processing and the UI (the bound noted in `NEXT-ROUND-ANALYSIS-20260922.md` §3 G3);
3. **the fat session is the wrong shape for throughput on both models**, 21-22 % below the published
   total, while remaining the shape to pick for latency.

The control also draws the honest error bar: `yolov8n` at `8x1` measured **151.50 inf/s** in round 14
and **146.62** today (-3.2 %), with a different iteration count and with engagement runs interleaved
into this sweep. The ordering of the shapes is stable; the absolute value carries a few percent.

## 2. The AI-cluster claim, per row

Every one of the eight shape/model rows has an engagement run sampled while it was in flight:
`spacemit-tcm-smi` showing **blocks 0-7 busy** (98-136 sampled lines per run) with the workload's
thread ids in the PID column, and **17 threads** in the process. The samples are in the evidence
package next to the timing logs.

## 3. Two instrument defects, and one provider failure, all kept

**The sampler named the wrong process.** `setsid nohup ... &` may fork, so `$!` names the wrapper;
the first pass therefore reported zero busy TCM lines while the cluster was fully engaged - a false
negative that would have read as "the cluster is not used". The samplers now start the workload
directly.

**The TCM table's PID column can name a thread, not the process.** Measured directly: the harness
process was pid 3621948 and the eight busy blocks named 3621987, one of its thread ids; the
`perf_test` runs in the sibling package named the process itself. Both samplers now match the
workload by *any* of its thread ids, recomputed per sample. The same investigation corrected the
header comment of `ort-two-session-pipeline.cpp`: `same` mode takes the model path **twice**
(`pipeline <model> <model> same ...`), and the shorter form fails with `Load model same failed`.

**Provider session creation failed twice, at random.** `yolov8n 8x1 r3` (which stopped the first
sweep) and `yolov8s 4x2 r3` both failed *after two clean rounds of the identical shape and flags*
with `SpacemiT EP init: Provider options key/value cannot be empty`, and the identical command
succeeded on the next invocation. This is not a timing number and it is not being averaged away: the
logs are in the package, the affected rows report the median of the rounds that ran, and the
occurrence is now question 6 of `VENDOR-QUERY-20260922.md` with its reproduction.

## 4. What this changes in the plan

G2 is closed: the throughput rule may be applied beyond `yolov8n`, with the qualifier that it is
confirmed on two detector models and not yet on a stage that must run beside another (the VLM
encoder/decoder case is measured separately in `MODEL-ALTERNATIVES-20260922.md` §12-§14.2, where the
rule does *not* transfer). The "ahead of the published total" claim now rests on two models at
`8x1` (+9.2 %, +10.8 %) and on `4x2` agreeing with the official number to 0.4 %, which is what G3's
like-for-like question can build on. The next item in the plan is unchanged: **G4**, the VLM
service acceptance basis, which is a document rather than a measurement.
