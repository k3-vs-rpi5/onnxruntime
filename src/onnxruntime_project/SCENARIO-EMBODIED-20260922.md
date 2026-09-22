# Scenario S8: the embodied stack - detection and depth on their own, then with the language service

Second scenario under `SCENARIO-VALIDATION-20260922.md` (class C5), approved by the operator on
2026-09-22. The stack is the one an embodied deployment actually has: a detector (`yolov8n`) feeding
a depth model (`yolo26n-depth.fp16`) in one process, plus the 4-slot VLM language service
(`fastvlm-0.5B`). Evidence: `/root/k3-evidence/onnxruntime/scenario-embodied-20260922/`. Instrument:
`tools/board-scenario-embodied.sh`.

## 1. The perception stack alone (one process, 4 + 4 threads)

| shape | detector | depth | frame | throughput |
|---|---|---|---|---|
| serial | 14.03 ms | 113.36 ms | 127.39 ms | 7.85 fps |
| overlap | 15.15 ms (+8.0 %) | 113.06 ms | **113.06 ms** | **8.85 fps (+12.7 %)** |

TCM sampled inside both runs: blocks **0-7** busy. The split is not searched here (4+4 is the shape
the two-session harness takes without a global pool); what the pair shows is the same rule as the
detect->pose pair - overlapping two stages of unequal cost helps by the smaller stage's share, and
the depth model is 7.5x the detector.

## 2. The real interaction: perception beside the language service

Client: 2 simultaneous image requests x 32 tokens, two rounds; pipeline: overlap 4+4 running
throughout the client window.

| quantity | alone | with the other side running | change |
|---|---|---|---|
| VLM, round 1 (cold image) | 37.25 t/s | **22.56 t/s** | **-39.4 %** |
| VLM, round 2 (cached image) | 46.80 t/s | **24.96 t/s** | **-46.7 %** |
| pipeline frame | 113.06 ms (8.85 fps) | 132.22 ms (7.56 fps) | **-14.6 %** |
| detector stage | 15.15 ms | **50.86 ms** | **+236 %** |
| depth stage | 113.06 ms | 132.22 ms | +17.0 % |

Both sides pay, and they pay differently: the depth stage dwarfs the detector, so the *frame* only
loses 15 % while the detector stage loses 3.4x, and the language service - which is latency-shaped
here, two requests, not a saturated service - loses 40-47 % of its served rate. TCM sampled during
the overlap: blocks 0-7 busy, 148 samples.

## 3. Two runs that are wrong, kept on purpose

* `both-v2` reported "no interference" (VLM 36.87/47.73 against 37.25/46.80 alone, pipeline 8.856
  against 8.845 fps). It was a **sequence, not an overlap**: the pipeline ran to completion before the
  client rounds started, because the harness called the synchronous runner. The fix is in the
  instrument, and `both-v3` above is the corrected cell;
* `both-vlm-v1` crashed immediately (`stoi`) from an argument-shuffling bug in the same call site -
  also fixed, also kept.

Both are the reason the scheme's evidence contract says failed and invalid runs stay in the package
with their reason attached.

## 4. The claim, in the form the scheme requires

*In scenario S8 (C5, detector->depth one process 4+4 overlapping, beside a 2-request VLM service),
the perception stack alone runs at 113.06 ms/frame (8.85 fps) and the service at 37.25 / 46.80 t/s
(cold / cached); together the frame costs 132.22 ms (7.56 fps, -14.6 %), the detector stage +236 %,
and the service falls to 22.56 / 24.96 t/s (-39.4 % / -46.7 %). No published row exists for this
composition, so the acceptance basis is the interference matrix itself. The AI cluster was engaged
(blocks 0-7, one block per thread, named pids) and the board was left 8/8 free; evidence:
`scenario-embodied-20260922/`, per-file sha256 registry, self-check clean.*
