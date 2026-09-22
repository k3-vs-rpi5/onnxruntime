# Scenario S7: a perception stage beside the VLM service

The first scenario executed under `SCENARIO-VALIDATION-20260922.md` (class **C5**: a perception
stage sharing the board with a service). It is the deployment shape an embodied or analytics stack
actually has - detection running while a VLM answers questions about images - and it has **no
published row**, so its acceptance basis is the interference matrix: each side alone, then both
together, with the AI-cluster evidence attached to both.

Evidence: `/root/k3-evidence/onnxruntime/scenario-sense-language-20260922/`. Instrument:
`tools/board-scenario-sense-language.sh` (server lifecycle, clean-board check, in-flight TCM
sampling, and a detector sized to finish *inside* the client window so its average is a contention
number rather than a blend).

## 1. The two sides, alone

| side | configuration | measurement |
|---|---|---|
| VLM service | vendor `llama-server`, `fastvlm-0.5B`, encoder 8 threads, text `-t 8`, 4 slots, image `frame-a.png`, 4 simultaneous requests x 64 tokens | first round (cold image) **68.65 t/s**; cached-image rounds **84.73-86.37 t/s** (median ~85.8) |
| detector | `yolov8n.q`, one session, 2 threads | **23.22 ms** per inference (three independent runs: 23.23 / 23.16 / 23.22) |

(The same service at 4 x 32 tokens serves 47.91 t/s - the round-7 number 47.88 reproduced to 0.06 %.
Token count is part of the statistic; the two are not interchangeable.)

## 2. Both together

Detector at 2 threads started 3 s before the client and sized to run out inside the client's window;
eight client rounds of 4 x 64 tokens:

| quantity | alone | together | change |
|---|---|---|---|
| VLM, first round (cold image) | 68.65 t/s | **49.60 t/s** | **-27.7 %** |
| VLM, cached-image rounds | 84.73-86.37 (median ~85.8) | 56.31-59.89 (median ~58.4) | **-31.9 %** |
| VLM, the round after the detector exits | - | **85.56 t/s** | back to the alone band |
| detector | 23.22 ms | **49.91 ms** | **+115 %** |

The last row of the service column is the internal control that makes the rest trustworthy: the
detector finished just before the eighth round, and that round landed at 85.56 t/s - inside the
alone band, to the digit. The service is therefore not slow because of drift, warmth or a bad
server; it is slow exactly while the detector is running.

## 3. What it means, and what it does not

* **the board is not free capacity.** Eight AI-core TCM blocks are the whole resource; the VLM
  service's encoder and text model already use all eight, so a perception stage cannot ride along
  at its own rate. Sharing costs the service ~30 % of its served rate **and** the detector ~2.2x its
  latency - both sides pay, and the split is not zero-sum;
* **2 threads is not "small".** The detector asked for two blocks of eight; the penalty is
  ~30 % of the service. Any deployment that wants a perception stage beside a VLM service should
  either accept that price or move the perception stage off the AI cluster entirely - which the
  measurements forbid for int8 vision graphs (the X100 is 97x slower, `YOLO-OPTIMISATION-20260921.md`
  §4);
* **what is not measured**: a detector at 1 thread, a service at encoder 4 + text 4 with the detector
  taking 4, or the same scenario with the Qwen3.5-0.8B service (whose per-request cost is larger).
  The scheme's L4 leaves those as named cells rather than assumptions, and the vendor's ASR solution
  (`-np 4`, encoder pinned to two cores) remains the only official allocation this project has ever
  tested and rejected as a rule for VLM service shapes.

## 4. The claim, in the form §5 of the scheme requires

*In scenario S7 (C5, detector 1x2 beside a 4-slot fastvlm service at encoder 8 / text 8), the
detector alone runs at 23.22 ms and the service at 68.65 / ~85.8 t/s (cold / cached image, 4 x 64
tokens); together the service drops to 49.60 / ~58.4 t/s (-27.7 % / -31.9 %) and the detector to
49.91 ms (+115 %); no published row exists for this composition, so the acceptance basis is the
measured interference itself, with the post-detector round returning to 85.56 t/s as the control.
The AI cluster was engaged (blocks 0-7 sampled busy, one block per compute thread) and the board was
left 8/8 free; evidence: `scenario-sense-language-20260922/`, per-file sha256 registry,
self-check clean.*
