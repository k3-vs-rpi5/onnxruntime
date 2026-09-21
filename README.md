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
  interference ladder. `METHOD-ALIGNMENT-20260921.md` collects the vendor's K3
  model-zoo data (local documentation byte-identical to upstream), compares the
  vendor's benchmark method with ours aspect by aspect, and fixes the acceptance
  protocol: judge an optimisation both against the published row and against a
  method that stands on its own, and never quote one without the other.
  `BASELINE3-20260921.md` closes the first int8-encoder attempt with numbers: the
  dynamic-quantisation route (`xslim --dynq`) is slower than the shipped fp16
  encoder (518.6 against 150.5 ms), a plain fp32 copy is 10x slower, and the
  `DENSE_ACCURACY_LEVEL` knob moves the dynq model only within +-7 %, so the
  static-QDQ-with-calibration path is the one left open. `BASELINE4-20260921.md`
  turns the vendor's own model-zoo repositories (`model-zoo-vision`, head of
  2026-09-21, and `model-zoo-vlm`) into validation material: seven new vision
  rows including their no-DFL detector (9.08 ms) and two VLM-class encoders,
  their VLM matrix script run on the board (all four functional checks pass; its
  own metrics gate fails because the streaming paths are not counted), and an
  independent confirmation that a vendor-published dynq encoder lands on the slow
  fp32 path (2.10 s at 224 sq on 4 AI cores). `OPTIMISATION-20260921.md` measures
  the deployment shape: batching pays 1.41-1.47x per frame on a CNN, concurrency
  inside one 8-thread session saturates at ~110 inf/s (extra streams only add
  latency, splitting the cores costs -36 %), and the two clusters tax each other
  by 10-25 % in both directions - with one invalid capture kept as such because
  two EP processes were fighting over the same eight TCM blocks.
  `OPTIMISATION2-20260921.md` measures the frame around the model with the
  vendor's own pipeline harness (pre- and post-processing together are 1-5 % of a
  frame, which closes OpenCL pre-processing on measurement rather than taste) and
  settles a method question with numbers: the published matrix command's `-x 1`
  under-prices a graph whose EP claims only part of its nodes by **2.32x**, with
  the named CPU-fallback nodes, the TCM samples and the provider's own trace
  agreeing three ways.
  `METHOD-ALIGNMENT-20260921.md` §7 closes the two rows that never reconciled. For
  the LLM decode rows, package version (including the vendor's own 0.1.1 release),
  command, model identity and the runtime banner are now field-identical to the
  published capture while the numbers read -1.8 % on prefill and -10.8 % on decode,
  leaving the OS image generation as the single remaining variable. For the VLM
  rows, the cell's statistic is undefined and the candidate readings differ by
  40 %; only the vision encoder's own inference time matches the row (-5.2 to
  -7.6 %), so the row is an order-of-magnitude reference and not an end-to-end
  target - acceptance rule R10.
  Section 3 of `OPTIMISATION2-20260921.md` turns the same method into a screening
  pair (`tools/board-coverage-census.sh` + `tools/ort-provider-census.py`): of ten
  models, seven have no CPU fallback at all, `pp_liteseg` puts 25.5 % of its frame
  on the CPU EP (13 node executions, agreeing with the independent two-run
  estimate of 26.8 %), `yolo26n` 8.2 % but insensitive to the pool width, and the
  SigLIP2 encoder 0.6 % across 209 node executions.
  `VLM-SERVICE-SHAPE-20260921.md` asks the same question of the VLM service and
  gets the opposite answer: simultaneous requests share one pass over the weights,
  so four concurrent decode streams give **2.91x** aggregate for -19 % per stream,
  the image path reaches 1.77x because the vision stage re-encodes every frame and
  contends for the same eight cores, and the encoder's width (one line in the
  model's `config.json`) is worth 14 % of aggregate at that load.
  Section 5 of that document tries the three ways to take the vision stage off the
  critical path and rejects all three with numbers: the X100 cluster runs that
  encoder 30x slower (4.56 s against 152 ms), staggering the arrivals redistributes
  latency without raising throughput, and narrowing the encoder leaves decode worse
  off. `SESSION-OPTIONS-20260921.md` sweeps the session-level knobs on three models:
  the vendor's graph-optimisation default is the right one (level 99 is within
  +-2.5 %), `-o 0` costs 18.2 % and the parallel executor 9.8 % on the fallback-heavy
  segmenter, and the memory-policy family (arena, pattern, spinning, env allocators)
  is flat to +-1 %.
  Section 7 adds the VLM category's second model, the vendor's Qwen3.5-0.8B
  package: every round-7 direction generalises with smaller magnitudes (text decode
  2.91x -> 1.88x at four simultaneous requests, image path 1.77x -> 1.34x, encoder
  width worth 52.6 ms of time to first token instead of 87.3 ms), and the published
  row for this model matches no candidate statistic at all - our encoder-only
  number is 25-29 % faster than the row while time to first token is 22-47 %
  slower - which confirms acceptance rule R10 on a second model. Two operational
  traps came with it: the Qwen3.5 family needs `--reasoning-budget 0 --reasoning
  off` or every answer is empty while tokens are counted, and the served
  `image prefill` stage does not move with the encoder width in either model.
  Section 8 decomposes that stage with instruments already in hand: the server's
  own execution-provider trace (`SPACEMIT_EP_DEBUG_PROFILE` inside the server) shows
  one ~195 ms encoder run per request, a text-only control prices a prompt token at
  6.6-7.1 ms, and the 224/384 resolution pair fits `stage ~= encoder + 4.7 ms per
  image token` (384²: 162 tokens, 969 ms; 224²: 67 tokens, 367 ms). The per-image
  cost is therefore set by the image's token count, not by the encoder - the
  encoder is 20 % of the stage - which makes the resolution variant the lever and
  the encoder's thread count nearly irrelevant to it.
  Section 9 holds the image fixed as a controlled variable and beats the published
  row on the one axis where a like-for-like comparison exists: the vision encoder at
  384² is **-30.7 %** (4-core column) and **-24.7 %** (8-core column) against it,
  with the vendor's documented dense-accuracy setting measured as neutral. Nine
  image-neutral runtime configurations (flash attention, batch threads, micro-batch,
  mmap, KV quantisation, and combinations) are all neutral to +-0.8 % or worse -
  KV `q8_0` costs 20 % of the image path - and the vendor's own core-count axis is
  already at its optimum: LLM 8 + vision 8 gives 15.40 t/s under four-way load
  against 9.71/9.87 when the language model is cut to four threads. Lower-resolution
  variants are recorded as measured and out of scope (an image change): 224² would be
  worth ~600 ms per frame, and the nine-question screening set answers 12/18 with the
  *same* failures at 224/384/768.
  Section 10 locates the last 25 % between our service prefill and the model's published
  *language* row: the media backend costs 1.3 % (measured by starting a server
  without it), the service form costs 13 % (the vendor's own `llama-bench`
  prefills at 160.73-162.07 t/s against the server's 141.35 t/s on the same
  weights), and the published row is a further 12.7-13.7 % above the vendor's own
  tool - the same ~10 % optimism already documented for the older LLM rows, now
  measured on a prefill row. None of it is a runtime setting on this board; the
  reproducible margin is the encoder's (-24.7 % at eight cores).
  Nothing under `src/onnxruntime` has been modified.

## Workspace entry

```xml
<project name="onnxruntime.git" path="projects/onnxruntime" />
```
