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
  `YOLO-OPTIMISATION-20260921.md` collects the family's map, including three measurements
  the earlier rounds had not made: the provider's own options (eight threads on one
  stream is optimal for all six detectors; core affinity is worth +3.1 % only at four
  threads, which are 49 % slower), the eight cores as threads x streams (splitting
  costs 8-53 % of aggregate throughput), and the X100 as a co-processor (892.8 ms for
  `yolov8n` against 9.17 ms on the AI cluster, 97x). Sustained load does not decay
  (105.4 inf/s flat over three minutes). The open question is the fitted 4.2 ms per
  inference that does not scale with cores - 45 % of the frame - with the runtime's
  own `/dev/tcm_sync_mem` fallback as one candidate, priced but untested.
  Section 7 of that document answers the harder question - whether the *hardware* is
  at its limit - and the answer is no: against the vendor's own `cpufp` peak for the
  int8 matrix unit (511.53 GOPS per core, 2.046 TOPS on four cores), a YOLOv8n
  inference extracts **912 GOPS = 24.8 %** of the eight-core figure at the 1.8 GHz
  the board actually runs (the AI cores are at their maximum clock; the X100 is at
  2.0 of 2.15 GHz). The loss decomposes into ~43 % per-core kernel efficiency and
  60 % parallel scaling, and the workload is compute-bound - its weight streaming is
  ~3.1 GB/s against a 26 GiB/s roof. Configuration is exhausted; the fundamentals are
  not, and the missing factor lives inside the closed provider.
  With the K3 A100's dense int8 specification (~15 TOPS, quoted by the operator;
  the local documentation only carries K1's measured 511.53 GOPS/core) the same
  inference extracts **6.1 %**, and the provider's own per-operator trace says why:
  **half of the executed work is not matrix work at all** (activations 24.7 %, concat
  11.3 %, transpose 5.4 %, quantise 3.5 %, shape 2.8 %), and the conv half runs at
  ~12 % of the matrix peak. Neither roof is saturated - 6-25 % of compute, 11 % of
  memory - so the section also retracts its earlier "compute-bound" wording: the
  limiter is how the provider executes this graph.
  Section 8 extends that to the whole family with one provider trace per model: **40-50 % of
  the executed time is not int8 matrix work in every YOLO model** (40.2 % on `yolov8s`,
  49.3-50.2 % on the rest), and **no single operator carries more than ~6 %** - the head's
  Softmax is 5.6 % of `yolov8n`'s thread time, against 512 convolution, 544 activation, 136
  concatenation and 48 transpose calls per inference. Each architecture has its own
  non-matrix signature (segmentation head 15 % in Transpose, `yolo12n` 31 % in activations,
  DFL head ~12 % in Softmax/Concat/Sigmoid, no-DFL head 13 % in ReduceSum and transposes).
  Pricing for the record, both out of contract: removing the non-matrix half would be worth
  ~2x, removing only the segmentation head's transposes up to 15 % on that model.
  `MODEL-ALTERNATIVES-20260922.md` opens the operator-directed substitution track -
  replacing what the board is weak at, domain by domain, in a separate scope from the
  frozen models. It records the two measured weaknesses (`pp_liteseg` 25.5 % of its
  frame on the CPU EP, `yolo26n` 8.2 %), maps the vendor's own 42-family catalogue onto
  the common vision domains, names the same-domain alternatives already measured
  EP-complete (`yolov8n-seg` 12.85 ms against the published 12.75; `yolov8n`/`s`,
  `yolo12n`), and lists the empty domains by priority with a HEAD-checked menu: pose
  3.80 MB, depth 10.46 MB, OCR det+rec 2.03 MB, face recognition 1.07 MB - the four
  highest-priority candidates together are 16.3 MB, so acquisition is not the obstacle.
  It also records an evidence-integrity catch: the file filed as `rfdetr-nano.fp16.onnx`
  is a 434-byte archive error page, not a model.
  `OPERATOR-CONVERTER-20260922.md` answers the next question - can we rewrite the graph
  so the provider claims all of it - with the fallback inventories measured here:
  depth 18.4 % (`Resize` 6.34 ms, `Cast` 2.14, `Pow` 2.00, `Exp` 0.11 - every one but
  `Exp` is **in the provider's supported list**, so these are capability-check failures,
  not missing operators), `pp_liteseg` 25.5 % and `yolo26n` 8.2 %. The vendor's own
  `onnxslim` was run on the depth model and changed **nothing** (op counts and the 15 CPU
  node executions identical), so a capability-driven converter is genuinely missing
  rather than already provided. The document specifies the rule library in order of
  measured prize, the numeric gate that must precede any speed claim, and picks
  `yolo26n-depth` as the first target since closing its fallback would take the frame
  from 67.62 ms toward ~55 ms.
  Section 5 of that document is the operator-directed requirement for the conversion
  logic - **query first, then act** - with evidence in three grades (documented
  constraint, cross-model claimed/refused evidence, vendor example code) and a probe
  permitted only to confirm a derived rule, never to fish. Its evidence registry is the
  result of that discipline: `Resize` nearest+asymmetric is claimed everywhere;
  linear+half_pixel is claimed **inside a QDQ graph** (`pp_liteseg`, 7/7, verifiable in
  the provider's own subgraph dump) but refused as a standalone fp32/fp16 node; and the
  natural Gather-based replacement **segfaults the provider** while every one of its
  primitives passes in isolation - a robustness finding in its own right. Four candidate
  rules were killed by evidence rather than shipped as guesses; the surviving line of
  query (a Q/D/Q-wrapped probe) is named, along with its 6.34 ms prize.
  The probe answered something better (section 6): the discriminator is **context**, not
  the operator - a `half_pixel + linear` Resize alone is 100 % CPU-EP, while the same
  node **after a Conv** is claimed with zero CPU nodes. That explains `pp_liteseg`
  (seven resizes inside large claimed regions) and the depth model (a head cut into
  fragments by an unsupported `Exp`). Removing that `Exp` (`Exp -> Pow(E, x)`, numeric
  gate 0.03 fp16 ulp) did exactly what the mechanism predicted - CPU node executions
  15 -> 8, CPU share 18.4 % -> 8.0 % - and made the frame **16 % slower**
  (64.52/64.59 ms against 74.93/74.83 ms), so the rule is rejected on net effect:
  "fewer CPU nodes" is not the same claim as "faster". The next query is named and
  priced: why the provider's execution of the reconnected region costs ~16 ms more than
  the CPU fragment it replaced, with a one-Conv-plus-one-Resize probe behind it.
  `VENDOR-QUERY-20260922.md` (Chinese, for sending) is that question written as a
  support report: three reproducible issues - a supported operator declined when it
  stands alone but claimed next to a convolution; a reconnected region that the provider
  executes 16 % slower than the CPU fragment it replaced; and a Gather-based graph that
  **segfaults** the provider instead of being declined - each with its probe, its numbers
  and the specific questions asked of the vendor. The minimal reproductions travel in the
  evidence directory (`model-alternatives/converter-r1/vendor-query-probes/`).
  Section 8 of the alternatives document closes the first acceptance round on the four
  approved candidates: `yolov8n-pose` is **EP-complete at 11.195 ms against the published
  11.16 ms - the published vision row reproduced to 0.3 % for the third time**; the OCR
  detector (35.99 ms) and recogniser (2.358 ms) and the face embedder (3.137 ms) are
  EP-complete with new baselines established, and the recogniser matters because it is a
  **vendor-shipped dynamic-quantisation model that the provider claims in full** - our own
  `xslim`-produced dynq was refused, so the difference is who built the artefact. TCM was
  sampled inside a 400-inference pose run: `available_blocks=0/8`. The depth candidate is
  accepted with a measured defect (64.84 ms, 18.4 % of the frame on the CPU side, four
  rewrite families tried, none carried as pending).
  Sections 9 and 10 finish the catalogue and the OCR pipeline. Eleven more candidates:
  **three more published rows reproduced** (`mobilenet_v3_small` 2.719 against 2.75 ms,
  `mobilenet_v3_large` 4.529 against 4.52, `efficientnet_v2_s` 13.860 against 13.53 - six
  independent reproductions of the vision tables now, against zero for the LLM decode
  rows), eight of eleven EP-complete, and three shapes the provider declines: an
  open-vocabulary head (`yoloe` 37.9 %), a many-op decoder (`mobilesam_decoder`, 270 node
  executions per frame, 20.3 %) and a top-k feature extractor (`superpoint`, 13.8 %). The
  round also **qualifies round 3's dynq verdict**: the vendor's own `.dynq` recogniser is
  EP-complete at 2.340 ms, so the technique is fine and only *our* xslim-produced dynq of a
  vision transformer was refused - "who built the artefact" was the variable. Finally the
  OCR pipeline: detector 35.989 ms, recogniser 2.340 ms, and run together 38.420 / 6.095 ms
  - a frame costs 38.33 ms serialised against 38.42 ms overlapped, so the two stages should
  be **serialised on the same eight cores**, not given one each.
  Nothing under `src/onnxruntime` has been modified.

## Workspace entry

```xml
<project name="onnxruntime.git" path="projects/onnxruntime" />
```
