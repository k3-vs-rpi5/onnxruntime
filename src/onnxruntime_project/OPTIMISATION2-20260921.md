# Sixth round: the frame around the model, and the session shape the published command fixes

Round 5 fixed the deployment shape (one session, eight threads, batch where the workload allows) and
left two questions open: what share of a *frame* is the model at all, and whether the published
matrix command measures the shape a deployment runs.

Instruments: the vendor's own pipeline harness - `model-zoo-vision` HEAD `45c3878` (2026-09-21),
built on the board from their sources, driven by their `tests/scheduled/run_infer_perf.sh`
(md5 `be545ad0...`, `vision_infer_benchmark` md5 `ba0959b0...`) - and the sealed 2.0.6
`onnxruntime_perf_test`. Raw output: `/root/k3-evidence/onnxruntime/baseline-20260921/`
(`session-shape.txt`, `modelzoo-vision/`). Identity control: the two models below are byte-identical
in the library's cache, in `/root/onnx-zoo2` and in the evidence copy (`pp_liteseg` `03a6bfe2...`,
`yolov8n_no_dfl` `56fdce64...`), so both harnesses measured the same bytes.

## 1. Where a frame's time goes: 95-96 % of it is the model

| model (their example config) | preprocess | model | postprocess | infer total | FPS | draw |
|---|---|---|---|---|---|---|
| `yolov8n_no_dfl` 640², real frame | 0.305 ms | **9.273 ms** | 0.025 ms | 9.625 ms | 103.9 | - |
| `pp_liteseg` 512x1024, real frame | 1.057 ms | **83.168 ms** | 3.421 ms | 87.684 ms | 11.4 | 5.293 ms |
| `yolov8n_no_dfl`, noise frame (control) | 0.308 ms | 9.177 ms | 0.024 ms | 9.536 ms | - | - |
| `pp_liteseg`, noise frame (control) | 0.952 ms | 83.984 ms | 3.405 ms | 88.377 ms | 11.3 | 5.213 ms |

(FPS is their harness's own `1000 / infer total`; checked against the one row whose raw log prints
both - 88.377 ms -> 11.3152.)

The noise control reproduces the real frame to +-0.9 %, so the split is a property of the
architecture and not of the input content. Reading:

* the model is **96.4 %** of a detection frame and **94.8 %** of a segmentation frame;
* the vendor's OpenCL pre-processing backend is **off by default** in their own build
  (`VISION_WITH_OPENCL` in their `CMakeLists.txt`; the board's single OpenCL platform is present and
  unused by this library). A backend that made pre-processing *free* would be worth **3.2 %** of a
  detector frame and **1.2 %** of a segmenter frame - less than the upload/download and sync it
  would add, and not worth a second board package set. Closed on measurement, no further board
  install;
* post-processing is 0.3 % (detector) and 3.9 % (segmenter). Their segmenter's `draw` step is
  5.29 ms, 6.0 % on top of the frame, and is display work: it must not be counted in service
  throughput, which is why their harness reports it separately and we keep it that way;
* cross-check of the model number by a second, independent harness: their library measures this
  segmenter at **83.17 ms** of model time, the sealed `onnxruntime_perf_test` at **85.95 ms** with
  the same eight AI threads. 3 % apart - and both far from what the published matrix command gives
  the same bytes, which is section 2.

## 2. The published matrix command pins ONNX Runtime's own pool at one thread

The published vision command is
`onnxruntime_perf_test ${model} -e spacemit -r 10 -x 1 -S 1 -s -c 1 -i "SPACEMIT_EP_INTRA_THREAD_NUM|${cores}" -I`
(quoted verbatim in `METHOD-ALIGNMENT-20260921.md` §2). It names the EP's width explicitly, but
fixes ONNX Runtime's own intra-op pool at one thread. Re-running the eight models this round selected
at `-x 1` against `-x 8`, everything else identical (EP width 8 in both):

| model | `-x 1` | `-x 8` | ratio |
|---|---|---|---|
| `yolov8n_no_dfl` | 9.183 ms | 9.268 ms | 0.99 |
| `yolo12n` | 22.652 ms | 22.674 ms | 1.00 |
| `yolo26n` | 14.422 ms | 14.531 ms | 0.99 |
| `pp_liteseg` | **199.471 ms** | **85.952 ms** | **2.32** |
| `det_10g_fixed` | 12.743 ms | 12.748 ms | 1.00 |
| `siglip2_vision_encoder_fp16_proj.dynq` | 2837.593 ms | 2794.568 ms | 1.02 |
| `image_encoder.fp16` | 44.472 ms | 44.705 ms | 0.99 |
| `vit_b_16.q` | 23.268 ms | 24.051 ms | 0.97 |

Seven of eight are flat; `pp_liteseg` moves by 2.32x. `-x` sizes ONNX Runtime's CPU pool and nothing
else in that command - and the AI side is pinned by the option they set - so the effect has to be
work that runs on the CPU EP.

### The named cause

ORT's per-node profile (`-p`) turns the attribution into a list. Per iteration this model executes
**13 nodes on `CPUExecutionProvider`**, and their op names and input shapes say what they are - the
pyramid-pooling block of PP-LiteSeg plus its re-entry points into the quantised graph:

| op | node | input | `-x 1` (pool 1) | `-x 8` (pool 8) |
|---|---|---|---|---|
| QuantizeLinear | `PPQ_Operation_391` | f32 1x128x64x128 | 49.67 ms | 7.05 ms |
| AveragePool | `AveragePool.0` | f32 1x128x128x256 | 25.30 ms | 4.19 ms |
| QuantizeLinear | `PPQ_Operation_455` | f32 1x256x32x64 | 24.75 ms | 3.37 ms |
| AveragePool | `AveragePool.1` | f32 1x256x64x128 | 12.39 ms | 1.94 ms |
| QuantizeLinear | `PPQ_Operation_535` | f32 1x512x16x32 | 12.15 ms | 1.80 ms |
| AveragePool | `AveragePool.2` | f32 1x512x32x64 | 5.83 ms | 1.21 ms |
| Transpose x3, DequantizeLinear x3 | EP boundary | int8 tensors | 7.10 ms | 3.82 ms |
| Cast | `Cast.0` | int64 1x512x1024 | 0.61 ms | 0.73 ms |
| **CPU total per iteration** | | | **137.79 ms** | **23.74 ms** |
| **EP span per iteration, same runs** | | | **78.13 ms** | **78.18 ms** |

The EP's own subgraph dump agrees on the covered set: across the four dumps (529 nodes against the
source graph's 457) every op type of the source graph is accounted for except **3 `AveragePool` and
1 `Cast`** - the same nodes the profile names on the CPU (`dump-ppliteseg/`).

### Three controls

*(a) The AI side is identically configured in the two runs being compared.* EP width 8 in both, and
`spacemit-tcm-smi` sampled every 200 ms through each: **8/8 blocks busy, all naming that run's pid**,
40 occupied samples per run (`modelzoo-vision/session-shape/session-shape2.txt`).

*(b) The EP does not move; only what surrounds it does - and two instruments see it the same way.*
The provider's own trace (`SPACEMIT_EP_DEBUG_PROFILE`, `cat=Session`, 4 subgraphs + init, 11 calls)
reads **787.5 ms against 785.4 ms** of subgraph execution, 0.3 % apart, while its outer
`model_session` span halves (2358.6 against 1109.9 ms) and the frame goes 204.19 -> 91.04 ms. The
ORT profile of the pair reads the EP span at **78.13 against 78.18 ms per iteration** while the CPU
side moves 137.79 -> 23.74 ms. The two levels differ by 9 % (71.6 ms per call against 78.1) because
they bound different intervals - ORT's node span includes the provider's dispatch - which is exactly
why the claim here is about movement, not about level.

*(c) Attribution and quantity agree.* The named CPU nodes differ by 114.05 ms per iteration; the
frame differs by 113.8 ms in the same pair, 113.5 ms in the unprofiled census row (199.471 ->
85.952). The instruments never share a run, which is why the check is worth stating.

### A second, separate effect in the same knob: the EP's width follows `-x` when its option is absent

The provider documents that if only `session_options.intra_thread_num` is set it is equivalent to
`SPACEMIT_EP_INTRA_THREAD_NUM = intra_thread_num`. Measured, with `spacemit-tcm-smi` as the ground
truth for the AI side (`pp_liteseg`, `-r 10`, no other option):

| command | AI cores (TCM blocks naming the pid) | ORT CPU pool | average |
|---|---|---|---|
| `-x 1`, no EP option | **1** | 1 | 545.8 ms (run of 12 samples; an earlier run of the same shape read 530.9 ms) |
| `-x 1`, `SPACEMIT_EP_INTRA_THREAD_NUM\|8` | 8 | 1 | 204.19 ms |
| `-x 8`, no EP option | 8 | 8 | 83.07 ms |
| `-x 8`, `SPACEMIT_EP_INTRA_THREAD_NUM\|8` | 8 | 8 | 85.95 ms |

So one character in this command line moves this model by **6.6x**, and the two effects decompose:
the AI width is worth 545.8 -> 204.2 ms and the CPU pool 204.2 -> 85.95 ms. The vendor's published
command sets the EP option explicitly, so the AI side of their rows is correct; but a deployment
that sets only the ORT thread count gets its AI parallelism silently from it, which is a trap worth
naming.

### A control that was wrong, kept as a note

Round 6 first read the profile pair *without* the EP thread option and recorded 530.9 ms at `-x 1`
against 88.6 ms at `-x 8`, then explained the difference as profiler overhead. With the option set,
the same pair reads 206.6 against 92.9 ms, and the TCM sample shows why the first pair read what it
read: those two runs were not the same AI configuration at all (1 block against 8). The correction
is this section's table; the class of mistake is what rule (a) - a TCM sample inside every run whose
AI participation is being compared - exists to catch. The earlier files are kept in the evidence
directory under `ortprof-noep-*` so the wrong pair is still visible as such.

### Why the corrected shape is the right one

The vendor's own library measures this segmenter at **83-85 ms** - the `-x 8` shape, not the `-x 1`
one. Their library and their published matrix therefore disagree by 2.3x on the same model bytes,
and the matrix side is the one that is wrong.

## 3. Screening instrument: which models hand work back to the CPU EP at all

Section 2 fixed one model, but "does this model have a CPU fallback" is a property of every model a
deployment wants to use, and the sensitivity to `-x` is not the same thing as the fallback itself
(`yolo26n` has one and does not care). So the method became a tool: `tools/board-coverage-census.sh`
profiles a list of models in the deployment shape (EP 8, pool 8) and
`tools/ort-provider-census.py` turns each profile into per-frame provider shares.

Ten models, one `-r 5` profile each, the CPU side divided by that run's own measured frame
(`coverage-census/`):

| model | measured frame | CPU EP | CPU share | CPU node executions/frame |
|---|---|---|---|---|
| `pp_liteseg.q` | 84.81 ms | 21.67 ms | **25.5 %** | 13 (QuantizeLinear 11.44, AveragePool 6.30, Cast 0.65, boundary 3.29) |
| `yolo26n.q` | 17.96 ms | 1.47 ms | 8.2 % | 16 (Tile, TopK, GatherElements, Div, Mod, ...) |
| `siglip2_vision_encoder_fp16_proj.dynq` | 2846.23 ms | 16.37 ms | 0.6 % | 209 (Reshape, Slice, Concat, Gather, Shape) |
| `yolov8n_no_dfl.q`, `yolo12n.q`, `det_10g_fixed.q`, `resnet50.q`, `vit_b_16.q`, `image_encoder.fp16`, `fastvlm_vision.f16` | 6.47 - 151.90 ms | 0.00 ms | 0 % | 0 |

The census and section 2's two-run method are independent, and they agree: 21.67 ms of a measured
84.81 ms frame is **25.5 %**, against **26.8 %** from the `-x 1`/`-x 8` pair (23.74 ms of 88.60 ms).

What it buys, in order of usefulness:

* **coverage is binary and cheap to know**: seven of ten are EP-complete, so they are insensitive to
  the pool width by construction; three are not, and each of the three is a different shape;
* **node count does not predict sensitivity.** `pp_liteseg` (13 nodes) moves 2.32x with the pool
  width because its nodes are parallel-friendly (pooling and quantisation over large activations);
  `yolo26n` (16 nodes, **8.2 %** of the frame) does not move at all (0.99 in the §2 census) because
  its nodes are small sequential decoders - Tile, TopK, GatherElements - which pool width cannot
  help; `siglip2` (209 nodes) is 0.6 %. So the census says *where to look* and how much is there, and
  the unprofiled `-x 1`/`-x 8` pair says whether sizing the pool fixes it;
* **the boundary is visible**: the EP inserts `Transpose`/`DequantizeLinear` nodes around each
  fallback region, and the tool reports them separately (3.29 ms/frame on `pp_liteseg`, 9.75 ms/frame
  on `siglip2`) - that share is a consequence of the fallback, not an operator the graph declares.

Instrument note, and it is the reason the tool needs the run logs next to the profiles: **the
provider's own node spans in this profile are not wall clock.** They sum to more than the frame -
186.26 ms against a measured 25.91 ms on `vit_b_16`, 265.36 against 151.90 on the fastvlm encoder -
because one subgraph span covers work the provider parallelises internally. The CPU EP's spans are
read as real time (a CPU node's span is wall clock on the executor's own thread), and the share is
therefore taken against the *measured* frame. The first version of this tool divided by the span sum
and would have reported `pp_liteseg` at 21.1 % instead of 25.5 % and the EP side as real time; the
run logs are what caught it.

## 4. What this round changes

Accepted, no code change anywhere:

1. **Session shape rule (deployment).** Where a model has nodes the EP does not claim, ONNX
   Runtime's intra-op pool must be sized to the deployment's width instead of left at the tool's
   one-thread default; and the EP's own width must be set explicitly rather than inherited. Worth
   2.32x on this model and nothing on the other seven, so it is cheap to apply everywhere and must
   be stated in every acceptance run.
2. **Pipeline share (budget).** Pre/post-processing is 1-5 % of a frame on this stack; optimisation
   time does not go there, and OpenCL/GPU pre-processing is closed on measurement.
3. **Published-row caveat.** The matrix command `-x 1` is a lower bound for any graph with a CPU
   fallback. Our earlier absolute rows for such graphs inherit that caveat; the ratios and the
   conclusions of rounds 1-5 are unaffected - the eight-model census shows the effect in one model,
   and no published-row comparison used that model.

Measured and **out of scope** by the workspace contract (the model file is a fixed input): at the
corrected shape the CPU EP still owns **23.74 ms of the 92.9 ms profiled frame (26 %)**, 12.22 ms of
it in the three `QuantizeLinear` re-entry nodes alone. A graph edit that let the EP claim the
pooling block would be worth at most that; it is recorded here as the number and is not carried as a
pending optimisation.

Rejected by measurement: `-x 1` as "the comparable shape" for fallback-bearing graphs; OpenCL
pre-processing as a lever; and this round's own first explanation, which the TCM sample refuted.
