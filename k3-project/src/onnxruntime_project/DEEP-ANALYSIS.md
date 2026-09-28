# ONNX Runtime on K3: deep analysis and optimisation directions

Grounding: the llama.cpp campaign's measured rules for this board's AI cluster (the A100 experience),
the vendor's own K3 numbers in `.local/docs-ai/en/compute_stack/ai_compute_stack/modelzoo.md`, two
real `SPACEMIT_EP_DEBUG_PROFILE` traces taken from the board on 2026-09-17 (now under
`/root/k3-evidence/onnxruntime/planning-20260920/`), the sealed 2.0.6 package contents, and the
monitor's placement result. Planning only - no source and no provider behaviour is changed.

## 1. The A100 rules that transfer

| measured on this board | number | what it implies for the ONNX stack |
|---|---|---|
| two doors to the AI cluster: an IME2-enabled build and submission through a vendor runtime | IME2 needs `-march=..._xsmtvdotii` / `-mcpu=spacemit-a100`; the sanctioned runtimes are the llama.cpp backend, Spine-Runtime and this provider | the ONNX EP is one of the three doors; there is no fourth |
| Linux-side instruments do not see AI-core work | `perf` counted 1.6 M cycles on cpu8-15 against 773 M on cpu0 for a run that produced 124 t/s; `/proc/stat` credited cpu8/cpu15 with 0 user jiffies | A100 claims need the runtime's own registration plus `spacemit-tcm-smi` occupancy naming the pid - never per-CPU counters |
| AI cluster memory roof | 25.7-26.2 GiB/s read (8 cores); X100 18.9 GiB/s; the two paths largely add (~43 GiB/s) | vision encoders that stream weights land on the same roof; a two-cluster split is a design question, not a free win |
| TCM is 8 blocks of 384 KiB, one affinity mask per block pair | `spacemit-tcm-smi`: 8/8 blocks, cpu-mask 0x300/0xc00/... | TCM serves per-core staging, not bulk: the campaign measured 7 MiB/s pushing into it from the X100 |
| each AI thread must register with the driver | 252 us per `/proc/set_ai_thread` write, 56 us affinity, 2 us TCM claim | any runtime that creates threads per graph pays this per graph - 1960 threads in one 128-token run |
| device thread creation is not free | disposable pool: ~1.1 ms of worker setup per graph; durable process-wide pool: +2.7..+5.0 % and registrations 1040-1960 -> 8 | the same question applies to the EP: how many threads does it create, and when |
| the barrier table is process-global and indexed by thread index | two live pools collide: 8.28 -> 2.91 t/s, single waits up to 12 ms | two runtimes in one process (e.g. EP vision + llama.cpp LLM) share the driver; thread sets and their synchronisation are a first-class design constraint |
| decode is DRAM-bound, prefill is compute-bound | 1.47 GB of weights per token at ~24.5 GB/s = ~60 ms; batching pays until ~13.5 ms/row on a 30B | the same split decides where an ONNX model's time goes: CNN/ViT inference at batch 1 - compute-bound; LLM decode - bandwidth-bound |
| MoE verification reads disjoint expert sets | a 2-row verify costs 1.54x a 1-row decode | Qwen3-VL-30B-A3B (MoE) pays this on every decode step, not only under speculation |
| int8/6-bit operands pay where bytes are the constraint | compact q6_K: +1.5 % TG; the vendor's own vision tables: ViT-B int8 23.76 ms against fp16 48.75 ms at 8 cores | quantisation strategy is workload-dependent: ~2x for compute-bound vision, ~1.5 % for bandwidth-bound decode |

## 2. The ONNX provider in those terms

`libspacemit_ep.so` is the closed half of the stack; the package documents the whole allowed surface:

* registration through `Ort::SessionOptionsSpaceMITEnvInit`, and no source in the fork;
* thread shape: `SPACEMIT_EP_INTRA_THREAD_NUM` (documented as launching **2x** that many compute
  threads), `..._INTRA_THREAD_AFFINITY`, `..._INTER_THREAD_NUM`, `..._USE_GLOBAL_INTRA_THREAD` -
  i.e. the EP can be told which cores it may use, which is how the vendor's own VLM benchmark splits
  8 LLM cores from 4 or 8 vision cores;
* observability: `SPACEMIT_EP_DEBUG_PROFILE` (Chrome trace), `SPACEMIT_EP_DUMP_SUBGRAPHS`,
  `SPACEMIT_EP_DUMP_TENSORS`, plus `SPACEMIT_EP_DENSE_ACCURACY_LEVEL` and the op filters;
* extension: the published plugin API (`AddOperator` / `AddDispatch`), shipped with LayerNorm, GELU
  and QGELU examples - precisely the transformer-vision operators the EP's documented op list
  (Conv, ConvTranspose, Gemm, MatMul, the QDQ family, pooling) does not name.

## 3. Workload taxonomy with the vendor's K3 numbers

All from `modelzoo.md` (spacemit-ort 2.0.6, Bianbu 4.0rc3, `onnxruntime_perf_test -e spacemit`,
1/2/4/8 AI cores), plus the workspace's own trace and monitor numbers:

| family | example | K3 latency | shape of the work |
|---|---|---|---|
| small CNN | ResNet18 int8 224² | 8.00 / 4.74 / 2.90 / 2.07 ms | compute-bound, batch 1, tiny weights |
| mid CNN | ResNet50 int8 / fp16 | 21.00..5.47 / 37.64..11.36 ms | compute-bound; int8 ~1.7-2x fp16 |
| detector / segmenter / pose | YOLOv8n / s / m int8 640² | 9.51 / 16.65 / 32.31 ms (8 cores) | Conv-heavy, latency-sensitive, robotics workhorse |
| detection + mask | YOLOv8n-seg int8 | 12.75 ms | as above plus a mask head |
| vision transformer | ViT-B/16 int8 / fp16 224² | **23.76** / 48.75 ms | attention + LayerNorm + GELU: ops the EP doc does not list |
| VLM, end to end | fastvlm-0.5B 512² | 256.47 (LLM8+Vision4) / **164.50** (LLM8+Vision8) ms | encoder + prefill + decode; two engines |
| VLM, large MoE | Qwen3-VL-30B-A3B 768² | 7928 / 4754 ms | encoder + MoE decode |
| ASR | Qwen3-ASR 0.6B | RTF 0.169 | streaming-ish, latency-bound |
| our own trace | squeezenet, 4,992 inferences | 41 operators; top 3 convs 34.7 %; Transpose 1.8 % | the whole graph runs in one provided subgraph |

Two facts from the workspace that sharpen this: the monitor measured **32.05 inf/s with the CPU EP on
the X100 against 81.95 with the provider on the AI cores, identical output** (31.2 vs 12.2 ms per
inference), and the trace's per-inference operator time sums to ~11.4 ms - the two agree, so the trace
is a usable instrument.

## 4. VLM: where the time is and what to do

The vendor's VLM benchmark is not "one model on one runtime": `llama-server --media-backend smt
--smt-config-dir ./` runs the **vision encoder through this ONNX EP** while the LLM decode runs in
llama.cpp, with `SPACEMIT_EP_DENSE_ACCURACY_LEVEL=1` and `-ctk f16 -ctv f16`. So the VLM stack on
this board is two vendor runtimes sharing eight AI cores, and every rule in section 1 about threads,
TCM and barrier tables applies to the pair, not to each engine alone.

From the fastvlm-0.5B row, the encoder is where the lever is: moving the vision side from 4 to 8
cores cuts 92 ms of the 256 ms budget, i.e. the encoder is on the critical path while the LLM side
keeps its 8 threads - **12 to 16 threads are asked of 8 physical AI cores**. The vendor also ships an
fp16 encoder (`fastvlm_vision.f16.onnx`, in an 0.8 GB package) rather than an int8 one, even though
int8 is ~2x faster on ViT - an accuracy/precision trade the `DENSE_ACCURACY_LEVEL` knob is meant to
manage.

Directions, each testable with the package's own instruments:

1. **Encoder op coverage.** Dump the subgraph claims for `fastvlm_vision.f16.onnx` (and for a ViT
   ONNX) and compare against the 41-operator picture we already have for squeezenet. Any LayerNorm,
   GELU, Softmax or attention node *not* claimed means a fallback and a data round-trip through the
   X100 - the single largest structural risk in the VLM path.
2. **Thread-budget matrix.** The vendor samples only two points (LLM8+Vision4, LLM8+Vision8). Sweep
   the vision/LLM split and measure total time to first answer, with the EP's affinity option placing
   the vision threads and llama.cpp's `-t` placing the decode threads. Include the documented "2x
   intra threads" behaviour: count threads rather than trust the option.
3. **Precision and accuracy level.** fp16 encoder against an int8/QDQ encoder, and
   `DENSE_ACCURACY_LEVEL` levels, judged by `SPACEMIT_EP_DUMP_TENSORS` equivalence against a
   reference plus the answer text - the same "numeric gate before speed claims" rule the kernel work
   used.
4. **Per-inference overheads.** A single-image call is one encoder graph plus many decode steps.
   The traces we have show 4,992 invocations per operator but only 624 `Session` entries, so the
   session-level events are sampled or truncated somewhere and must be understood before they are
   used as a per-inference cost. Count what the EP creates per session - threads, TCM claims,
   subgraph objects - because that fixed cost is what a 10-30 ms encoder budget feels.
5. **Cross-engine interference.** With two runtimes on eight cores, the interference is measurable
   the same way the X100-load ladder was: run the vision encoder alone, then under a decode load,
   and record both rates.
6. **Plugin path for uncovered ops.** The package's LayerNorm/GELU/QGELU/QDQ plugin examples are the
   sanctioned way to keep a transformer vision tower on the AI cluster. This is op-level work with a
   documented ABI, not a provider modification.

## 5. Embodied AI: the robotics-shaped workloads

The same model zoo is the robotics perception stack: YOLOv8 detection/segmentation/pose at 9.5-39.6
ms, ResNet-class backbones at 2-11 ms, ViT at 24-49 ms, and small VLMs at 164-800 ms. A control loop
of 30-100 Hz (3-33 ms) is therefore reachable for *perception* and not for *VLM policies*, which
forces a hierarchical design:

| loop | budget | what fits (measured) |
|---|---|---|
| safety / servo, 100 Hz | 10 ms | nothing heavy on the AI cluster; X100 control code, or a 2 ms ResNet18-class head |
| perception, 30 Hz | 33 ms | YOLOv8n (9.5) / n-seg (12.75) / yolov8s (16.65) with headroom for pre/post-processing |
| scene understanding, 10 Hz | 100 ms | yolov8m (32.3), ViT-B int8 (23.8) |
| language-conditioned policy, 4-6 Hz | 170-260 ms | fastvlm-0.5B (164.5) - and note that its 512² image is one camera frame, not a stream |
| large VLM policy | 5-8 s | Qwen3-VL-30B-A3B (4.75 s) - offline/planning, not a control loop |

Directions specific to this class:

1. **Two-cluster pipeline.** X100 for camera capture, resize/normalise, post-processing (NMS, masks)
   and control; AI cores for the network. The EP's affinity option makes the AI-core side explicit,
   and the campaign's interference ladder says a busy X100 costs AI-core bandwidth (14.70 -> 12.57
   t/s at 6.8 GB/s of X100 read load) - so the split has to be measured, not assumed free.
2. **Small-graph workloads are dominated by fixed costs.** A diffusion policy or an iterative
   denoiser runs tens of small graphs per action; at the campaign's measured ~1.1 ms per graph of
   device-thread setup (plus 8 registrations), the fixed cost is the same order as the compute.
   Measure sessions-per-action and per-session overhead before optimising kernels.
3. **Worst case, not mean.** Robotics acceptance is P99/P99.9 latency and jitter, and this board has
   a known jitter source (thread creation per graph, TCM handoff). The harness should record the
   distribution, not an average: the monitor's per-tick timeline plus the provider trace give both.
4. **Determinism.** Fixed thread counts, fixed affinity, int8 kernels: the same greedy-output
   discipline used for LLM text applies to a policy's action vector. Tensor dumps are the gate.
5. **Thermal and power envelope.** The AI cores run at a fixed 1.8 GHz userspace governor; a robot's
   sustained budget is set by the enclosure. Energy per inference (monitor package data) belongs in
   the acceptance table next to latency for any always-on perception loop.

## 6. Ranked optimisation directions (whole project)

| # | direction | hypothesis and expected value | how to measure | scope |
|---|---|---|---|---|
| D1 | EP thread/TCM/session census | the EP creates its threads once per session, not per inference; if it creates them per graph, the fix is a durable EP session (worth the campaign's measured 2.7-5.0 % on the LLM side, likely more for small graphs) | count threads of the server process, TCM occupancy via `spacemit-tcm-smi`, session entries in the trace | documented knobs only |
| D2 | Subgraph/op coverage audit (CNN, ViT/CLIP, VLM encoder) | fallbacks to the X100 are the largest hidden cost; expect several in transformer towers | `SPACEMIT_EP_DUMP_SUBGRAPHS` + profile trace per model | diagnostic |
| D3 | Precision / accuracy-level matrix | int8 is ~2x fp16 on ViT-class models; the VLM package ships fp16. **Measured 2026-09-21**: `xslim --dynq` is *slower* than fp16 (518.6 ms against 150.5 ms at 8 cores; a plain fp32 copy 1549 ms) and `DENSE_ACCURACY_LEVEL` moves it within +-7 %, so the dynamic route is closed; static QDQ with calibration is the open one | tensor-dump equivalence on real image input + answer text, then latency | uses `DENSE_ACCURACY_LEVEL`, op filters, xslim |
| D4 | VLM thread-split matrix | the vendor samples 2 of ≥6 useful points; the optimum likely moves with the LLM size | total time-to-answer over {Vision 2,4,6,8} x {LLM 8,6,4}, monitor packages | documented knobs |
| D5 | Plugin ops for vision transformers | LayerNorm/GELU/attention coverage keeps the tower on the AI cluster; samples exist | plugin build from the package, subgraph re-dump, tensor equivalence | published plugin API |
| D6 | Small-graph overhead reduction (diffusion policy, per-frame loops) | fixed cost per graph/session becomes the dominant term below ~10 ms of compute | trace session counts + per-session wall time, monitor timeline | harness + documented options |
| D7 | Two-cluster placement with measured interference | X100 work steals AI-core bandwidth; the split must be measured per workload | interference ladder (idle / 6.8 / 9.4 / 19.3 GB/s) as in the campaign | diagnostic |
| D8 | Real-time measurement methodology | mean throughput hides the jitter that matters for control loops | P50/P99/P99.9 latency and per-inference timeline from the monitor | harness |

## 7. Measurement plan and acceptance (contract-grade)

* Every provider number names the three identities: fork commit `d76541ad`, release tarball
  `bebcdfb7...`, and the deb pair actually loaded (`076d9e83...` / `d8fa3633...`).
* Baseline is the CPU EP on the X100 with explicit affinity, same model and thread count, plus the
  identical-output check.
* AI-cluster claims carry the runtime's registration and `spacemit-tcm-smi` occupancy naming the
  workload's pid, together with the throughput arithmetic the model's shape implies.
* Numeric gates precede speed claims: `DUMP_TENSORS` equivalence for any precision/coverage change,
  the campaign's byte-identical-text and bit-identical-perplexity discipline where generation is
  involved.
* Sealed evidence packages (payload verification + `SHA256SUMS`) and no report written as a side
  effect of a run.

## 8. Risks and open questions

* The provider's per-graph thread behaviour is unknown until D1 runs; the "2x intra threads" note
  suggests it is not simply `intra_thread_num`.
* Op coverage for attention/LayerNorm/GELU is unknown until D2 runs; the plugin path (D5) exists
  precisely because the coverage is incomplete.
* VLM accuracy at int8 encoder precision is unmeasured; the vendor's own benchmark sidesteps it with
  fp16 encoders and `DENSE_ACCURACY_LEVEL=1`.
* Which VLM is the acceptance target: the vendor zoo's fastvlm-0.5B package is already on the host
  (moved out of `/tmp` this round to `.local/models/vlm/fastvlm-mm-0.5b-q4_1.tar.gz`, sha256
  `495dcaca0b3c6b859495b48b5f736c3485cd821f282990504258b672de9435bf`; it contains
  `fastvlm_vision.f16.onnx` and `fastvlm-text-0.5B-Q4_1.gguf`), so it is the cheapest honest starting
  point; Qwen3-VL-30B-A3B is the hard one and matches the LLM work's existing 30B MoE knowledge.
* Robotics acceptance also needs a control-side definition of done (which loop, which jitter bound);
  that is a human decision, not a measurement.
