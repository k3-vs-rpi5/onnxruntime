# YOLO: the optimisation map, and what would refill it

YOLO is this project's optimisation focus, so this document collects everything measured about the
family on the K3 and states where the pool of *verified* optimisations stands. It adds three
measurements that had not been made when the earlier rounds declared the runtime "exhausted": the
execution provider's own options, the second cluster as a co-processor, and sustained behaviour.

Models: the official zoo's `yolov8n` / `yolov8s` / `yolov8n-seg` (int8, 640²), and the newer
`model-zoo-vision` family `yolov8n_no_dfl` / `yolo12n` / `yolo26n`. Raw:
`/root/k3-evidence/onnxruntime/baseline-20260921/yolo-ep-options/`.

## 1. There is no headroom hidden in the official numbers

The published rows were reproduced in round 1 and again today (8 AI cores, vendor flags):

| model | published (8 cores) | ours | model | published (8 cores) | ours |
|---|---|---|---|---|---|
| `yolov8n` int8 | 9.51 ms | 9.54 ms | `yolo12n` int8 | 21.73 ms | 22.57 ms |
| `yolov8s` int8 | 16.65 ms | 16.60 ms | `yolov8n_no_dfl` | - | 9.17 ms |
| `yolov8n-seg` int8 | 12.75 ms | 12.85 ms | `yolo26n` | - | 14.37 ms |

Within 0.3-4 %, i.e. **the published YOLO rows are what this board does**, so there is no
"official is much faster" gap to close on this family - unlike the LLM and VLM rows, where the
published numbers sit 10-13 % above what the vendor's own tools produce here.

## 2. The provider's own options (never swept before this round)

The YOLO graphs are EP-complete (round 6 §3), so the provider is the only place a gain could live.
Six models, one stream unless noted:

| configuration | `yolov8n_no_dfl` | `yolov8n` | `yolov8s` | `yolov8n-seg` | `yolo12n` | `yolo26n` |
|---|---|---|---|---|---|---|
| 8 threads, default affinity | **9.17 ms** | **9.54** | **16.60** | **12.85** | **22.57** | **14.37** |
| 4 threads, default | 14.12 | - | - | - | - | - |
| 4 threads, affinity `8;9;10;11` (contiguous) | 14.10 | - | - | - | - | - |
| 4 threads, affinity `8;10;12;14` (one per TCM pair) | 13.70 | 13.37 | 23.72 | 19.24 | 33.35 | 20.55 |
| 4 threads, affinity `9;11;13;15` | 13.82 | - | - | - | - | - |
| 2 threads, affinity `8;12` | 24.35 | - | - | - | - | - |
| 1 thread | 44.30 | - | - | - | - | - |

Two findings:

* **affinity is real but small, and only matters where you would not want it**: at four threads,
  taking one core from each TCM pair (`8;10;12;14`) beats taking four neighbours (`8;9;10;11`) by
  **3.1 %** - the pairs visible in `spacemit-tcm-smi` are not equivalent, so cores 8/9 do share
  something. But four threads are 49 % slower than eight, so this is not a deployment lever;
* **eight threads on one stream is optimal for every model in the family**, which is also why the
  session-level sweep of round 8 found nothing: there is no session setting that changes a graph the
  provider claims entirely.

## 3. Eight cores as threads x streams: splitting always loses

The provider documents `SPACEMIT_EP_INTER_THREAD_NUM` as multi-stream execution inside one session
(`INTRA x INTER` total threads). Measured as aggregate throughput under simultaneous requests:

| allocation | `yolov8n_no_dfl` | `yolov8n` | `yolov8s` | `yolov8n-seg` | `yolo12n` | `yolo26n` |
|---|---|---|---|---|---|---|
| 8 threads, 1 stream | **109.0 inf/s** | **104.8** | **60.2** | **77.8** | **44.3** | **69.6** |
| 4 x 2 streams | 94.9 (-13 %) | 93.1 (-11 %) | 52.6 (-13 %) | 68.7 (-12 %) | 40.2 (-9 %) | 63.8 (-8 %) |
| 2 x 4 streams | 68.3 (-37 %) | 68.6 (-35 %) | 41.1 (-32 %) | 51.6 (-34 %) | 26.3 (-41 %) | 45.4 (-35 %) |
| 1 x 8 streams | 51.5 (-53 %) | - | - | - | - | - |

The same conclusion as round 5's ORT-level concurrency test, now at the provider level: a YOLO
inference does not share work between streams, so splitting the eight cores only adds latency.

## 4. The second cluster is not a co-processor for YOLO

Running the same models on the X100 through ONNX Runtime's CPU EP with its RVV kernels:

| model | 4 X100 cores | 8 X100 cores | 8 AI cores | ratio |
|---|---|---|---|---|
| `yolov8n_no_dfl` | 1411.7 ms | **892.8 ms** | 9.17 ms | **97x** |
| `yolov8n` int8 | 1402.7 ms | 892.8 ms | 9.54 ms | 94x |
| `yolov8s` int8 | 3152.6 ms | 1878.9 ms | 16.60 ms | 113x |

(Round 1 measured the same effect for two other int8 graphs: 45x for resnet50, 23x for ViT - the
QDQ graphs dequantise into fp32 paths the CPU EP has no fast kernel for.) So "run a second model on
the X100 while the first runs on the AI cluster" - the natural way to raise a multi-model pipeline's
throughput - is closed for YOLO by three orders of magnitude. The X100's only role in a YOLO
pipeline remains pre/post-processing, which the vendor's own harness prices at 0.33 ms of a 9.63 ms
frame (3.4 %).

## 5. Sustained load does not decay

3 x 6,000 inferences (~57 s each) of `yolov8n` int8 on eight AI cores: **105.48 / 105.48 / 105.34
inf/s**, p50 9.447 / 9.448 / 9.458 ms, temperatures 31 -> 37 °C. No thermal or clock decay, so a
service can be planned at the burst rate.

## 6. What is left, and its price

The measured non-scaling part of a YOLO inference is the one open question. Fitting `T(n) = s + p/n`
to the thread curve (`yolov8n_no_dfl`: 44.30 / 24.35 / 13.70 / 9.17 ms at 1 / 2 / 4 / 8 threads) gives
**s ~= 4.2 ms and p ~= 40.2 ms**, i.e. about **45 % of the eight-thread frame does not scale with
cores**. Two candidates, one testable and one not:

* the runtime's own log says, on every run, that it **cannot open `/dev/tcm_sync_mem` and falls back
  to heap-allocated barriers** (`alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2`). The device
  does not exist on this board (only `/dev/tcm` does), and the vendor's own code references it. If
  the shared-memory path is faster, part of the 4.2 ms could be recoverable - but providing a device
  node is a board-provisioning change, which needs human approval, and the size of the prize is
  untested, so it is recorded here as a question with a number attached rather than as a pending
  optimisation;
* the alternative is that the 4.2 ms is genuine non-parallel work (dispatch, weight staging, TCM
  fill) inside a closed provider. The llama.cpp campaign's own accounting on the same cluster shows
  ~5 ms of pair barriers per *token* there - but that is over ~253 operator boundaries, where a YOLO
  inference has **one** provider subgraph, so barriers are unlikely to be the whole 4.2 ms.

Everything else that could refill the pool requires a contract change, and each has a measured
price:

| lever | measured price | requires |
|---|---|---|
| batch-N export (the vendor ships `resnet50.b4`: 15.608 ms for four frames against 4 x 5.748 ms) | **1.41-1.47x per frame** | a re-exported YOLO (model change) |
| removing the CPU-side nodes on `yolo26n` (16 executions, 8.2 % of its frame) | up to **8.2 %** of that model's frame | a graph edit (model change) |
| kernel work inside the provider | unknown | the vendor |

Within the current contract - model fixed, runtime and deployment in scope - the YOLO family on this
board is at its measured optimum: every provider option, session option, core allocation, cluster
choice and sustained-load effect has been measured, and none of them moves it.

## 7. Is the *hardware* at its limit? No - a roofline check

Everything above is about configuration. It says nothing about whether the silicon is being used
well, and the honest answer to that question is **no**. The vendor publishes both the peak and the
instrument to check it.

### 7.1 The denominators, from the vendor's own micro-benchmark

`docs-ai/architecture/concept.md` reports the result of running `cpufp` on the AI core: the int8
matrix unit (`vmadot(s32,s8,s8)`) sustains **511.53 GOPS on one core** and **2.046 TOPS on four
cores** - linear scaling, and their own arithmetic confirms it (a 4x8x4 tile is 128 MACs; at 2 GHz
that is 256 GMAC/s = 512 GOPS, counting a MAC as two operations).

Measured on this board:

* the AI cores run at **1.8 GHz under load and 1.8 GHz is their maximum** (`cpuinfo_max_freq` on
  cpu8-15, userspace governor) - so no DVFS headroom exists; scaling the published per-core figure to
  the clock we actually have gives **460 GOPS per core, 3.68 TOPS for the eight-core cluster**;
* the X100 cores run at 2.0 GHz against a 2.15 GHz maximum - **7.5 % is being left on that cluster**,
  which matters only for the 0.33 ms of pre/post-processing a YOLO frame spends there;
* memory: the A100 cluster's measured read roof is ~26 GiB/s, and YOLOv8n's int8 weights are 3.54 MB.
  Even if all eight threads read the whole weight file per inference, that is 28 MB per inference, or
  **~3.1 GB/s at 109 inf/s - 11 % of the roof**.

### 7.2 What YOLOv8n actually extracts

YOLOv8n at 640² is ~8.7 GFLOPs (the published architecture figure, ~4.4 GMACs) per inference. Two
denominators are in play and both are worth carrying, because they bracket the answer:

* **3.68 TOPS** - the K1 A60 figure above, clock-scaled to this board's 1.8 GHz (the only *measured*
  number the local documentation contains);
* **~15 TOPS** - the K3 A100 cluster's dense int8 specification as the operator states it. Our local
  documentation does not carry a K3 peak, so this one is quoted, not measured here, and it is the
  number the rest of this section uses so that the conclusion does not depend on the smaller figure.

| measurement | achieved | against 3.68 TOPS | against ~15 TOPS |
|---|---|---|---|
| eight cores, 9.538 ms | **912 GOPS** | 24.8 % | **6.1 %** |
| one thread, 44.30 ms (no-DFL model) | 196 GOPS | 42.6 % | 10.5 % |

The two numbers together decompose the loss: **per-core kernel efficiency is ~43 %**, and **parallel
scaling is 4.83x on eight cores (60 %)**, and 0.43 x 0.60 = 26 % - the cluster figure, against the
smaller denominator.

**A correction, because the earlier draft of this section said "compute-bound" and that is the wrong
word.** Position on a roofline is not the same as saturation. Measured, this workload saturates
*neither* roof: 6-25 % of the matrix peak and 11 % of the memory roof. What is true is that its
arithmetic intensity (310 FLOP/byte) sits below a 15 TOPS machine balance (~536 FLOP/byte), so if a
kernel ever did reach peak, memory would become the limit - but at 3 GB/s of a 26 GiB/s roof it is
nowhere near that either. The limiter is neither silicon throughput nor bandwidth: it is how the
provider executes this graph.

### 7.2.1 Where the executed time actually goes

The provider's own trace (`SPACEMIT_EP_DEBUG_PROFILE` on a YOLOv8n run, 174 operator types) answers
that directly. Summed over the eight threads, per inference:

| family | ms per inference (thread-summed) | share | calls per inference |
|---|---|---|---|
| **conv / GEMM (the only int8 matrix work)** | 34.752 | **50.8 %** | 512 |
| activations and softmax | 16.926 | 24.7 % | 544 |
| Concat | 7.714 | 11.3 % | 136 |
| Transpose | 3.714 | 5.4 % | 48 |
| quantise / dequantise (PPQ) | 2.376 | 3.5 % | 8 |
| shape and data movement | 1.946 | 2.8 % | 96 |
| other | 0.987 | 1.4 % | 48 |

Two consequences that answer "how can 15 TOPS be a bottleneck?" - it is not, and the trace shows why:

1. **half of the executed work cannot use the matrix units at all.** Activations, softmax, concat,
   transpose, quantise/dequantise and shape ops are 49.2 % of thread time and contain essentially none
   of the model's 8.7 GFLOPs. A 15 TOPS int8 peak is irrelevant to them; they run on the vector and
   scalar paths;
2. **even the matrix half runs far from peak.** Attributing the whole 8.7 GFLOPs to the conv family
   gives at best 8.7 GFLOPs / (0.508 x 9.538 ms) = **~1.8 TOPS, about 12 % of 15 TOPS** - and that is
   an upper bound, since the conv time also contains its own data movement.

So the missing factor is spread across the graph: a matrix half that runs at ~12 % of peak and a
non-matrix half that is half the time. Both are properties of the provider's execution, and neither
is reachable by runtime configuration.

Caveats, stated because they bound the claim: (a) the published micro-benchmark is K1's A60 core at
2 GHz, and this is a K3 A100 at 1.8 GHz - the per-core figure is the same ISA family, not the same
silicon, so the denominator should eventually be re-measured here with the same tool; (b) not every
one of the 8.7 GFLOPs is executable as an int8 matrix op in this graph - the QDQ dequantisation, the
concat/reshape head and the final decode run as other ops - so the share of the 8.7 GFLOPs that the
IME units could in principle retire is below 100 %. Neither caveat closes a gap of this size.

### 7.3 What that means for the optimisation programme

So the two statements are both true and they are different statements:

* **configuration**: exhausted. Session options, provider options, core affinity, stream splitting,
  the second cluster, sustained load - measured, all at or worse than the current setting;
* **fundamentals**: not exhausted. The cluster delivers ~25 % of its int8 peak on this workload, with
  the loss sitting in per-core kernel efficiency and parallel scaling. That is where a further 2-4x
  would have to come from, and it is inside the closed provider or in how the model's graph is
  executed - the two places this project cannot reach without a contract change or vendor work.

The one measurement that was still missing to make that claim airtight was a per-board matrix
micro-benchmark on cpu8-15 through the documented runtime, to replace the scaled K1 figure with a
measured K3 denominator. **It was taken on 2026-09-22** (`AI-CLUSTER-PEAK-20260922.md`): a pure-matrix
QDQ probe reaches **2754 GOPS at eight threads** - 18.4 % of the quoted ~15 TOPS and 75 % of the
clock-scaled K1 figure - with one TCM block per compute thread and the workload's pid named in the
table, and one core alone reaches 893 GOPS, 1.94x the clock-scaled K1 per-core number. So the
denominator in the tables above is now measured here, and `yolov8n`'s 912 GOPS end-to-end is 33 % of
what the provider itself delivers on a pure matrix graph.

## 8. Full operator census across the family

Section 7.2.1 read one model. This is all six, each with its own provider trace (20 repetitions, 21
subgraph executions per trace, deployment shape), analysed by
`tools/ep-op-breakdown.py` - family, module and single-operator views. Raw: `yolo-traces/`.

| model | wall, instrumented | wall, clean | thread-summed node time | **matrix share** | **non-matrix share** | largest non-matrix family |
|---|---|---|---|---|---|---|
| `yolov8n` | 13.07 ms | 9.54 ms | 68.29 ms | 50.7 % | 49.3 % | activation/softmax 25 % |
| `yolov8s` | 21.13 ms | 16.60 ms | 122.99 ms | **59.8 %** | 40.2 % | activation/softmax 22 % |
| `yolov8n-seg` | 18.05 ms | 12.85 ms | 95.75 ms | 50.3 % | 49.7 % | **Transpose 15 %** |
| `yolov8n_no_dfl` | 13.07 ms | 9.17 ms | 68.51 ms | 49.8 % | 50.2 % | activation/softmax 16 % |
| `yolo12n` | 32.74 ms | 22.57 ms | 160.35 ms | 49.9 % | 50.1 % | **activation/softmax 31 %** |
| `yolo26n` | 21.20 ms | 14.37 ms | 93.66 ms | 55.8 % | 44.2 % | activation/softmax 20 % |

(The instrumented wall is 1.27-1.48x the clean one, so the shares are the output of this table and the
milliseconds are not - the same rule the project applies to every profiler.)

Three findings, and they are family-wide rather than model-specific:

1. **Between 40 % and 50 % of the executed time is not int8 matrix work** in every model of the
   family. The dense-int8 peak applies only to the other half, so no single TOPS number describes
   this workload: the smaller models sit at 50/50, and only the larger `yolov8s` shifts to 60/40
   because its larger convolutions amortise the per-operator overheads better;
2. **there is no hotspot to fix.** The most expensive single operator in `yolov8n` is the detection
   head's `Softmax` at 5.6 % of thread time; per inference the graph executes 512 convolutions, 544
   activation operators, 136 concatenations and 48 transposes. Optimising one kernel - however well -
   cannot move this family by more than a few percent; improvement has to be structural;
3. **each architecture has its own non-matrix signature**, which is what a graph-level discussion
   would start from: the segmentation head spends **15 %** of its time in `Transpose` (four times
   `yolov8n`'s 5 %), `yolo12n` spends **31 %** in activations, `yolov8n`'s DFL head is
   Softmax + Concat + Sigmoid (~12 %), and the no-DFL variant replaces exactly that with a
   `ReduceSum` and two transposes (13 % together) - the head is the head, whichever way it is
   written.

What that prices, for the record and out of contract: removing the non-matrix half entirely would be
worth about **2x** on this family, and a graph edit that removed only the segmentation head's
transposes would be worth up to **15 %** on that one model. Neither is available here - they are
model and graph changes - but the numbers now exist instead of an assumption.
