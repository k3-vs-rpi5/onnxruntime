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
