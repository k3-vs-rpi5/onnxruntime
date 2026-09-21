# Fifth round: deployment-shape optimisation - what actually pays on this EP

The model-level questions (D1/D2/D3/D6/D7) are answered; this round asks the *deployment* question
instead: given one EP and eight AI cores, how should a service be shaped? Four measurements, same
instrument as everywhere else (the sealed 2.0.6 `onnxruntime_perf_test`, `-r 10/20`, seed 1,
percentiles on), plus the campaign's `bw_read` for the X100 side. Raw output:
`/root/k3-evidence/onnxruntime/baseline-20260921/optimisation-*.txt`.

## 1. Batching pays, and it is the only lever that beats single-stream saturation

| model | per call | per frame | throughput |
|---|---|---|---|
| `resnet50.q.onnx` (batch 1) | 5.748 ms | 5.748 ms | 173.8 inf/s |
| `resnet50.b4.q.onnx` (batch 4) | 15.608 ms | **3.902 ms** | 64.0 calls/s = **256.2 frames/s** |

**1.47x per frame**, and it reproduces the vendor's published row for the same model (15.38 ms) to
within 1.5 %. The reason it works where concurrency does not: four rows share one pass of the weight
stream, while four sessions do not.

## 2. Concurrency inside one process saturates; extra streams only add latency

`-c` is the tool's "runs invoked simultaneously" knob (`-x 1` keeps ONNX Runtime's own intra-op pool at
one thread; the EP's threads do the work):

| model (8 AI threads) | 1 stream | 2 streams | 4 streams |
|---|---|---|---|
| `yolov8n_no_dfl` per-call | **9.08 ms** | 18.25 ms | 34.95 ms |
| `yolov8n_no_dfl` aggregate | 110.05 inf/s | 106.42 inf/s | 105.35 inf/s |
| `image_encoder.fp16` (MobileCLIP2-S3) per-call | 44.61 ms | 87.04 ms | - |
| `image_encoder.fp16` aggregate | 22.41 inf/s | 22.39 inf/s | - |

Throughput is flat from one stream on: the EP already fills the cluster. Splitting the cores is worse,
not better: two sessions with four threads each give **70.18** inf/s against 110.05 for one session
with eight (and `resnet50.b4`: 39.9 against 63.4 calls/s), because the per-core efficiency at four
threads is far below linear (yolov8n: 14.06 ms at 4 threads against 9.08 at 8).

**Deployment rule**: one EP session, eight AI threads, requests queued - it is simultaneously the
lowest-latency and the highest-throughput shape. Never split the eight cores into two EP sessions.

## 3. The two clusters tax each other in both directions

| direction | alone | while the other side runs |
|---|---|---|
| X100 4-thread read while the EP runs yolov8n on 8 AI cores | 13.96 GiB/s | **12.50 GiB/s (-10.5 %)**; an earlier run of the same pair measured 13.98 -> 11.08 GiB/s (-20.7 %) |
| yolov8n on 8 AI cores while the X100 reads 4 threads | 9.08 ms / 110.05 inf/s | **10.61 ms / 94.22 inf/s (-14.4 %)**; the forward ladder (round 2) spans -4.4 to -25.1 % depending on load |

So a pipeline that pre-processes on the X100 while the model runs on the AI cluster must budget
**10-25 % on both sides**, or serialise the two: measure, do not assume the two clusters are free of
each other.

## 4. An invalid capture, and the rule it produced

One capture of the above ran while a previous EP process was still in flight. The numbers were 5-20x
worse (yolov8n 42.48 ms against 9.08; encoder 488.0 against 44.6), and `spacemit-tcm-smi` showed
**0/8 blocks free**: the two processes were fighting over the same eight TCM blocks. The capture is
kept as `optimisation-round.txt` and **marked invalid** - it is not averaged into any number above -
and the finding is now a rule:

**one EP process at a time on this board.** Two EP processes (or an EP process plus llama.cpp's smt
vision stage) serialise on the TCM table, exactly as round 1 found for the two engines inside one
process; a measurement is only valid with `spacemit-tcm-smi` showing the expected occupancy and no
unrelated EP process running.

## 5. What this round changes

Accepted (deployment-level, no code change anywhere):

1. batch where the workload allows - 1.41-1.47x per frame for batch 4 on a CNN;
2. one session, eight threads, request queue - best latency *and* best throughput;
3. budget 10-25 % when both clusters work at once, or serialise them;
4. one EP process per board at a time.

Rejected by measurement: more streams (flat throughput, linear latency growth), splitting the cores
between sessions (-36 % throughput), and (from round 3) dynamic quantisation as a speed path.

Still unmeasured, and the reason it is not in this round: the *pipeline* share of a frame
(pre-processing, letterbox/NMS/DFL decode, drawing). Their own repository ships the harness for it
(`vision_infer_benchmark`, runs=50, warmup=5) and an OpenCL pre-processing backend to compare against
the CPU one, but running it needs their library built on the board, which needs `cmake`,
`opencv-dev`, `yaml-cpp-dev` and the `eigen/openblas-spacemit` dev packages installed there. That is a
one-line decision for the operator; the expected pay-off is bounded by the numbers above: for a
9.08 ms detector, every millisecond of pre/post-processing is 11 % of the frame.
