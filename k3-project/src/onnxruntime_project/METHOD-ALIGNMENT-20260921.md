# Vendor model-zoo data and the acceptance/test method we hold ourselves to

The rule this document makes explicit: **an optimisation is judged twice** - against the vendor's
published model-zoo numbers, and against a test method that is good enough to stand on its own
(identity, warm/cold, distribution, numeric gate, AI-cluster evidence, interference isolation).
Where the two disagree, the disagreement is recorded with numbers rather than smoothed over.

Sources: the vendor documentation shipped in this workspace, re-fetched from
`github.com/spacemit-com/docs-ai` on 2026-09-21 and **byte-identical** to our local copy
(`modelzoo.md`, `onnxruntime.md`, `onnxruntime_ep_ops.md`), the sealed 2.0.6 package, and our own
measurements from `BASELINE-20260921.md` / `BASELINE2-20260921.md`.

## 1. Provenance blocks the vendor states per section

| section | engine | OS | date |
|---|---|---|---|
| vision / audio (ONNX) | spacemit-ort **2.0.6** (release tarball) | Bianbu 4.0rc3 | 2026-07-27 |
| LLM | spacemit-llama.cpp **0.1.1** | Bianbu 4.0rc3 | 2026-05-26 |
| VLM / ASR | spacemit-llama.cpp **0.1.1** + spacemit-ort **2.0.3** | Bianbu 4.0rc3 | 2026-05-26 |

Our board today: Bianbu 4.0.6, kernel 6.18.3-generic-update3, ort **2.0.6** (deb) and
`llama.cpp-tools-spacemit 0.1.7~a0+4` (build `5a23f07a4`, the same fork commit we pin). The VLM/ASR
sections were measured with ort 2.0.3, one package behind what we run - a declared deviation.

## 2. The vendor's methods, verbatim

**Vision (the tables we reproduce):**

    ./bin/onnxruntime_perf_test ${model} -e spacemit -r 10 -x 1 -S 1 -s -c 1 \
        -i "SPACEMIT_EP_INTRA_THREAD_NUM|${cores}" -I

**LLM:**

    ./bin/llama-bench -m ${model} -t ${cores} -p 128 -n 128 -mmp 0 -fa 1 -ub 128

**VLM (encoder through the ONNX EP, decoder through llama.cpp):**

    export LD_LIBRARY_PATH=./spacemit-llama.cpp/lib:./spacemit_ort/lib
    export SPACEMIT_EP_DENSE_ACCURACY_LEVEL=1
    llama-server -m <text-gguf> --media-backend smt --smt-config-dir ./ -ctk f16 -ctv f16 -t 8 -c 1024 ...

## 3. The vendor's K3 data (ms, 1/2/4/8 AI cores unless noted)

| family | model | quant | shape | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| CNN | resnet18 | int8 | 224² | 8.00 | 4.74 | 2.90 | 2.07 |
| CNN | resnet50 | int8 | 224² | 21.00 | 12.21 | 7.72 | 5.47 |
| CNN | resnet50 (batch 4) | int8 | 224² | 77.99 | 42.75 | 24.62 | 15.38 |
| CNN | resnet50 | fp16 | 224² | 37.64 | 21.95 | 14.89 | 11.36 |
| mobile | mobilenet_v1 / v2 | int8 | 224² | 12.67 / 17.69 | 7.21 / 9.92 | 3.91 / 5.22 | 2.35 / 3.26 |
| mobile | mobilenet_v3_small / large | fp16 | 224² | 8.71 / 16.59 | 5.14 / 9.67 | 3.23 / 5.95 | 2.75 / 4.52 |
| efficientnet | v1_b0 / v1_b1 / v2_s | int8 | 224² | 40.84 / 63.46 / 58.41 | 22.51 / 34.46 / 33.46 | 12.89 / 19.80 / 19.80 | 9.85 / 14.75 / 13.53 |
| transformer | vit_b_16 | int8 | 224² | 100.90 | 56.64 | 35.19 | 23.76 |
| transformer | vit_b_16 | fp16 | 224² | 138.46 | 86.37 | 62.11 | 48.75 |
| detect | yolov5n / s / m | int8 | 640² | 43.76 / 73.40 / 152.03 | 24.11 / 40.20 / 81.90 | 14.36 / 23.95 / 45.88 | 9.57 / 15.81 / 28.97 |
| detect | yolov6n / s | int8 | 640² | 31.90 / 65.37 | 18.04 / 35.98 | 10.95 / 21.33 | 7.53 / 13.42 |
| detect | yolov8n / s / m | int8 | 640² | 41.41 / 74.50 / 157.80 | 23.08 / 41.19 / 85.68 | 13.85 / 24.89 / 48.90 | 9.51 / 16.65 / 32.31 |
| segment | yolov8n / s / m -seg | int8 | 640² | 61.59 / 103.27 / 204.94 | 33.57 / 56.33 / 109.94 | 19.40 / 33.24 / 61.93 | 12.75 / 21.51 / 39.61 |
| pose | yolov8n / s / m -pose | int8 | 640² | 46.42 / 81.61 / 165.60 | 26.18 / 45.33 / 90.30 | 16.16 / 27.87 / 51.75 | 11.16 / 18.64 / 33.84 |
| detect | yolo12n / s / m | int8 | 640² | 107.56 / 191.31 / 378.55 | 57.85 / 102.05 / 200.75 | 32.55 / 56.90 / 110.23 | 21.73 / 36.24 / 69.17 |
| audio | sensevoice | dyn_int8 | - | 0.1124 | 0.1380 | - | - |

**LLM (t/s):** qwen3-0.6B 499.75 PP128 / 53.35 TG128 · qwen3-1.7B 229.79 / 23.11 ·
qwen3-4B 76.44 / 11.03 · **qwen3-moe-30B-A3B 55.67 / 12.32 (PP1280 44.03, TG1280 11.17)** ·
qwen3.5-0.8B 182.69 / 29.33 · qwen3.5-2B (Q4_1) 112.22 / 16.15 · HY-MT1.5-1.8B 157.81 / 20.15 ·
llama2-7B 50.40 / 7.07.

**VLM (end-to-end ms, LLM 8 cores + vision 4 / 8 cores):** fastvlm-0.5B 512² 256.47 / **164.50** ·
Qwen3-VL-30B-A3B 768² 7928.13 / 4753.55 · Qwen3.5-0.8B 384² 340.42 / 245.61 · 2B 901.56 / 794.03 ·
4B 904.73 / 798.71.

**ASR (RTF):** Qwen3-ASR 0.6B 0.169 (LLM 8 / encoder 4) · 1.7B 0.329 · Fun-ASR Nano 0.247 ·
Gemma4 ASR E2B 0.689 transcription / 0.578 translation.

## 4. How our reproduction stands against those rows

| family | our instrument | local vs published |
|---|---|---|
| vision / detection (7 models, 4 thread counts, 28 rows) | vendor's own `onnxruntime_perf_test`, vendor's flags | **-1.4 .. +2.5 %**, one outlier vit_b_16 fp16 @8 cores **+4.9 %** |
| VLM encoder (not published) | same instrument | 241.52 ms @4, **150.89 ms @8**; the vendor's 4->8 core end-to-end delta (92 ms) is explained by our encoder delta (90.6 ms) |
| VLM end to end | board's packaged `llama-server --media-backend smt` | first frame 680.72 ms (encoder 185-245 ms inside prompt eval, decode 22.24 ms/token); cached frame 285.44 ms |
| LLM | board's packaged `llama-bench` (0.1.7~a0+4, build 5a23f07a4; `libllama-bench-impl.so` `787e2dcc...`, packaged `libggml-cpu.so.0.16.0` `ce31ff72...`) with the vendor's flags, two runs each | **PP128 -1.6..-2.0 %** (30B: 54.56/54.78 against 55.67) and **-1.8..-2.6 %** (0.6B: 486.70/490.69 against 499.75); **TG128 -8.0..-9.2 %** (11.18/11.34 against 12.32) and **-10.8 %** (47.58/47.60 against 53.35) |
| LLM (our own fork build, campaign numbers) | `llama-server` counters | 30B TG 11.17 - i.e. the same ~10 % below the published TG128 and identical to the packaged binary's 11.18, so the gap is not our build |

Reading: the ONNX/vision stack reproduces to a couple of percent, so its published rows are usable as
acceptance references. The LLM stack reproduces its **prefill** rows but not its **decode** rows
(~10 % optimistic), with the vendored package and our rebuild agreeing with each other - which makes
the published TG128 rows unsuitable as absolute acceptance targets and leaves our own controlled A/B
as the deciding instrument there. Cause not established; candidates are the board image revision
(4.0rc3 against 4.0.6), the EC/firmware packages that changed since May, and thermal/DDR state at the
time of their measurement.

## 5. Method alignment, aspect by aspect

| aspect | vendor method as published | our method | verdict |
|---|---|---|---|
| instrument | their `onnxruntime_perf_test` / `llama-bench` / `llama-server` | the same binaries, invoked with the same flags | aligned (their binary, their flags) |
| repetitions | `-r 10` in the vision example; LLM uses llama-bench defaults | `-r 10`, headline numbers from >= 3 independent runs (median) | ours stricter |
| statistic | the tables quote one number per cell; the tool also prints P50/P90/P99 | we quote mean **and** the percentile rows, and P50/P90/P99 from a sustained loop where latency matters | ours stricter |
| warm-up | not stated; the tool reports "First inference time cost" separately from the average | we verified the arithmetic (0.0550235 s / 10 = 5.502348 ms) and report **both**, plus session-creation time | ours stricter (their tables hide a 5-41x first-inference cost) |
| thread configuration | `SPACEMIT_EP_INTRA_THREAD_NUM` / `-t` per row, with ONNX Runtime's own intra-op pool pinned at one thread (`-x 1`) | same, **plus** a measured thread/TCM census (N intra -> N+1 threads, N TCM blocks, one AI core each) and the ORT pool sized to the deployment | ours adds ground truth; the doc's "2x intra threads" note is wrong for 2.0.6, and `-x 1` is a 2.32x under-measurement for graphs with a CPU fallback (`OPTIMISATION2-20260921.md` §2) |
| engine identity | engine version, OS and date per table section | package digests (tarball `bebcdfb7...`; deb runtime `076d9e83...`, provider `d8fa3633...`), model digests, board state (governor, temperature, load) per run | ours stricter |
| numeric/quality gate | none published for the quantised variants | tensor-dump equivalence before any precision/coverage claim (60 tensors, bit-identical when nothing should change) | ours stricter |
| AI-cluster attribution | none | `spacemit-tcm-smi` occupancy naming the workload's pid + per-thread affinity + the workload's own rate | ours stricter (contract requirement) |
| cross-cluster interference | none | X100 read ladder with the load generator's own rate: 23.35 -> 29.20 ms on ViT for 0 -> 15.18 GiB/s | ours stricter |
| caching semantics (VLM) | not stated | measured: the encoder runs on **every** frame (`smt_encode_media` 30/30 in a 30-frame loop); only the LLM prompt is cached (prompt_n 1) | ours stricter |

### Documented statements our measurements contradict

1. `SPACEMIT_EP_INTRA_THREAD_NUM` is documented as launching "2 x intra_thread_num compute threads";
   measured: **exactly N** compute threads, N TCM blocks (1 -> 2 process threads, 4 -> 5, 8 -> 9).
2. `SPACEMIT_EP_DUMP_SUBGRAPHS` is documented as taking a value with "1 to enable"; passing a
   directory path silently disables it (the dumps land in the working directory with a fixed prefix).
3. The VLM tables do not say whether the image embedding was cached; the difference is
   680.72 ms against 285.44 ms on the same request pair.
4. The published LLM TG128 rows are ~10 % above what the vendored binary produces on this board
   (Section 4); the same rows' PP128 are within 3 %.

## 6. Acceptance protocol (what "pass" means here)

* **R1 identity** - every result names: board packages (ort tarball + deb digests, llama.cpp package
  version), model digest, patch/commit hashes of anything we built.
* **R2 instrument** - vendor-comparable numbers use the vendor's own binaries with the vendor's
  flags; anything else is labelled as our instrument. The session shape is part of the instrument:
  the EP thread count *and* ONNX Runtime's own intra-op pool are recorded with every number, and the
  pool is set to the deployment's width. The published matrix's `-x 1` is a declared deviation, kept
  only where a graph has no CPU fallback (measured flat on seven of eight models, 2.32x on the
  eighth).
* **R3 warm/cold** - session creation, first inference and steady state are reported separately; a
  latency claim says which one it is.
* **R4 repetitions** - >= 10 timed repeats per row; headline numbers are medians of >= 3 independent
  runs; a difference below the measured run-to-run spread is never called an improvement.
* **R5 distribution** - latency-sensitive work (perception loops, control) quotes P50/P90/P99 and the
  sustained-loop span, not only the mean.
* **R6 numeric gate** - any precision, layout, coverage or operator change passes tensor-dump
  equivalence (or a stated tolerance with the reasoning) before its speed is quoted. The dump
  itself perturbs execution (measured: resnet50 5.48 ms without `SPACEMIT_EP_DUMP_TENSORS`, 29.15 ms
  with), so a dump run is a numeric instrument only and never a timing instrument.
* **R7 AI-cluster evidence** - work claimed on the AI cluster carries TCM occupancy naming the pid and
  the measured thread/affinity state; per-CPU counters are never used.
* **R8 interference isolation** - a two-cluster or pre/post-processing claim states the load on the
  other cluster and the measured penalty.
* **R9 state disclosure** - governor, frequency, temperature and system load are recorded with each
  number; large temperature or load excursions invalidate a run rather than being averaged in.
* **R10 named statistic** - a published row is used as an acceptance target only when the statistic
  it reports is known. Where it is not (the vendor's VLM table, §7.2), the row is an
  order-of-magnitude reference only and the acceptance basis is our own named statistic -
  encoder-only, time-to-first-token or end-to-end - stated with every number.
* **R10 cache disclosure** - server-based measurements state the cache state (prompt/media) and the
  per-new-frame cost separately.
* **R11 official comparison** - every claimed win states the corresponding published row (if one
  exists) and the delta against it, or says explicitly that no published row exists.
* **R12 reports are never a side effect** - a benchmark run writes raw output only; the report is a
  separate, reviewed artefact.

### Calibrated bands from the measurements so far

| quantity | measured spread | use |
|---|---|---|
| local vs published, vision rows | -1.4 .. +2.5 % (one +4.9 % outlier) | a vision row outside +/-3 % is investigated, not accepted |
| sustained inference (18,000 runs, YOLOv8n int8) | mean 9.478 / 9.478 / 9.490 ms; P99/P50 = 1.04 | a claim must exceed ~1 % to be visible outside noise; jitter is not a problem on this stack |
| local vs published, LLM prefill | -2.0 .. -2.6 % | published PP rows usable as references |
| local vs published, LLM decode | -9.2 .. -10.8 %, agreed by two independent builds | published TG rows are **not** acceptance targets; local A/B decides |
| VLM first frame vs cached frame | 680.72 / 285.44 ms | never quote one without the other |

## 7. Consequences for the project

1. The ONNX vision/VLM rows may be used as acceptance references directly (R11 satisfied by the
   reproduction band above).
2. LLM-side comparisons (the llama.cpp project, and any VLM decoder claim) must use our own A/B with
   the vendored binary or our build, and may only cite the published TG rows as context.
3. The next method improvement with the highest value is a *packaged-vs-fork* check for the ONNX
   stack: our vision rows were produced with the release tarball, while the board's deb pair is a
   different binary (Section 2 of `TECHNICAL-ANALYSIS.md`); the two should be measured against each
   other once, so future numbers can state which one they used.
4. Tooling note for the int8-encoder direction: `xslim` (the vendor's PTQ tool) is not installed
   here. Its 2.1.1 metadata requires `torch>=2.0`, `torchvision`, `onnxruntime`, `opencv-python`,
   `pandas`, `onnx-graphsurgeon` and more; the package resolves `--dynq` (dynamic quantization, no
   calibration set) and `--fp16` without a config file, so the cheapest first attempt is a CPU-only
   torch wheel plus `--no-deps` for the rest, run against `fastvlm_vision.f16.onnx`. This round
   installed pip and started that path but did not finish it; the measurement above (ViT int8 vs
   fp16 = 2.17x) is what makes it worth the next slot.

## 8. Packaged tarball against the board's deb pair (the one-time check of Section 7.3)

Same instrument, same models, same flags; the loader was verified (`LD_LIBRARY_PATH=/usr/lib ldd`
resolves `libonnxruntime.so.1` to `/usr/lib`):

| library set | digests | resnet50 int8 @8 | vit_b_16 int8 @8 |
|---|---|---|---|
| release tarball 2.0.6 | runtime `93bb7560...`, provider `dcc95030...` | 5.469 ms | 23.387 ms |
| board deb 2.0.6 | runtime `076d9e83...`, provider `d8fa3633...` | 5.477 ms (+0.15 %) | 23.483 ms (+0.41 %) |

and the numeric gate agrees: with `SPACEMIT_EP_DUMP_TENSORS` on resnet50 in both sets, **all 60
intermediate tensors are bit-identical**. So the two 2.0.6 builds are behaviourally equivalent for
the workloads measured here; a future number only has to say which set it used. Our round-1 tables
used the tarball, the VLM rounds used the deb pair through the board's packaged llama-server.

## 7. The two rows that do not reconcile: what was checked, and what is left

Two published rows have never reproduced on this board: the LLM decode rows (`tg128`, ~10 % below)
and the VLM end-to-end rows. This section closes the controllable axes one by one, so that what
remains is a named variable rather than a suspicion.

### 7.1 LLM `tg128`: every axis we control is identical; the OS image is what is left

The published capture is the vendor's own terminal output for their example command, so it can be
compared field by field, not just number by number:

| axis | published (2026-05-26) | ours (2026-09-21) | verdict |
|---|---|---|---|
| package | llama.cpp `0.1.1` (their release tarball) | the board's `0.1.7~a0+4` (build `5a23f07a4`) **and** their own `0.1.1` release, both run on this board | both measured |
| command | `llama-bench -m ... -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128` | identical | identical |
| model | `qwen3 0.6B Q4_0`, 358.78 MiB, 596.05 M | our run prints the same three fields | identical |
| runtime banner | all nine fields, plus the `tcm_sync_mem` failure and the `init_barrier` heap fallback | **field-identical**, including both failure lines | identical |
| thread/batch/mmap columns | `8 / 128 / 1 / 0` | identical | identical |
| statistic | `llama-bench` `pp128`/`tg128` mean +- sd | identical | identical |
| result `pp128` | 499.75 +- 0.22 | 484.13 (their `0.1.1`) / 490.69 (packaged) | **-3.1 % / -1.8 %** |
| result `tg128` | 53.35 +- 0.03 | 47.30 (their `0.1.1`) / 47.60 (packaged) | **-11.3 % / -10.8 %** |
| **OS** | **bianbu-4.0rc3** | **Bianbu 4.0.6, kernel 6.18.3** | **the only remaining variable** |

Version drift is therefore eliminated (their own release package lands at 47.30, not 53.35), runtime
configuration is eliminated (the banner is the same line), and identity is eliminated (the model row
is the same row). What is left is the OS/firmware generation, and the one experiment that would
settle it - re-imaging the board to `4.0rc3` - is a human operation outside an AI contributor's
authority. Until then the rule of §4 stands and is now evidence-backed: the published **prefill**
rows are usable acceptance references, the published **decode** rows are an order-of-magnitude
reference only, and the acceptance line for decode is our own controlled A/B on one board state.
Evidence: `/root/k3-evidence/llama.cpp/modelzoo-and-freeze-20260920T0140Z/README.md` (both packages)
and `/root/k3-evidence/onnxruntime/baseline-20260921/llm-packaged.txt` (the packaged run, with the
binary, library and model digests).

### 7.2 VLM: the row's statistic is undefined, and the candidate readings disagree by 40 %

The VLM table's column header names the configuration (`LLM 8 Core + VisionEncoder N Core/ms`) and
never says which quantity is in the cell: the vision stage, the time to first token, or a whole
request. It matters, because the candidates differ by 40 %. Measured in one session on this board
with the vendor's own model (`fastvlm-mm-0.5b-q4_1`), their documented server command and their own
request payload (mime string included), at both encoder widths:

| statistic | N = 4 | N = 8 | against the published row |
|---|---|---|---|
| **published row** | **256.47** | **164.50** | - |
| vision encoder alone (our `perf_test` on `fastvlm_vision.f16.onnx`, 512²) | **243.02** | **152.01** | **-5.2 % / -7.6 %** |
| time to first token, their server, client-side | 317.0 | 229.7 | +23.6 % / +39.6 % |
| server's own `image prefill (smt) processed in N ms` | 159 | 162 | not the quantity - see below |
| same request, 24 tokens, end to end | 1013.6 | 937.9 | - |

The 4 -> 8 core step is identical in all three framings - **91.97 ms** in the published row, **91.01
ms** in our encoder measurement, 87.3 ms in our TTFT - so the scaling is reproduced whatever the
unit is. The *level* only matches under one reading: **the cell is the vision encoder's own inference
time**, and the column header then describes the encoder that produced it. Under the other reading
(time to first token) the published row is 24-40 % better than the same board, model, server and
flags produce, and no configuration we can find closes that. Two consequences for acceptance: the
VLM rows are usable as an order-of-magnitude reference against our encoder number (-5 to -8 %), and
they are **not** usable as an end-to-end service target, because the row does not contain the
request-level overhead (~60-90 ms above the encoder on this board).

Two instrument notes from the same session, both worth carrying:

* **the server's own `image prefill (smt) processed in N ms` line must not be quoted as the vision
  stage cost.** It read 159 ms and 162 ms at four and eight encoder threads while the client-visible
  TTFT moved 87 ms, i.e. the line is insensitive to the knob that defines the table's two columns.
  Our first attempt at this comparison used it and would have concluded "no difference";
* **the encoder width is not an environment variable here.** It is
  `vision_model.spacemit_ep_intra_thread_num` in the model directory's `config.json` (documented for
  the Qwen3.5 layout in `llama.cpp.md` §2.4.2). The ASR examples in the same documentation do set
  `SPACEMIT_EP_INTRA_THREAD_NUM` on the command line, so the two paths differ; setting the variable
  on the VLM path leaves the banner at 4 and makes the two "columns" identical, which is how the
  mistake was caught. Both runs of the corrected pair are in
  `/root/k3-evidence/onnxruntime/baseline-20260921/vlm-clocks/` with their server logs and TCM
  samples.
