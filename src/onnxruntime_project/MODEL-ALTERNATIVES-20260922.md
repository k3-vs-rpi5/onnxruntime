# Model alternatives: replacing what this board is weak at, domain by domain

**Scope note.** The optimisation track treats the model as a fixed input. The operator has directed a
second, separate track: where this board is *weak* - either at a specific model or across a whole
vision domain - find efficient, accuracy-competitive alternatives. Nothing here changes the frozen
models; it names candidates and the procedure that decides them. Model acquisition is listed as a
step to be approved, not taken silently.

## 1. The weaknesses, as measured rather than assumed

| weakness | measurement | what it costs |
|---|---|---|
| segmentation (`pp_liteseg.q`) | 13 node executions per frame on the CPU EP, **25.5 % of the frame**; the whole frame moves 2.32x with ONNX Runtime's pool width | 199.47 ms where 85.95 ms is available |
| detection (`yolo26n.q`) | 16 CPU node executions per frame, **8.2 % of the frame** - and insensitive to the pool width, because its nodes are sequential decoders | ~1.5 ms of a 17.96 ms frame |
| several domains | no board material at all: pose, depth, OCR, lanes/drivable area, tracking, feature matching, promptable segmentation, face *recognition*, open-vocabulary detection, audio | unknown |

The instrument that produced the first two rows is the one that should gate every candidate:
`tools/board-coverage-census.sh` plus `tools/ort-provider-census.py` answer "does the provider claim
this graph, and what share of the frame does it hand back to the CPU" in one run per model.

## 2. The candidate pool is the vendor's own catalogue

`model-zoo-vision` (HEAD `45c3878`) ships **42 model families** with their own configs and download
scripts - i.e. already integrated against this provider, which makes them the cheapest candidates to
verify. By domain:

| domain | families in the vendor catalogue | on this board today |
|---|---|---|
| detection | `yolov5`, `yolov8` (+ no-DFL), `yolov11`, `yolo12`, `yolo26`, `deimv2`, `rfdetr`, `yoloe`, `yolo_world` | yolov8n/s, yolov8n-no-DFL, yolo12n, yolo26n |
| segmentation | `yolov8_seg`, `yolo26_sem`, `mobileseg`, `pp_liteseg`, `yolop`, `yolopv2`, `banet2d` | yolov8n-seg, pp_liteseg |
| pose / tracking | `yolov8_pose`, `mixformer`, `bytetrack`, `ocsort`, `nanotrack`, `avtrack` | none |
| depth | `yolo26_depth` | none |
| classification | `resnet`, `mobilenet`, `mobilenetv1`, `efficientnet`, `efficientnet_v2s`, `vit`, `emotion` | resnet18/50, vit (int8 + fp16) |
| embedding / retrieval | `mobileclip2`, `siglip2` | image_encoder.fp16 (MobileCLIP2-S3), siglip2 dynq |
| face | `scrfd`, `arcface`, `adaface`, `yolov5-face` | SCRFD `det_10g_fixed` only |
| OCR | `ppocr` (PP-OCRv6 tiny det + rec) | none |
| local features / promptable | `superpoint`, `lightglue`, `mobilesam1` | none |

One correction to the record, found while building this list: the file this project had filed as
`rfdetr-nano.fp16.onnx` is a 434-byte archive error response, not a model - RF-DETR was never
downloaded, so it is a candidate, not a measured one.

## 3. Substitutions for the two measured weaknesses

**Segmentation.** The same domain already contains models that pass the provider test we failed
`pp_liteseg` on:

| candidate | status | evidence |
|---|---|---|
| `yolov8n-seg` (official zoo) | **measured EP-complete** (0 CPU nodes) | 12.85 ms at 8 cores against the published 12.75 ms - reproduced |
| `mobileseg` | candidate, unmeasured | vendor example with its own config |
| `yolo26_sem` | candidate, unmeasured | semantic segmentation, `fp16` |
| `yolop` / `yolopv2` (288x512) | candidate, unmeasured | driving: drivable area + lanes, i.e. the same use case `pp_liteseg` was chosen for |

So for a driving-scene segmenter the honest position is: the board already has one EP-complete
alternative measured (`yolov8n-seg`), and three unmeasured ones whose published/vendor numbers we have
not verified. `pp_liteseg` remains the only model in this project with a measured 2.32x configuration
penalty.

**Detection.** `yolo26n`'s 16 CPU-side nodes are structural to that model; the same-domain models we
have *measured* as EP-complete are `yolov8n` (9.54 ms), `yolov8n_no_dfl` (9.17 ms), `yolov8s`
(16.60 ms) and `yolo12n` (22.57 ms). Candidates that have not been measured here: `yolov11`,
`deimv2`, `rfdetr` (retrieval failed once), `yoloe` and `yolo_world` (open-vocabulary).

## 4. Filling the empty domains, in the order the project's own focus suggests

Priority is set by the work this project is already doing - VLM and embodied perception - and by what
can be verified against something other than our own word:

1. **pose** (`yolov8n-pose`): the vendor publishes a K3 row for it (11.16 ms at 8 cores), so a
   candidate can be checked against an official number the same way the vision tables were;
2. **depth** (`yolo26n-depth`): the perception input an embodied stack needs next (obstacle distance);
   no published row, so our own two-cluster and thread rules apply unchanged;
3. **OCR** (`ppocr`, PP-OCRv6 tiny det + rec): a common vision domain entirely absent here, and one
   where the provider's operator list makes a coverage census worth running before anything else;
4. **face recognition** (`arcface` mobile / `adaface`): the detector side is already on the board
   (SCRFD `det_10g_fixed`), so this completes a pipeline of two models;
5. **open-vocabulary detection** (`yoloe`, `yolov8s-worldv2`): text-prompted detection, the perception
   half of a language-conditioned robot task;
6. **local features** (`superpoint` + `lightglue`) and **promptable segmentation** (`mobilesam1`):
   the remaining embodied building blocks;
7. **classification breadth** (`mobilenet` v3, `efficientnet_v2s`): lowest priority - the domain is
   already covered by three measured models, and both candidates have official rows.

## 5. The acceptance procedure every candidate goes through

Nothing in this project is called "better" without these steps, and a substitution decision is a
deployment decision, not an optimisation result:

1. **coverage census first** (one run): if the provider does not claim the graph, price the fallback
   before going further - `pp_liteseg` is the worked example of what an unclaimed graph costs
   (25.5 % of the frame, 2.32x of the pool-width rule);
2. **published-row check** where the vendor publishes one (`yolov8n-pose`, `mobilenet`, `vit`, ...):
   reproduce it on this board before trusting any candidate number;
3. **deployment shape**: one session, eight threads, ORT pool sized to the deployment, never split
   the cores (measured: -36 %);
4. **numeric gate** for anything quantised: tensor-dump equivalence, since a candidate that changes
   answers is not a substitute;
5. **both clusters declared**: the X100 is not a co-processor for these graphs (97x on YOLOv8n), so
   pre/post stages there are the only sanctioned use, and the interference budget is 10-25 %.

## 6. What this needs from the operator

The candidates above are one download each from the vendor's own archive (10-100 MB per model, no
model conversion needed since they ship as ONNX). Within the current material the work that needs no
download is: the pose candidate is the only one with a published row, and it, the segmentation
alternatives and the OCR pair are the four that would move this track forward first.

## 7. The menu, with sizes checked

Filenames were taken from the vendor's own `download_models.sh` in each example and every URL was
probed with a `HEAD` before being listed, so the sizes below are the archive's own `Content-Length`:

| priority | domain | file (under `.../model_zoo/vision/`) | size |
|---|---|---|---|
| 1 | pose | `yolov8_pose/yolov8n-pose.q.onnx` | 3.80 MB |
| 2 | depth | `yolo26_depth/yolo26n-depth.fp16.onnx` | 10.46 MB |
| 3 | OCR (det + rec) | `ppocr/PP-OCRv6_tiny_det_640x640.fp16.onnx` + `PP-OCRv6_tiny_rec_48x320.dynq.onnx` | 0.86 + 1.17 MB |
| 4 | face recognition | `arcface/arcface_mobilefacenet_cut.q.onnx` (and `adaface/adaface_ir101_webface12m_merged.dynq.onnx`, 62.48 MB) | 1.07 MB |
| 5 | open-vocabulary detection | `yoloe/yoloe-v8s-seg.dynq.onnx` | 13.22 MB |
| 6 | local features + matcher | `superpoint/superpoint_512x512_top512_batch1.q.onnx` + `lightglue/lightglue_for_superpoint_512_depth9.fp16.onnx` | 1.27 + 21.75 MB |
| 7 | promptable segmentation | `mobilesam1/mobilesam_encoder_tiny_sim.fp16.onnx` + `mobilesam1/mobilesam_decoder_sim.dynq.onnx` | 13.37 + 4.92 MB |
| 7 | semantic segmentation | `yolo26_sem/yolo26n-sem.fp16.onnx` | 3.05 MB |

The four highest-priority candidates together are **16.3 MB** - the acquisition is not the obstacle;
the decision is.

**One procedure note, learned from a mistake this project made.** Filenames must be taken from the
vendor's scripts, never guessed, and every URL must be checked for a sane `Content-Length` before its
payload is treated as a model. Guessing produced the opposite: a `NoSuchKey` XML error page of 434
bytes was filed as `rfdetr-nano.fp16.onnx` and sat in this project's evidence for a day looking like a
model (the same trap returned 442- and 447-byte pages for two guessed face-recognition names in this
round). A candidate list built by guessing names would have been mostly fiction.

## 8. The first four candidates, measured

Downloaded (17 MB total), pushed, and put through the acceptance procedure in order - coverage census,
published-row check where one exists, then the deployment shape with a TCM sample inside the run. All
numbers below are unprofiled, `-r 20`, `-x 8`, one session:

| candidate | census | EP 4 threads | EP 8 threads | published row | verdict |
|---|---|---|---|---|---|
| `yolov8n-pose.q` | **EP-complete** (0 CPU nodes) | 16.185 ms | **11.195 ms** | **11.16 ms** | **reproduced to 0.3 %** |
| `PP-OCRv6_tiny_det_640x640.fp16` | **EP-complete** | 56.374 ms | **35.989 ms** | none | accepted, baseline established |
| `PP-OCRv6_tiny_rec_48x320.dynq` | **EP-complete** | 3.114 ms | **2.358 ms** | none | accepted, baseline established |
| `arcface_mobilefacenet_cut.q` | **EP-complete** | 4.227 ms | **3.137 ms** | none | accepted, baseline established |
| `yolo26n-depth.fp16` | 18.4 % head fallback | 98.834 ms | **64.844 ms** | none | accepted with a measured defect (below) |

Three things worth carrying out of this table:

1. **the pose candidate confirms the published vision rows a third time.** 11.195 ms against the
   official 11.16 ms is inside the noise of a row reproduced from the vendor's own table - the same
   relationship the rest of this project found for vision, and the opposite of the LLM/VLM rows, which
   sit 10-13 % above what the vendor's own tools produce here;
2. **a vendor-shipped dynamic-quantisation model is claimed in full.** The OCR recogniser is `.dynq`
   and the provider takes every node of it, which matters because our own earlier attempt to produce a
   dynq model with `xslim` was refused (`Conv` bound to different types) and the dynamic route was
   written off. The difference is who produced the artefact, not the technique;
3. **TCM engagement is evidenced, not assumed.** Sampled while a 400-inference pose run was in flight:
   `available_blocks=0/8` - all eight AI-core blocks held by that pid. Every number in this table has
   that class of evidence behind it (`spacemit-tcm-smi`, plus the thread census the campaign uses).

The depth candidate is the exception and now carries a measured defect instead of a hope: 18.4 % of its
frame is a CPU-side head (`Resize` 6.34 ms, `Cast` 2.14, `Pow` 2.00, `Exp` 0.11), four rewrite
families were tried against it, and none is carried as pending - the two that were mathematically
safe changed nothing, the one the mechanism pointed at made the frame 16 % slower, and the one that
reimplemented the operator crashed the provider. It is reported to the vendor with minimal
reproductions (`VENDOR-QUERY-20260922.md`); until that comes back, this candidate is usable at
64.84 ms and its defect is priced.

## 9. The rest of the catalogue: eleven more candidates

Everything else on the priority list was then fetched, HEAD-checked and measured the same way
(census first, then unprofiled latency at four and eight EP threads). The vendor's own scripts and the
archive's `Content-Length` were used for filenames and sizes; two guessed names returned the usual
400-byte error pages and were replaced from the scripts.

| candidate (domain) | census | EP 4 | EP 8 | official row | verdict |
|---|---|---|---|---|---|
| `mobilenet_v3_small.fp16` (classification) | EP-complete | 3.335 ms | **2.719 ms** | **2.75 ms** | **reproduced (-1.1 %)** |
| `mobilenet_v3_large.fp16` (classification) | EP-complete | 5.982 ms | **4.529 ms** | **4.52 ms** | **reproduced (+0.2 %)** |
| `efficientnet_v2_s.q` (classification) | EP-complete | 19.828 ms | **13.860 ms** | **13.53 ms** | **reproduced (+2.4 %)** |
| `mixformer_v2.q` (pose) | EP-complete | 22.006 ms | **14.984 ms** | none | accepted, baseline established |
| `yolo26n-sem.fp16` (semantic segmentation) | EP-complete | 96.194 ms | **52.255 ms** | none | accepted, baseline established |
| `lightglue_for_superpoint_512_depth9.fp16` (matcher) | EP-complete | 61.075 ms | **41.139 ms** | none | accepted, baseline established |
| `banet_2d_small_512x640...q` (lanes) | EP-complete | 257.010 ms | **146.787 ms** | none | accepted, baseline established |
| `mobilesam_encoder_tiny_sim.fp16` (promptable seg) | EP-complete | 63.942 ms | **40.304 ms** | none | accepted, baseline established |
| `superpoint_512x512_top512_batch1.q` (local features) | **13.8 % fallback** (4 nodes) | 26.655 ms | 15.985 ms | none | accepted with a measured defect |
| `mobilesam_decoder_sim.dynq` | **20.3 % fallback** (270 node executions per frame) | 31.318 ms | 28.983 ms | none | accepted with a measured defect |
| `yoloe-v8s-seg.dynq` (open-vocabulary) | **37.9 % fallback** (12 nodes) | 184.463 ms | 157.545 ms | none | accepted with a measured defect |

Three more published rows land inside a couple of percent - `mobilenet_v3_small`, `mobilenet_v3_large`
and `efficientnet_v2_s`, joining `yolov8n-pose` from section 8. That is now **six independent
reproductions of the vendor's vision tables** in this project, against zero for its LLM decode rows.

The fallback pattern is worth reading as an architecture map: the provider claims the trees
(MobileNet, EfficientNet, MixFormer, LightGlue, BANet, the segmentation head, the SAM encoder) and
declines three shapes - an open-vocabulary head (`yoloe`, 37.9 %), a decoder with many small operators
(`mobilesam_decoder`, 270 node executions per frame, 20.3 %) and a local-feature extractor with a
top-k output (`superpoint`, 4 nodes, 13.8 %). None of the three is a Resize-style incident: they are
the next candidates for the converter once the vendor answers the context question.

### 9.1 Correction: dynamic quantisation is not what failed

Round 3 wrote off the dynamic-quantisation route after our own `xslim --dynq` output was refused by
the provider on an fp16 encoder and, on an fp32 copy, ran at 2.10 s. This round qualifies that:

* the vendor's own **`PP-OCRv6_tiny_rec_48x320.dynq`** is **EP-complete** and runs in **2.340 ms**;
* `mobilesam_decoder_sim.dynq` and `yoloe-v8s-seg.dynq` are partly claimed but still execute on the
  provider for the majority of their nodes.

So the technique is not the problem - **who built the artefact is**. A `.dynq` model produced by the
vendor's toolchain for a graph it was designed for is claimed; our own dynq of a vision transformer
was refused in fp16 and slow in fp32. The rejected finding stays scoped to that encoder rather than to
dynamic quantisation as a class.

## 10. The OCR pipeline, end to end

Detection and recognition are two models and one real service shape, so they were measured as a
pipeline rather than as two rows. Input: a generated 960x640 frame with three lines of text
(`ocr-img/text-960x640.png`); both stages on the provider, eight threads, `-r 30` (solo) and `-r 200`
/ `-r 2000` (concurrent), image fixed throughout.

| stage | solo | concurrent with the other stage |
|---|---|---|
| `PP-OCRv6_tiny_det_640x640.fp16` | 35.989 ms | **38.420 ms (+6.8 %)** |
| `PP-OCRv6_tiny_rec_48x320.dynq` | 2.340 ms | **6.095 ms (+160 %)** |

Reading: the detector is **94 %** of a frame, so a two-stage pipeline can win at most the 6 % the
recogniser costs - and running the two stages at once costs the detector 6.8 % in interference. A
frame therefore takes **38.33 ms serialised against 38.42 ms overlapped**: the overlap buys nothing.
The engineering conclusion is the opposite of the usual one: **serialise these two stages on the same
eight cores** rather than giving each its own, and do not spend effort on the 2.3 ms stage.

### 10.1 Correction: that was TCM contention, not a pipeline shape

The paragraph above was written before the instrument that decides the question was pointed at it, and
`spacemit-tcm-smi` refutes the framing. Sampled while the two stages ran "concurrently":

```
available_blocks=0/8
ID 0..7   busy   ...   PID 3505078        <-- every block names ONE process
```

**All eight AI-core TCM blocks belong to a single process** - the first one to start - and the second
process runs with none. So two EP *processes* on this board do not share the cluster; the second is
starved of the resource the first took in full. The same table reads the same way in the second
pipeline measured below. What section 10 actually measured is therefore *two processes fighting over
the TCM table*, which is a real and reportable deployment finding (and the mechanism behind round 5's
"one EP process per board" rule), but it is **not** a measurement of a two-stage pipeline.

Two things follow, and both are recorded rather than patched over:

* the deployment rule stands and is now explainable: a service that puts two models in two processes
  pays contention on a global eight-block resource, so the practical shape remains **one EP process
  per board**, stages serialised inside it;
* the pipeline question - whether two stages of similar cost should overlap - is **unmeasured**. It
  requires two *sessions* inside one process sharing the provider's compute pool, which is exactly what
  the vendor's `SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD` documents ("all models share the same compute
  resources") and which `onnxruntime_perf_test` cannot express: it runs one model per process. That is
  a host application, not a flag, and it is the next artefact this track needs.

### 10.2 Two more pipelines, measured the same way (with the TCM table this time)

For completeness, and because the numbers are still useful as *contention* data, two further pairs
were measured with the same three configurations:

| pipeline (stage A / stage B) | stage | solo, 8 threads | overlapped 8 + 8 | split 4 + 4 overlapped |
|---|---|---|---|---|
| SAM: `mobilesam_encoder_tiny` / `mobilesam_decoder` | A | 40.304 ms | **43.565 ms** | 67.228 ms |
| | B | 28.983 ms | **38.866 ms** | 49.638 ms |
| detect→pose: `yolov8n` / `mixformer_v2` | A | 9.540 ms | **19.170 ms** | 29.242 ms |
| | B | 14.984 ms | **22.946 ms** | 34.865 ms |
| OCR: `PP-OCRv6_tiny_det` / `PP-OCRv6_tiny_rec` | A | 35.989 ms | 39.000 ms | - |
| | B | 2.340 ms | 5.637 ms | - |

Read with the TCM table in mind, the three pairs span +8 %/+34 % (SAM), +101 %/+53 %
(detect→pose) and +8 %/+141 % (OCR) - a spread that says more about which process won the TCM race
and what each model can do without blocks than about the stages themselves. That spread is the reason
this section now carries a caveat instead of a deployment rule for pipelines: **the numbers are
measurements of a contended board, and they are labelled that way.**

## 11. The pipeline question, answered inside one process

Section 10.1 named the missing artefact: two *sessions* in one process, which `onnxruntime_perf_test`
cannot express because it runs one model per process. It now exists as
`tools/ort-two-session-pipeline.cpp`, cross-compiled with the workspace toolchain and run on the board.
It loads the provider the way the vendor's own media engine does (`dlopen("libspacemit_ep.so")` and
`OrtSessionOptionsSpaceMITEnvInit`) because the provider is **not** registered with the runtime as an
ordinary one - a Python `providers=[...]` list silently falls back to the CPU EP, which is how a
harness can measure the CPU while believing it measured the AI cluster.

Pipeline: `yolov8n` (detection) then `mixformer_v2` (pose), both EP-side, one process, two sessions.

| shape | stage A | stage B | frame | throughput |
|---|---|---|---|---|
| serial, 4 + 4 cores | 14.019 ms | 22.001 ms | 36.02 ms | 27.76 fps |
| **overlap, 4 + 4 cores** | 15.281 ms (+9.0 %) | 23.202 ms (+5.5 %) | **23.20 ms** | **43.10 fps** |
| overlap, 4 + 4 with `USE_GLOBAL_INTRA_THREAD=1` | 15.304 ms | 23.120 ms | 23.12 ms | 43.25 fps |
| serial, 8 + 8 cores | - | - | **impossible**: `Not enough available AI cores for the thread pool: required 8, available 0` | - |
| two processes, one model each, run alternately at 8 cores (derived from the solo rows) | 9.540 ms | 14.984 ms | 24.52 ms | 40.78 fps |

Three findings, each with its mechanism:

1. **the AI cores are allocated per session, and 8 + 8 is not a shape**: the first session takes all
   eight cores and the second cannot create at all. The vendor's documented
   `SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD` does **not** lift that: with it set, `8 + 8` still fails and
   `4 + 4` performs identically to the non-shared case (43.25 against 43.10 fps). Two sessions in one
   process therefore **must split the cluster**, and 4 + 4 is the split that works;
2. **overlapping inside the process pays 1.55x** over serialising the same two sessions - and it does
   so cheaply: +9.0 % on the detector and +5.5 % on the pose model, against the 8 %/34 % to 101 %/53 %
   that the *cross-process* attempts showed. That contrast is the whole point of the earlier caveat:
   the cross-process numbers were contention, these are a pipeline;
3. **the in-process pipeline also beats the shape this project had recommended**: running the two
   models alternately, each with all eight cores and one process at a time, gives 24.52 ms per frame
   (40.78 fps) against the overlapped 4 + 4 session pair at 23.20 ms (43.10 fps) - **5.7 % better**,
   because a stage that waits for the other to finish leaves the cluster idle for the whole of the
   other stage. The earlier rule ("one EP process per board, stages serialised") is therefore updated:
   one process, **two sessions, cores split 4 + 4, stages overlapped**.

The TCM evidence matches the configuration: sampled during the overlapped run the eight blocks are
busy with **two different thread-group ids**, four blocks each - the two sessions sharing the cluster
the way the split says they should, instead of one process holding all eight as in section 10.1.

Method note, because it cost two build-measure cycles: this harness's first version read the input
shape from a `TypeInfo` temporary and got rank 0, and its second version wrapped an input buffer that
was local to the constructor - `CreateTensor` does not copy - so the first `Run` read freed memory and
segfaulted. Both were found by printing what the code saw rather than by reasoning about the provider,
which is the order section 5.0 prescribes.

### 11.1 The split has to be searched, not assumed - and it follows thread scaling, not stage cost

The detect→pose pair is 1.6:1 in cost, and 4 + 4 was simply the split the provider accepts. The SAM pair
is a better teacher: `mobilesam_encoder_tiny` loses 59 % of its speed going from eight threads to four
(40.304 -> 63.942 ms) while `mobilesam_decoder` loses 8 % (28.983 -> 31.318 ms), so the two stages scale
very differently and the split should follow that rather than the cost ratio. The harness was extended
to take a thread count per stage and the split searched:

| shape (encoder + decoder cores) | encoder | decoder | frame | throughput |
|---|---|---|---|---|
| serial 4 + 4 (in process) | 64.561 ms | 31.240 ms | 95.80 ms | 10.44 fps |
| overlap 4 + 4 | 65.811 ms | 36.562 ms | 65.81 ms | 15.20 fps |
| overlap 5 + 3 | 60.109 ms | 39.340 ms | 60.11 ms | 16.64 fps |
| **overlap 6 + 2** | 52.798 ms | 43.519 ms | **52.80 ms** | **18.94 fps** |
| overlap 7 + 1 | 44.806 ms | 61.260 ms | 61.26 ms | 16.32 fps |
| two processes, one model each, alternately at 8 cores (derived) | 40.304 ms | 28.983 ms | 69.29 ms | 14.43 fps |

The mechanism is visible in the table: the encoder keeps gaining from cores (65.8 -> 52.8 -> 44.8 ms as
it goes 4 -> 6 -> 7) while the decoder loses them badly past two (36.6 -> 43.5 -> 61.3 ms), so the
optimum sits **on the edge where the encoder is still the bottleneck** - 6 + 2 gives 18.94 fps, and one
more core for the encoder turns the decoder into the bottleneck and costs 14 % instead. Against the
best non-pipeline shape this project knows (two processes alternating at eight cores, 14.43 fps), the
searched split is **+31 %**; against serialising the same in-process pair, **+81 %**.

So the refined deployment rule is: **one process, one session per model, stages overlapped, and the
core split searched rather than assumed** - the split that wins is the one that leaves the
strongest-scaling stage one core short of handing the bottleneck to its neighbour.

## 12. The same method on the VLM service: it does not transfer, and the official row agrees we are ahead

The VLM is also a two-stage workload - a vision encoder on the ONNX EP and a text decoder in
llama.cpp - so the section-11 method was applied to it. Both stages were measured as scaling curves on
this board, and the encoder carries the comparison the operator asked for on every number:

| encoder width | ours | vendor's published row | delta |
|---|---|---|---|
| 4 AI cores (`fastvlm_vision.f16`, 512²) | **243.08 ms** | 256.47 ms | **-5.2 %** |
| 8 AI cores | **150.02 ms** | 164.50 ms | **-8.8 %** |

so on the one axis the published row defines, this project is *faster* than the official figure, and
by a margin larger than the row's own run-to-run spread. The full curve: 735.52 / 410.08 / 294.48 /
243.08 / 205.10 / 181.77 / 170.05 / 150.02 ms at 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 threads.

The decoder curve, measured the same way (`llama-bench`, the vendor's LLM flags, `tg128` on the VLM's
own text GGUF): 22.26 / 25.85 / 35.75 / 43.56 / 48.38 / 51.80 / 52.93 / 53.67 tokens/s - the same
eight widths, and note the shape: 1 -> 2 threads buys 16 %, 2 -> 4 buys 68 %.

Composing them for a single-stream frame (one image plus a 20-token answer, frame = max(encode(a),
decode(b)) with a + b = 8):

| split (encoder, decoder) | encode | decode (20 tokens) | frame |
|---|---|---|---|
| 1, 7 | 735.52 ms | 377.9 ms | 735.5 ms |
| **2, 6** | 410.08 ms | 386.1 ms | **410.1 ms** |
| 3, 5 | 294.48 ms | 413.4 ms | 413.4 ms |
| 4, 4 | 243.08 ms | 459.1 ms | 459.1 ms |
| 5, 3 | 205.10 ms | 559.4 ms | 559.4 ms |
| 8, 8 serialised (encode, then decode - what the server does today) | 150.02 ms | 372.6 ms | 522.6 ms |

The composition's answer is **(2, 6) at 410.1 ms/frame, +27 % over the serialised shape**, and it obeys
section 11.1's rule - the split lands where the encoder stops being the bottleneck and the decoder has
not yet become one.

**The real service refuses it.** Running the vendor's own `llama-server` with four slots and four
simultaneous image requests (32 tokens each):

| served shape | aggregate |
|---|---|
| encoder 8 threads, text `-t 8` | **47.88 t/s** |
| encoder 4 threads, text `-t 8` | 42.14 t/s |
| encoder 2 threads, text `-t 6` - the composed optimum | **32.22 t/s (-33 %)** |

The reason is a difference in *what overlaps*. The ORT pipelines overlap **stages** of one frame, and a
split then decides which stage is the bottleneck. The VLM server overlaps **requests**: four slots run
their encode and decode phases interleaved, and in that regime both phases want the whole cluster -
starving the encoder to two threads costs more than any stage overlap can return. So the section 11
method is a **single-stream** tool; it does not transfer to a multi-slot service, and the deployment
rule for the VLM service stays what it was measured to be: **one server, four slots, encoder 8, text 8,
requests queued.**

Two things this section is careful about, because the operator's standing instruction is not to
celebrate alone: every number above sits next to the official row or is explicitly marked as having
none (the decoder has no published figure, and the published *service* row defines no statistic at all
- acceptance rule R10), and the composed prediction that the real server refused is kept in the table
as the counter-example rather than deleted.

## 13. Research round: the vendor's own multi-model deployment method

The section-11/12 work was done before reading the vendor's own multi-model solution documents, so this
is the round that puts the scheme next to theirs. Two documents carry the method and the numbers.

**`solutions/aicomputer_solution/multi_stream_vision.md`** - a 12-stream 1080p video analytics
application, with a published table per model (total FPS over all twelve streams, VPU+GPU and DDR
bandwidth alongside): `yolov8n` **134.29**, `yolov5n` 135.35, `yolov8n-pose` 111.78, `yolo26n` 104.18,
`yolov11n` 101.81, `yolov8n-seg` 98.56, `yolov8s` 77.74, `yolov8m` 39.23, with **19-22 GB/s of DDR
bandwidth** per run. The method sentence that matters here: the application "keeps only the latest frame
for each channel and distributes inference tasks to **multiple sessions** in round-robin order".

**`solutions/aicomputer_solution/multi_stream_asr.md`** - a two-stage speech pipeline whose allocation
is documented to the core: a shared `llama-server` with **`-np 4`** slots, **two encoder threads with
affinity `8;9`**, and **six decoder threads on cores `10,11,12,13,14,15`**. Its targets are stated as
well (online frame latency ~50 ms, offline RTF <= 0.4).

So the vendor's own answer to "how do several models share this board" is **multi-session inside one
process, with the stages pinned to disjoint cores** - which is the shape section 11 arrived at from the
measurements, and one lever section 11 never pulled: we sized the pools (thread counts) but never
pinned them.

## 14. Validation round: the scheme against those official numbers, on vision and on VLM

### 14.1 Small vision models: many small sessions beat one big one, and beat the published total

The vendor's application uses multiple sessions; section 11 measured two. The harness gained a `same`
mode (N identical sessions inside one process, 8/N AI threads each, run concurrently), and `yolov8n`
was swept over the session count - all eight cores used in every row:

| sessions x threads each | slowest inference | aggregate | against the published 134.29 FPS |
|---|---|---|---|
| 1 x 8 | 9.410 ms | 106.27 inf/s | -20.9 % |
| 2 x 4 | 16.854 ms | 118.66 inf/s | -11.6 % |
| 4 x 2 | 30.407 ms | 131.55 inf/s | -2.0 % |
| **8 x 1** | 52.807 ms | **151.50 inf/s** | **+12.8 %** |

(8 x 1 reproduced twice: 151.47 and 151.50. TCM sampled inside the concurrent phase: **8/8 blocks busy,
eight distinct thread ids**, one per session - the shape the configuration claims.)

Three things fall out of this table, and the third is a correction:

1. **the throughput shape is not the latency shape**: one 8-thread session has the lowest latency
   (9.41 ms) and the lowest aggregate (106.27/s); eight 1-thread sessions have 5.6x the latency and
   **+42 % the throughput**. A service picks one, and the vendor's application is the throughput kind;
2. **against the official number we are now ahead rather than behind**: 151.50 inf/s is **+12.8 %**
   over the published 12-stream total for the same model, and the 4 x 2 shape sits within 2 % of it.
   Before this round the honest position was that our best measured aggregate (106/s) was 21 % *below*
   the vendor's published total - which is exactly the "measuring in isolation" the operator warned
   about, and it was a real gap rather than a presentation problem;
3. **round 5's "concurrency is flat" is scoped, not wrong**: that measurement used one session with
   concurrent runs (`-c`) and two-process splits, and both of those are indeed flat or negative. The
   in-process multi-*session* shape is a third thing, and it is the one that scales - by 42 % over the
   single session, with the gain monotone in the session count. The deployment rule for throughput is
   therefore **N sessions of 8/N threads each**, not one fat session.

### 14.2 VLM: the vendor's own two-stage allocation does not transfer to the multi-slot service

Section 12 found that starving the VLM encoder to two threads costs 33 %. The research round then
supplied the strongest possible counter-argument: the vendor's own ASR solution *does* allocate 2 + 6,
and it pins the stages to disjoint cores (`8;9` and `10-15`) - the lever we had not tried. So it was
tested: the VLM's encoder at two threads with `spacemit_ep_intra_thread_affinity = "8;9"` and the text
model at `-t 6`, four slots, four simultaneous image requests.

| served shape | aggregate |
|---|---|
| encoder 8, text 8 (measured best) | **47.88 t/s** |
| encoder 2 + text 6, pinned to 8;9 | 32.06 t/s |
| encoder 2 + text 6, unpinned | 32.22 t/s |

The affinity pinning changes nothing (32.06 against 32.22), and the shape is still 33 % below the best
measured shape. The ASR rule therefore does not transfer, and the reason is structural rather than a
matter of tuning: in the ASR solution a *streaming* encoder runs continuously beside an offline
recogniser, so two cores for the encoder is a real division of labour; in the VLM service each of four
slots runs its own encode-then-decode, so the encoder is on the critical path of every request and
needs its width. **Both rules are correct for their own workload**, and this project now has both
measured with the official allocation as the thing being tested rather than assumed.
