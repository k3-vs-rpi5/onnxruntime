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
