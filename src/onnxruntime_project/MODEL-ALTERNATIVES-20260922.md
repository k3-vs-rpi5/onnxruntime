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
