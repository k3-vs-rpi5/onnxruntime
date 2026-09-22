# What to do next, and why: an analysis round (2026-09-22)

**This document is analysis, not measurement.** The operator stopped the round that was about to
sweep again and asked for the question that precedes it: *what should the next rounds actually be?*
So this round used **no board time** - no benchmark, no sweep, no optimisation, no model change,
nothing under `src/onnxruntime` - and every number below is a pointer to a measurement that is
already sealed, named by the report and the evidence directory that hold it. Where a claim is only a
bound or an assumption, it says so.

It answers three questions in order: **where we stand against the vendor's own material** (§1),
**which questions are already closed and must not be re-opened** (§2), and **which open questions
are one measurement away, what each costs, and what it would change** (§3-§6). §7 is the list of
decisions only the operator can make, and it closes the three stale questions `PLAN.md` has carried
since 2026-09-20.

Board state at the end of this round, read-only: TCM `8/8` free, no compute process running, thermal
zones 31-35 °C. Nothing was pushed to or executed on the board.

## 1. Where the project stands, in the terms that decide the next move

The only axis that matters for planning is: *does the vendor publish a row for this workload, and
where do we sit relative to it?* Three states are possible - reproduced, ahead/behind on a defined
statistic, or **no defined statistic to compare with** - and each state implies a different next
action.

| workload | vendor's own material | our standing | what it decides |
|---|---|---|---|
| vision, single stream, 7-8 models x 1/2/4/8 cores | published K3 table (ms) | **reproduced**: repeatedly inside -1.4 .. +2.5 %, one `vit_b_16` fp16 @8 outlier at +4.9 % | the published vision rows are usable acceptance targets; the remaining work is *coverage*, not reconciliation |
| vision, multi-stream throughput | 12-stream application table: `yolov8n` **134.29 FPS total**, `yolov8s` 77.74, `yolo26n` 104.18, ... | **ahead on a bound, not like-for-like**: our 8x1 inference aggregate is **151.50 inf/s (+12.8 %)** but the published number is a whole application (VPU decode + GPU preprocess + Qt UI), ours is a raw session loop | the one number where a like-for-like test would either harden the claim or falsify it (§3 G3) |
| vision, per-domain rows | `yolov8n-pose` 11.16 ms, `yolov8n-seg` 12.75 ms, `resnet50.b4` 15.38 ms, `mobilenet_v3_*`, `efficientnet_v2_s` | reproduced: `yolov8n-seg` 12.85, `yolov8n-pose` 11.195 (+0.3 %), `mobilenet_v3_small/large` (-1.1 %/+0.2 %), `efficientnet_v2_s` (+2.4 %) - **six independent reproductions in total**; depth (`yolo26n-depth` 64.84 ms, 18.4 % CPU-side head), OCR, face and the promptable pair now have baselines too; the batch-4 row is not measured here | coverage is now wide; the missing piece is a *composed* scenario, not more single models (§3 G5) |
| VLM encoder | published row at 4/8 cores: 256.47 / 164.50 ms | **ahead**: 243.08 ms (-5.2 %) and 150.02 ms (-8.8 %) | the only axis the published VLM row defines; it is closed in our favour |
| VLM end-to-end / service | **no published statistic** - the row's cell is undefined, and the candidate readings differ by 40 % | our own named statistics only: TTFT, end-to-end, 4-slot aggregate 47.88 t/s | requires our own acceptance basis to be *written down* as a contract (R10, §3 G4) |
| VLM decoder (text GGUF) | inherits the LLM rows | 22.26 .. 53.67 t/s over 1..8 threads; **no vendor row to attach** | same as LLM below |
| LLM prefill (context for the VLM decoder) | `pp128` rows | reproduced: -1.6 .. -2.6 % | usable as a reference |
| LLM decode (context for the VLM decoder) | `tg128` rows | **not reproducible on this board**: -8.0 .. -11.3 %, agreed by the vendored `0.1.1`, the vendored `0.1.7~a0+4` and our own fork build; OS image is the only remaining variable | published decode rows are context only; our own A/B decides (§3 G6) |
| ASR | RTF rows per model (`0.169` .. `0.689`), plus a vendor application runnable from a package | **unmeasured by this project** | empty domain with an official row *and* a vendor harness - cheap to fill if the scope widens |
| X100 as a co-processor | none | measured and closed: 97x behind the AI cluster on YOLO, 23-113x on other int8 graphs; pre/post only | do not spend another round here |
| compute efficiency (roofline) | `cpufp` on **K1** A60: 511.53 GOPS/core at 2 GHz | achieved 912 GOPS (yolov8n, 8 threads) = 6.1 % of a *quoted* ~15 TOPS, or 24.8 % of that K1 number clock-scaled to 1.8 GHz; matrix half alone ~1.8 TOPS | the denominator itself is the open question, and we already built the instrument to close it (§3 G1) |

Two readings of this table set the programme:

1. **Where a published row exists, we are at or ahead of it** on every vision and VLM-encoder row we
   have measured. The remaining deficits are not performance but *comparability*: the multi-stream
   total (a bound), the VLM service row (undefined statistic), and the LLM decode rows (the vendor's
   own binaries do not reproduce them here).
2. **Where we are behind, the cause is already located.** On YOLO it is not configuration (that is
   exhausted) and not the silicon's rated peak (we sit at 6-25 % of it) but the provider's execution:
   half the frame is non-matrix work and the matrix half runs at ~12 % of the dense-int8 peak
   (`YOLO-OPTIMISATION-20260921.md` §7-§8). That is a graph/provider property, and the levers that
   reach it are priced but out of contract (§4).

## 2. What is already decided - do not spend another round here

Each row below is a question that was open at some point and is now closed *with a measurement*. They
are listed because the cheapest round is the one that is not run.

| closed question | the measurement that closed it |
|---|---|
| "which session shape for latency?" | 1 session x 8 threads: `yolov8n` 9.410 ms frame / P99/P50 1.04, sustained 3x6,000 runs at 105.4 inf/s with no decay |
| "which shape for throughput?" | **N sessions x 8/N threads**: 8x1 = 151.50 inf/s against 1x8 = 106.27 (+42 % aggregate for 5.6x the latency); monotone in the session count; and the vendor's own application documents **eight parallel sessions by default** (`multi_stream_vision.md`, "AI inference"), i.e. their shipping default is our measured optimum |
| "which shape for a multi-slot VLM service?" | one server, 4 slots, encoder 8, text 8, requests queued = **47.88 t/s**; the single-stream optimum (2,6) loses 33 % in the real service; the vendor's ASR allocation (encoder pinned `8;9`, decoder 6) loses the same |
| "does splitting the eight cores help a single stream?" | no, in every measured form: 4x2 streams -11 %, 2x4 -37 %, `-c` concurrency flat, two-process splits flat |
| "is the second cluster a co-processor?" | no: 97x slower for YOLO, 23-113x for other int8 graphs; its sanctioned role is pre/post (0.33 ms of a 9.63 ms frame) |
| "is there thermals/decay headroom lost at 8 cores?" | no: 31->37 °C, flat 105.4/105.4/105.3 inf/s over 57 s runs; AI cores fixed 1.8 GHz (also their maximum) |
| "does the vendor's published vision tooling differ from ours?" | no in effect: tarball vs deb digests differ, but 60 tensors bit-identical and timings within 0.41 %; a future number only has to name which set it used |
| "is a single unsupported operator what fragments a region?" | disproved: the first accepted converter rule moved the boundary by exactly 3 nodes and left the 20.3 % fallback unmoved - the region did not merge (`OPERATOR-CONVERTER-20260922.md` §8); the discriminator is *context* (`fe7f8f8`) |
| "does the vendor's own simplifier close the fallbacks?" | no, measured: `onnxslim 0.1.87` changes neither op counts nor the re-census (15 CPU node executions, 18.7 % before and after), because a generic simplifier has no notion of the provider's capability check |
| "what does the OCR pipeline's first result mean?" | corrected: it was TCM contention, not a pipeline shape (`MODEL-ALTERNATIVES-20260922.md` §10.1); the corrected rule is that a split must be *searched* and follows thread scaling, not stage cost |
| "can the VLM's image stage be scheduled or cached to save the encoder?" | the encoder re-runs on every frame (30/30 in a loop); nine image-neutral reconfigurations were all neutral or worse; the published row is the encoder, so this is closed |

## 3. The open questions, priced - one measurement each

Seven questions, ordered by (value to the acceptance story) / (cost); the first six are priced and
the last (G7) is kept open deliberately. Every one is an *instrument or a validation*, never a
re-sweep of something §2 closed.

### G1. Measure the AI cluster's own int8 ceiling, on this board (the denominator) - **done 2026-09-22**

Result: `AI-CLUSTER-PEAK-20260922.md` (ceiling **2754 GOPS at eight threads**, 18.4 % of the quoted
peak; one core 893 GOPS with one TCM block; one block per compute thread at 1/2/4/8). The text below
is kept as the plan that was executed.

*Question.* The roofline section compares YOLO's 912 GOPS against a **quoted ~15 TOPS** for the K3
cluster and a **K1** `cpufp` number clock-scaled. What does *this* board's A100 cluster actually
deliver through *this* provider on pure matrix work?

*Why it matters.* It is the honest answer to "is the hardware at its limit?": it separates "the
provider is inefficient" from "the silicon is smaller than the quote". It also replaces the weakest
number in the whole project's argument, which is the one the operator challenged directly.

*Cost.* Low and already half-built: `make-ime-probe.py` generates pure-QDQ matrix probes
(`ime-probe-1024.onnx`, `ime-probe-2048.onnx`), and the run is one `perf_test` matrix over 1/2/4/8
threads. No model change, no contract change - the graph is an instrument, exactly like the memory
read-roof ladder.

*What it changes.* Turns §7 of `YOLO-OPTIMISATION-20260921.md` from "against a quoted denominator"
into "against a measured one", and gives the per-core efficiency of *this* core (the K1 figure is a
different product). If the measured peak lands far below 15 TOPS, the "provider is the limiter"
conclusion gets stronger, not weaker; if it lands near it, the 6.1 % becomes the headline number for
the vendor conversation.

### G2. Does the multi-session rule hold beyond one small model? - **done 2026-09-22**

Result: `MULTI-SESSION-SCALING-20260922.md` (`yolov8s` at `8x1` = **86.14 inf/s, +10.8 %** over the
published 77.74; `4x2` lands within 0.4 % of it on both models; the rule generalises, and the same
sweep produced vendor question 6 - a random session-creation failure).

*Question.* The throughput rule (N sessions x 8/N threads) was established on `yolov8n` alone. The
vendor's application applies the same shape to `yolov8s` and `yolov8m`.

*Why it matters.* If the rule is size-dependent, the deployment rule needs a qualifier, and so does
whatever we tell a customer about sizing an embodied stack. If it holds, the rule generalises and
the `yolov8s`-class models (16.65 ms published) get their own official anchor.

*Cost.* One hour: two points (1x8 and 8x1), not a sweep, on `yolov8s`; optionally the VLM encoder as
a third point, since it is the stage that competes with the decoder in the service rule.

*What it changes.* Nothing about the already-accepted rule; it decides how far the rule may be
extrapolated in the next report. Also produces the first `yolov8s` multi-stream comparison against
the published 77.74 FPS total.

### G3. Make the multi-stream comparison like-for-like, or state it as a bound forever

*Question.* Our +12.8 % is measured against a different thing than the vendor's number: their
134.29 FPS is a 12-stream application that decodes 1080p H.264 through the VPU, preprocesses on the
GPU, runs inference through eight sessions and displays through Qt; ours is eight bare sessions
in a loop. We have never run their application.

*Why it matters.* This is the single place where our most visible claim ("ahead of the published
total") is not method-aligned, and the operator's standing rule is precisely that numbers must not
flatter us. Two outcomes are both acceptable: their app on our board lands near their number (our
bound becomes a like-for-like win), or it lands lower here (the app's overhead is real and our bound
stands as a bound). What is not acceptable is continuing to quote it as if it were apples to apples.

*Options, priced.*

1. **Run the vendor's own application** (`github.com/spacemit-com/multi-stream-vision`, K3 Bianbu
   LXQt/GNOME is its supported platform; 12 x 1080p H.264 inputs). Strongest possible comparison -
   their code, their method, our board. Cost: a clone/build on the host, its model/video assets, and
   board-side dependencies (Qt5, OpenCL, MPP), i.e. the first thing in this list that may need
   operator approval for package installs.
2. **Replicate the loop shape without the UI/VPU**: 12 channels of one model, latest-only queue,
   round-robin over eight sessions, measured with our harness. Cheap (the `same` mode of
   `ort-two-session-pipeline.cpp` is most of it), but it only *bounds* the app cost from below.
3. **Keep it as a bound** and say so in one sentence wherever the number appears.

*Recommendation.* (2) now, (1) only if the operator wants the app result itself; (3) is the fallback
for any report published before either runs.

### G4. Write the VLM service acceptance basis (a contract, not a measurement) - **done 2026-09-22**

Result: `VLM-ACCEPTANCE-BASIS-20260922.md` - the named statistics (S1-S7), the rules that follow, and
the acceptance procedure for a change on this path.

*Question.* The published VLM row has no named statistic (§7.2 of `METHOD-ALIGNMENT-20260921.md`),
so R10 makes it an order-of-magnitude reference and leaves "what we hold ourselves to" undefined.

*Why it matters.* Both current project targets (ONNX, VLM) end in service-shaped claims, and a
service claim without a named statistic is the one thing the operator's "don't celebrate alone" rule
cannot survive.

*Cost.* None in board time: the numbers already exist (encoder-only ms per width, TTFT, end-to-end
with and without cache, 4-slot aggregate t/s, per-new-frame cost). What is missing is the sentence
that says which of them is *the* acceptance basis and which are diagnostics.

*What it changes.* Every future VLM number would carry its official-row delta when one exists and
its own named statistic when one does not - the same structure the vision tables already use.

### G5. One composed embodied/perception scenario, under the three deployment rules

*Question.* The project has measured *models* and *two-stage pipelines*, but no composed scenario of
the kind the VLM/embodied direction implies: e.g. detector + pose/depth + a language stage, sharing
eight cores.

*Why it matters.* "VLM + embodied" is the stated near-term target, and the rules we would apply
(three shapes, TCM contention, split search, no X100 offload) have never been exercised as a whole
on a scenario that a customer would recognise.

*Cost.* Lower than it looks: nearly every building block already has a baseline - detector
(`yolov8n` 9.54 ms), pose (`yolov8n-pose` 11.195 ms against the published 11.16), depth
(`yolo26n-depth` 64.84 ms with its 18.4 % CPU-side head), face (`arcface` 3.137 ms), OCR
(det 35.99 + rec 2.36 ms), semantic segmentation (`yolo26n-sem` 52.26 ms) and the promptable pair
(`mobilesam` encoder 40.30 ms). The only building block the menu still lists without a measurement
is tracking (`bytetrack`/`ocsort`/`nanotrack`). So the cost is one pipeline session plus, if the
chosen scenario needs it, a third stage on a tool that already runs two.

*What it changes.* Converts a menu into a demonstration with an official row where one exists, and
exercises the TCM table as the arbiter (8/8 blocks named by pid).

### G6. The LLM decode rows: pursue or permanently demote

*Question.* `tg128` is -8 .. -11 % on this board with the vendor's own binaries, and the only
remaining variable is the OS image (`4.0rc3` at publication vs `4.0.6` here).

*Why it matters.* It is the one official row we cannot reproduce, and it contaminates any VLM
end-to-end claim, because the VLM's decode stage is an LLM.

*Cost.* Re-imaging the board is a human operation with real cost and risk, and it would invalidate
the current board state as an evidence base. The alternative costs nothing: keep the decode rows as
context, keep our own A/B as the acceptance instrument, and say so (already the rule).

*Recommendation.* Demote permanently unless a customer deliverable depends on the published decode
row; do not re-image for it.

### G7 (kept open, not priced). `/dev/tcm_sync_mem`

Every provider run logs that it cannot open `/dev/tcm_sync_mem` and falls back to heap barriers; the
device node does not exist on this board (only `/dev/tcm` does), and it is referenced by the vendor's
own code. The non-scaling part of a YOLO frame is ~4.2 ms (s in `T(s+p/n)`), and barriers are one
candidate. This needs a board-provisioning decision (a device node we would have to create
ourselves), and the size of the prize is untested, so it stays a *question with a number attached*
until the operator decides whether board provisioning may be touched at all. A cheaper first step,
if the question becomes live: ask the vendor what the node is and whether the heap fallback is
expected on this image (it is question-shaped, not measurement-shaped).

## 4. What is not ours to decide or do

**Vendor-side, already asked** (`VENDOR-QUERY-20260922.md`, five items, each with a minimal
reproduction): isolated supported operators refused by the connectivity rule; a reconnected region
that came out 16 % *slower*; a `Gather`-shaped crash; the provider `MatMul` path 6.9x slower than the
CPU EP; a second execution crash with no diagnostic output. Nothing in §3 depends on these answers,
which is why they can wait.

**Priced but out of contract** (model or graph changes; recorded so nobody re-derives them):

| lever | measured price | gate |
|---|---|---|
| batch-N export (`resnet50.b4` 15.608 ms for four frames against 4x5.748) | **1.41-1.47x per frame** | re-export = model change |
| removing `yolo26n`'s 16 CPU-side nodes | up to **8.2 %** of that model's frame | graph edit |
| the segmentation head's transposes (`yolov8n-seg`, 15 % of the frame) | up to **15 %** on that model | graph edit |
| removing the non-matrix half entirely | ~**2x** family-wide | not achievable inside the provider |
| `OneHot` -> `Gather` rule already derived | ceiling **0.06 ms** (+ a boundary move) | not worth the machinery, and the fragmentation hypothesis is already disproved |

**Board provisioning** (human): the TCM device node (G7), re-imaging (G6), X100 governor
(2.0 vs 2.15 GHz = 7.5 % of a stage that is 3.4 % of a frame - skip).

**Approval gates from the workspace contract** that any §3 item may touch: package installs on the
board (G3 option 1), a new tools project admission (the converter already lives in this project;
nothing new is proposed), and the first push of anything.

## 5. The recommended next round, in order

1. **G1** - build the IME probe, run the 1/2/4/8-thread matrix, seal it as an instrument result.
   *No approval needed; one board session; the single highest-value number currently missing.*
2. **G2** - two-point multi-session check on `yolov8s`; attach the published 16.65 ms and 77.74 FPS
   rows.
3. **G4** - write the VLM acceptance basis (documentation only), using the existing numbers.
4. **G3 option 2** - 12-channel loop replication, so the multi-stream sentence has a
   shape-matched number under it.
5. **G5** - one composed scenario (pose or depth + detector + language stage), run under the three
   deployment rules with the TCM table sampled.
6. **Housekeeping** - rewrite `PLAN.md` to the state this document describes; it still carries three
   open questions from 2026-09-20 that the rounds have since answered (§7 D1-D3).

**Not in the next round, deliberately**: any further YOLO session/provider/affinity sweep (closed in
§2), any model- or graph-change lever without an explicit contract decision, any re-image, and any
new measurement whose official row does not exist *and* whose own named statistic has not been
written down first.

## 6. What each item does to "are we comparing with the official numbers?"

| item | attaches to | before | after |
|---|---|---|---|
| G1 | none (own instrument; relates to the quoted ~15 TOPS) | "6.1 % of a quoted peak" | "x % of a peak measured on this board through this provider" |
| G2 | published `yolov8s` 16.65 ms / 77.74 FPS total | rule from one model | rule with a size qualifier (or a generalisation) |
| G3 | published 134.29 FPS total | a bound quoted against an application | a shape-matched number, or the app itself |
| G4 | published VLM row (undefined statistic) | our numbers with no contract around them | our own named statistic, with the official row as context |
| G5 | published `yolov8n-pose` 11.16 ms (for the pose variant) | isolated models | a composed scenario with at least one official anchor |
| G6 | published `tg128` rows | "not reproducible, cause unknown-but-named" | a standing rule: context only, our A/B decides |

## 7. Decisions requested from the operator

The first three close questions `PLAN.md` has carried since the planning round; the rest are new.

1. **D1 - acceptance target (old question 1).** Proposed closure: the acceptance target is the
   vendor's own catalogue and its published rows (vision tables + VLM encoder), plus one composed
   scenario (G5). Any specific customer model would arrive as an addition, not a replacement.
2. **D2 - instrument (old question 2).** Proposed closure: the vendor's own binaries with the
   vendor's flags, plus this project's censuses/gates; building the fork's own
   `onnxruntime_perf_test` is not proposed, since the shipped one already reproduces the rows.
3. **D3 - the two-cluster question (old question 3).** Proposed closure: settled by measurement -
   the X100 is not a co-processor for these graphs (97x), so the project keeps the rule "AI cluster
   for inference, X100 for pre/post with a declared interference budget (10-25 %)".
4. **D4 - may the vendor's multi-stream application be fetched and run on the board (G3 option 1)?**
   It is the strongest available official comparison and the only item here that may need board
   package installs.
5. **D5 - the VLM acceptance basis (G4).** Confirm: encoder-only ms as the official-comparable
   statistic, plus a service statistic of our own definition (4-slot aggregate t/s, per-new-frame
   cost, TTFT), stated with cache state in every number.
6. **D6 - which composed scenario for the next phase (G5): pose, depth, or detector + language?**
   The pose variant is the only one with a published row today.
7. **D7 - the contract boundary.** Confirm it stays: model, quantisation, weight layout and graph
   edits remain out of scope as *optimisations*, while the substitution/adoption track may choose a
   different model for a domain (that track's own document lives in
   `MODEL-ALTERNATIVES-20260922.md`, and its candidates are priced in §4 above).
8. **D8 - LLM decode rows (G6).** Confirm the demotion to context, or ask for the re-image.

## Appendix: pointers

* Standing measurement contract with the official rows inline:
  `METHOD-ALIGNMENT-20260921.md` (vendor methods verbatim §2, published K3 tables §3, our
  reproduction §4, aspect-by-aspect alignment §5, R1-R12 §6, the two unreconciled rows §7).
* The YOLO programme and the roofline: `YOLO-OPTIMISATION-20260921.md` §6-§8.
* The VLM service shape and the official-row reconciliation:
  `VLM-SERVICE-SHAPE-20260921.md`, `MODEL-ALTERNATIVES-20260922.md` §12-§14.
* The converter and its rule registry: `OPERATOR-CONVERTER-20260922.md`.
* Vendor questions with reproductions: `VENDOR-QUERY-20260922.md`.
* Evidence root: `/root/k3-evidence/onnxruntime/baseline-20260921/` (README with the per-file
  digest registry; `yolo-traces/`, `yolo-ep-options/`, `vlm-service-shape/`, `vlm-clocks/`,
  `coverage-census/`, `coverage-census-zoo/`, `session-options/`, `modelzoo-vision/`,
  `modelzoo-vlm/`, `sustained-yolo.txt`, `llm-packaged.txt`).
* Instruments referenced by §3: `.local/onnx-baseline/make-ime-probe.py` and its generated
  `ime-probe-{1024,2048}.onnx` (G1, generated 2026-09-22, **never run**; local, not in git),
  `tools/ort-two-session-pipeline.cpp` (`same` multi-session mode, G2),
  `tools/board-coverage-census.sh` + `tools/ort-provider-census.py` (G5 pre-check),
  `tools/ep-op-breakdown.py` (operator census).
