# Scenario testing and validation: the scheme this project holds itself to

Asked for by the operator on 2026-09-22, and written before the first scenario was run under it. A
**scenario** here is a deployment shape with a workload and a stated acceptance basis - not a
benchmark, not a model, and not an optimisation. The scheme exists so that any scenario, including
ones added later, is measured the same way and its claim is checkable from its own evidence package.

## 1. The five-step ladder: every scenario passes all five

| step | what it establishes | instrument | what it must show |
|---|---|---|---|
| **L1 engagement** | the work really ran on the AI cluster, on a clean board | `spacemit-tcm-smi` sampled in flight, `ps -L` | blocks busy naming the workload, one block per compute thread, board left 8/8 free |
| **L2 correctness** | the scenario computes what it claims | tensor-dump equivalence for precision/layout changes; identical answers on a fixed input set for scheduling changes | bit-identical, or a stated tolerance with the reason |
| **L3 standalone baseline** | every stage's own cost, before composition | `onnxruntime_perf_test` (vendor flags) / `llama-bench` / the vendor's own tools | mean and P50/P90 for each stage, with its width |
| **L4 composition** | the scenario's own shape, chosen by which of the three deployment rules applies | `ort-two-session-pipeline`, the vendor application, the VLM service harness | latency, throughput, or served rate for the composed shape, plus each stage's cost *inside* it |
| **L5 comparison** | the claim is not made in isolation | the published row when one exists; our own named statistic otherwise | the official delta, or an explicit "no published row exists" |

The three deployment rules the scheme applies at L4 (all measured, see `PLAN.md`):

* **latency** - one session, eight threads;
* **throughput** - N sessions of 8/N threads each (`4x2` matches the vendor's own application shape,
  `8x1` is 9-11 % above it);
* **service** - one server, four slots, encoder 8, text 8, requests queued.

The classes, and what counts as an anchor for each:

| class | shape | acceptance basis |
|---|---|---|
| **C1 perception, single stream** | 1 session x 8 threads | the published single-stream rows (reproduced within ±2.5 %) |
| **C2 multi-stream analytics** | N sessions x 8/N threads or the vendor application | the published application totals (reproduced within ±2.6 % by their own application, `VENDOR-APP-20260922.md`) |
| **C3 pipeline, multi-model single stream** | two stages in one process, serial vs overlapped, split searched | **no published row**: own statistics - frame latency, fps - with the split that was measured |
| **C4 multi-slot service (VLM)** | one server, N slots, encoder/text widths | encoder-only latency against the published row; the served statistics are ours (`VLM-ACCEPTANCE-BASIS-20260922.md` S1-S7) |
| **C5 hybrid: perception beside a service** | C2/C3 stage sharing the board with C4 | **no published row**: the interference matrix - each side alone, both together |

## 2. The evidence contract per scenario

One directory under `/root/k3-evidence/onnxruntime/<scenario>-<date>/`, with:

* a README that states the question, the instruments, the results and the caveats;
* every raw log, the server logs and the TCM samples;
* the board state before and after (governor, frequency, temperature, free blocks);
* a per-file sha256 registry with a self-check that must come out 0 mismatched / 0 missing /
  0 unlisted;
* **failed and invalid runs kept**, labelled as such, with the reason - they are how the next
  person avoids the same trap.

## 3. Rules of engagement

1. **Clean board first.** The harness refuses to start unless no inference process runs and the
   TCM table reads 8/8. A killed inference process can leak all eight blocks (measured with the
   vendor's application), and the documented remedy `spacemit-tcm-smi -c` is applied only with
   nothing running and is reported.
2. **Never measure through a shared resource you have not re-checked.** Two EP processes do not
   share this board (the first takes all eight blocks); scenarios that need two stages run inside
   one process, or the second one is measured and declared as contention.
3. **The official row goes next to the number, or the number says it has none.** Same rule as
   everywhere else in this project.
4. **A scenario result is a measurement, not a tuning.** Any parameter change is one variable, with
   the before-number stated; a scenario is never used to smuggle in an unapproved optimisation.
5. **Steady state is defined before it is quoted.** The vendor application needs ~100 s per channel
   before its per-second series is flat; every run states its window, and medians are taken over a
   named number of samples.

## 4. The scenario matrix as it stands today

| # | scenario | class | state | headline numbers |
|---|---|---|---|---|
| S1 | single-frame detection (`yolov8n`, 640²) | C1 | measured | 9.41-9.54 ms, P99/P50 1.04, sustained 105.4 inf/s with no decay |
| S2 | 12-stream analytics, our sessions | C2 | measured | `4x2` = 134.82 / 77.43 inf/s (`yolov8n`/`yolov8s`), `8x1` = 146.62 / 86.14 |
| S3 | 12-stream analytics, vendor application | C2 | measured | 130.85 FPS shipped config; 136.92 with their decode-throttle note; `yolov8s` 79.15 |
| S4 | detect -> pose pipeline | C3 | measured | 43.10 fps overlapped 4+4 against 27.76 serial; split must be searched |
| S5 | OCR pipeline (det + rec) | C3 | measured | contention-resolved; TCM explains the spread |
| S6 | VLM 4-slot service | C4 | measured | 47.91 t/s cold-image 4x32 tokens; 68.65 first round and ~85.8 cached rounds at 4x64 tokens |
| S7 | **perception beside the VLM service** | C5 | measured 2026-09-22 - `SCENARIO-SENSE-LANGUAGE-20260922.md` | VLM cold-image round -27.7 %, cached rounds -31.9 %, detector +115 % latency; the last round (detector finished) returns to the baseline to the digit |
| S8 | embodied stack with depth/pose + language | C5 | **next** - needs a scenario decision (D6) and the depth/pose candidates already on the board |

## 5. What a completed scenario claim looks like

One sentence, and its parts: *"In scenario X (class, shape), stage A runs at N ms and stage B at
M ms; composed under rule R the scenario delivers T (units) at P50/P99; against the published row
for the same quantity it is +/-D %, and where no row exists the statistic is our own; the AI cluster
was engaged (TCM, pid) and the board was clean before and after; evidence: path, digests, self-check
clean."*

Nothing shorter is accepted, here or in any report this project publishes.
