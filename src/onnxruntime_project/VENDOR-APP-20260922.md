# The like-for-like multi-stream comparison: the vendor's own application, on this board

This is item **G3** of `NEXT-ROUND-ANALYSIS-20260922.md`, in its strongest form. The project's
"ahead of the published total" claim had a method gap: the published number is an *application*
(VPU decode, GPU OpenCL pre-processing, eight ONNX Runtime sessions, Qt UI), while ours was a bare
session loop. The operator authorised fetching and running that application, so the gap is now
closed by running their code on our board rather than by arguing about it.

Evidence: `/root/k3-evidence/onnxruntime/vendor-app-20260922/` (11 files, per-file sha256 registry,
self-check clean). Instrument: `tools/board-vendor-app.sh`.

## 1. Getting the application: the documented source path does not exist

The application is `yolo-demo`, and its documentation offers two install paths
(`docs-ai/solutions/aicomputer_solution/multi_stream_vision.md`):

* **source** - `git clone https://github.com/spacemit-com/multi-stream-vision.git`: measured
  2026-09-22, that repository **404s** (the GitHub API returns `Not Found` while
  `spacemit-com/docs-ai` returns 200). The path in their own documentation is dead;
* **apt** (their recommended one) - `apt-get install -y yolo-demo`: version **1.0.2** from the
  Bianbu 4 archive. This is what was used.

The package ships the application, the default config, **seven 1080p sample videos**, and three
models (`yolov8n/s/m.q.onnx`). It runs without a display: `QT_QPA_PLATFORM=offscreen` plus
`display.mode: stats`, and it prints the very numbers its UI would show as
`INF PERF ch=N fps=...` lines, one per channel per second.

## 2. Results: their own totals, reproduced within ±2.6 %

12 streams, `infer_fps 25`, eight sessions x one thread (the shipped default), steady state taken as
the median of the **last 60 samples per channel**:

| run | model, from the app's own banner | per channel | **total** | against the published row |
|---|---|---|---|---|
| shipped config, `decode_fps 25` | `yolov8n` | 10.90 FPS | **130.85 FPS** | 134.29 -> **-2.6 %** |
| `decode_fps 13` (the vendor's own tuning note) | `yolov8n` | 11.41 FPS | **136.92 FPS** | 134.29 -> **+2.0 %** |
| `model.id: yolov8s`, full length | `yolov8s` | 6.60 FPS | **79.15 FPS** | 77.74 -> **+1.8 %** |

Three readings:

1. **the published application rows are what this board does.** Two models, two very different
   compute costs, reproduced inside ±2.6 % by their own code with their own sample videos. Together
   with the six single-stream reproductions this project already holds, the vision side of the
   vendor's table is now anchored from two directions;
2. **the tuning note in their config is real and worth 4.6 %.** Their comment says that decoding at
   the full source rate caps the inference rate at about nine frames per channel and that lowering
   `decode_fps` to 13-15 buys inference throughput "with CPU". Measured: 10.90 -> 11.41 FPS per
   channel (+4.6 %) when decode is throttled to 13. Both numbers are above their note's "~9 per
   channel" because the board is not running the Qt UI in stats mode;
3. **the same config file independently states this project's throughput rule**:
   `num_sessions: 8`, `intra_threads: 1`, with the comment `num_sessions * intra_threads <= 8(TCM
   块数)` and "the recommended default is eight sessions with one thread per session". The rule the
   projects measured from the outside is the rule the vendor ships.

## 3. The comparison that is now like-for-like

| shape | `yolov8n` | `yolov8s` |
|---|---|---|
| the vendor's application, 12 streams, shipped config | 130.85 FPS | 79.15 FPS |
| the project's bare sessions at the same shape (8 sessions x 1 thread) | **146.62 inf/s** | **86.14 inf/s** |
| difference | **+12.0 %** | **+8.8 %** |

The bare-session aggregate runs **above the whole application's total**, and the application's total
includes the decode, pre-processing and display work that the bare sessions do not do. This is the
statement the project can now make: at the vendor's own shape, on the vendor's own application, the
inference stage alone is 9-12 % above the full application's end-to-end total, and the full
application itself reproduces its published row. Section 14.1's earlier "+12.8 % against a bound" is
superseded by this measurement.

## 4. Operational findings (each cost time, so each is recorded)

* **a killed `yolo-demo` leaks TCM.** SIGKILL left all eight blocks allocated to thread ids that no
  longer existed, after which every start failed with `ERR FileSource[0] Failed to init MPP
  pipeline`; even a clean SIGTERM left one block behind twice out of three. The vendor's own
  documentation gives the remedy (`spacemit-tcm-smi -c`, only with no inference running), and the
  harness now refuses to start on a dirty board and reports the clear;
* **`-m <model>` does not switch the model** - the config's `model.id` wins, and the only evidence is
  the banner (`ModelManager: 9 models available, current=...`). The first `yolov8s` attempt silently
  ran `yolov8n` and is kept in the evidence as an invalid row;
* **a short run reads low.** The per-channel series is still climbing at 100 s; an ad-hoc look at the
  first run's tail read 7.3 FPS/channel against a steady-state 10.90. Every number above is the
  steady-state median, with the raw per-second series in the logs.

## 5. What this changes in the plan

G3 is closed in its strongest form: the comparison is no longer a bound, and the official
multi-stream rows are reproduced by the vendor's own application on this board. Two follow-ups are
carried rather than assumed: the **dead source URL** and the **model-selection behaviour** become
part of the vendor query (documentation-level questions), and the TCM leak becomes a rule of the
scenario harness - it is the first thing that must be checked before any vendor-application run.
