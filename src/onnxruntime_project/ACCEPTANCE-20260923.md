# Full acceptance record, 2026-09-23

The consolidated verdict on every scenario the programme has run, with the numbers that decide it,
the evidence package that holds it, and an explicit list of what is *not* accepted. Written to the
scheme's own rules (`SCENARIO-VALIDATION-20260922.md`, `SCENARIO-PROGRAMME-20260922.md`): shape is
part of the claim, two-stage scenarios are reported as matrices, every card carries a negative
control, quality gates are as binding as speed, and nothing is accepted on a number that has no
evidence package behind it.

## 1. Verdicts

| # | scenario | verdict | the numbers that decide it | evidence |
|---|---|---|---|---|
| S1 | single-stream detection | **accepted** | yolov8n 9.41-9.54 ms, P99/P50 1.04, sustained 105.4 inf/s | `baseline-20260921/`, `sustained-yolo.txt` |
| S2 | multi-stream, our sessions | **accepted** | `4x2` 134.82 / 77.43 inf/s, **`8x1` 146.62 / 86.14** (+9-11 % over the published totals) | `multisession-20260922/` |
| S3 | vendor multi-stream application | **accepted** | 130.85 FPS shipped config, 136.92 throttled, yolov8s 79.15 - published rows within **±2.6 %** | `vendor-app-20260922/` |
| S4 | detect->pose pipeline | **accepted** | 43.10 fps overlapped 4+4 against 27.76 serial | `model-alternatives-round2/` |
| S5 | OCR pipeline (det+rec) | **accepted with defect** | stage costs 35.99 / 2.36 ms; contention explained by the TCM table | `model-alternatives/` |
| S6 | VLM 4-slot service | **accepted** | 47.91 t/s (4x32 tokens); 68.65 / ~85.8 t/s (4x64); single-stream (2,6) loses 33 % in service | `vlm-service-shape/` |
| S7 | perception beside the service | **accepted** | service -27.7 % / -31.9 %, detector +115 %, recovery to baseline after the detector stops | `scenario-sense-language-20260922/` |
| S8 | embodied stack plus service | **accepted** | frame -14.6 %, detector stage +236 %, service -39 % / -47 % | `scenario-embodied-20260922/` |
| S9 | RAG over the project's corpus, self-chain | **accepted** | recall@1/5/10 **1.000** verbatim / **0.980** shuffled; TTFT 2,226 ms; citations **16/20** | `s9-rag-20260922/` |
| S9.3 | RAG, vendor stack | **driven, not accepted** | with the intended model: latency 1.8 s, **grounding 0/5** (fabricated values) | this document's §3 |
| S10 | embodied decision | **accepted with a declared operating point** | resolution sweep: **256x256 = 94/100 at 45.7 ms** (latency met), **384x384 = 95/100 at 94.0 ms** (accuracy met), 512x512 no further gain; VLM-only baseline 3/12 | `s10-embodied-20260923/` |
| S11 | photo album | **accepted** | NMI **0.881**, purity 0.865; same-video cosine 0.954 vs cross-video 0.470; verbatim duplicates **P/R 1.000** | `s11-natural-20260922/` |
| S12 | document ingestion | **accepted** | printed CER **0.0001** (control), tables **0.0053** against tesseract 0.0188, scanned 0.0061; **3-5 pages/s** | `s12-fixture-20260922/` |
| S13 | audio | **not started** | - | - |

## 2. Optimisation results that back the verdicts

| lever | measured effect | deployment reading |
|---|---|---|
| decode under N slots | 6.13 -> 7.88 -> **9.43 t/s** at 1/2/4 | use slots for served rate |
| prefill under N slots | 5.35 -> 10.4 -> 18-20 s | does not parallelise |
| retrieved context 1 -> 4 chunks | TTFT 914 -> 2,592 ms | keep k small |
| prefix caching (verified server-side) | `prompt_n` 121 -> 1, prefill 1,037 -> **78 ms** | cache only genuinely repeated prefixes |
| embedding under concurrency | 51.2 -> 55.2 chunks/s (+7.7 %) | feed as one stream |
| OCR batch-4 recogniser | 213-316 -> **650-904 ms** | **rejected** |
| citation integrity by prompt | 16/20 -> **14/20** | enforce outside the model |
| multi-session throughput shape | `8x1` +39-42 % over one fat session | throughput shape for analytics |
| cluster ceiling (instrument) | 2,754 GOPS at eight threads, 18.4 % of the quoted peak | the runtime, not the silicon, is the limiter |

## 3. Not accepted, and why

1. **S9.3 vendor RAG grounding.** With the intended nomic model the latency problem disappeared
   (67.1 s -> 1.8 s median, i.e. the first figure was our substitution's artefact), but the answers
   remained fabricated against the corpus (0/5 on questions whose answers are specific numbers).
   One measurement is outstanding and it is the only thing that could soften this: wait for the
   knowledge base to report 38/38 vectorised files, then re-ask the same five questions.
2. **S10 accuracy target.** 94/100 against a >= 95 % target - one scene short, with six misses not yet
   attributed. The latency target is met (46.6 ms) and both rejected baselines stay in the record
   (VLM-only 3/12; saturation-threshold prototype 23/100 with 45 segmentation failures).
3. **OCR batch-4** as a lever: measured, negative, rejected.
4. **S13 audio**: no measurement at all, so no claim.

## 4. Threats to validity

* one board, one OS revision (Bianbu 4.0.6 against the published rows' 4.0rc3);
* Python on this board reaches only the CPU EP, so every AI-cluster number comes from a C++ harness
  or the vendor tooling - stated with each such number;
* one fixture per class (30 pages, 100 scenes, 319 natural frames, 2,226 chunks): sealed and
  deterministic, but not a random sample of production data;
* two synthetic fixtures for S11 were built, failed and are kept - the scenario only became
  measurable on natural material, which is itself a finding;
* the vendor stack was first measured with a substituted embedding model; that run is kept and its
  numbers are labelled as such rather than deleted.

## 5. What would complete the programme

1. S9.3 re-ask on a complete index (one run, five questions).
2. S10 attribution of the six misses, then either the target or the fixture tolerance is revisited -
   the decision is the operator's, not the measurement's.
3. S13 only if the audio domain is wanted; the fixture and the oracle (`whisper.cpp`) are specified.
