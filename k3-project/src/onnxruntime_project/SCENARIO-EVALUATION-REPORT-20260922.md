# Scenario evaluation report, 2026-09-22

The summary of the scenario programme to date, written the way an evaluation report is normally
structured: scope, system under test, method, results, threats to validity, and what remains open.
Every number comes from a sealed evidence package; nothing here is projected or estimated.

## 1. Scope and system under test

* **board**: SpacemiT K3 (X100 cpu0-7 + A100 cpu8-15), Bianbu 4.0.6, kernel 6.18.3; 31 GB RAM,
  68 GB free disk; AI cores fixed at 1.8 GHz, X100 at 2.0 of 2.15 GHz.
* **runtime**: SpacemiT ONNX Runtime 2.0.6 (`libspacemit_ep.so`, release-tarball set in
  `/tmp/ort206`), llama.cpp `0.1.7~a0+4` (build `5a23f07a4`) with the vendor backend; Python 3.14
  with the same ORT for *glue only* - its provider list is `CPUExecutionProvider` alone, measured.
* **scenarios evaluated here**: document ingestion (S12, new), RAG over a knowledge base (S9, new),
  plus the previously closed S1-S8 (single-stream, multi-stream, vendor application, pipelines, VLM
  service, perception-beside-service, embodied stack).

## 2. Method (scheme v2, `SCENARIO-VALIDATION-20260922.md` + `SCENARIO-PROGRAMME-20260922.md`)

* every scenario passes the ladder **L1 engagement → L2 correctness → L3 standalone → L4
  composition → L5 comparison → L6 reference → L7 stability**; L2b is the task-quality gate;
* **shape is part of the claim** (latency / throughput / service), a two-stage scenario is reported
  as a matrix, and every card carries a **negative control**;
* AI-cluster claims are supported by `spacemit-tcm-smi` naming the workload, never by per-CPU
  counters; the board is verified clean (8/8 blocks) before and after;
* comparison is against **our own accepted baselines** (the operator's decision); no host-side x86
  reference is required.

## 3. Results

### 3.1 S12 document ingestion - **closed, band met**

| class | pages | our chain CER mean (max) | our WER mean | tesseract CER mean | verdict |
|---|---|---|---|---|---|
| printed (negative control) | 10 | **0.0001 (0.0008)** | 0.0006 | 0.0000 | pass (band <= 0.01) |
| scanned | 10 | 0.0061 (0.0114) | 0.0803 | 0.0020 | within band, gap stated |
| tables | 10 | **0.0053 (0.0107)** | 0.0208 | 0.0188 | **beats the reference** |

Cost per page: detector 56-87 ms, DB post-processing 2-3 ms, recognition 123-270 ms, whole page
**213-330 ms (3-5 pages/s)** with both sessions on the cluster at 4+4 threads. Five failed or
invalid runs are kept with their causes (two fixture-layout bugs, one wrong-decode, one
chunk-overlap, one letter-gap cut).

### 3.2 S9 RAG over the project's own corpus - **stages 1-2 measured, stage 3 mapped**

| stage | metric | result |
|---|---|---|
| retrieval | corpus / chunks | 38 files, 531 KB / 2,226 chunks (<=300 chars, hard cap for the 512-token server context) |
| retrieval | embedding throughput | **17.7 ms/chunk** (BGE-Small-Zh, `llama-server --embeddings`) |
| retrieval | recall@1 / @5 / @10, verbatim query | **1.000 / 1.000 / 1.000** |
| retrieval | recall@1 / @5 / @10, word-shuffled control | **0.980** (one miss, reported) |
| answer | TTFT median / P90 | **2,226 ms / 3,887 ms** |
| answer | total median (96 tokens) | **9,363 ms** |
| answer | citation correct (points at the source chunk) | **16/20** |
| answer | non-empty answers | 20/20 |
| reference | vendor `llm-sdk` 0.0.39 | installed, service active on `:8050`, knowledge-base API mapped; corpus not yet loaded |

### 3.3 Earlier scenarios (unchanged, for the whole picture)

| scenario | headline |
|---|---|
| S1 single-stream detection | yolov8n 9.41-9.54 ms, P99/P50 1.04, sustained 105.4 inf/s |
| S2 multi-stream (ours) | `4x2` = 134.82 / 77.43 inf/s; **`8x1` = 146.62 / 86.14** (+9-11 % over the published totals) |
| S3 vendor application | 12 streams: 130.85 FPS shipped config, 136.92 with their decode throttle, yolov8s 79.15 - published rows reproduced within ±2.6 % |
| S4-S5 pipelines | detect→pose 43.10 fps overlapped vs 27.76 serial; OCR pair contention explained by TCM |
| S6 VLM service | 4 slots: 47.91 t/s (32 tokens), 68.65 / ~85.8 (64 tokens); single-stream 2+6 loses 33 % in the service |
| S7 perception + service | service -27.7 % / -31.9 %, detector +115 %, recovery to baseline after the detector stops |
| S8 embodied stack + service | frame -14.6 %, detector stage +236 %, service -39 % / -47 % |

## 4. Threats to validity (stated, not buried)

1. **one board, one OS revision** (Bianbu 4.0.6); the official rows were measured on 4.0rc3, which is
   the named cause of the LLM decode-row mismatch (the vendor's own binaries agree with each other,
   not with the published number).
2. **Python is glue only**: any Python ONNX Runtime number on this board is a CPU-EP number; every
   AI-cluster claim here comes from C++ harnesses that load the provider explicitly, or from the
   vendor's own tooling.
3. **one fixture per class** (30 pages, 1,000 images, 2,226 chunks). The fixtures are sealed and
   deterministic, but they are not a random sample of production documents or photos.
4. **negative controls are the guard**: printed pages must read ~0, shuffled-word queries must not
   collapse, scanned pages must not read 0 - each is reported, including failures.
5. **the reference application for RAG is installed but not yet fed the corpus**, so S9's reference
   column is empty by intent rather than filled with a different configuration.

## 5. Conclusions

* the ingestion chain (S12) is **accepted**: it meets its band on the negative control, beats the
  independent engine on tables, and is characterised in cost and in five kept failures;
* the RAG retrieval stage (S9.1) is **strong and measured** (perfect verbatim recall, 0.98 on a
  word-order stress test); the answer stage is **usable but not yet accepted** - 16/20 citations
  correct, with invented citation indices (an answer citing `[4]` when three sources were supplied)
  recorded as a failure mode rather than hidden;
* the embodied and service scenarios (S7, S8) have delivered the programme's most actionable result:
  the eight AI cores are a shared, exhaustible resource, and the cost of sharing is measured on both
  sides;
* nothing in this report claims a win that is not accompanied by either a published-row delta or a
  named statistic of our own.

## 6. Open, in order

1. **S9.3** - load the sealed corpus into the vendor `llm-sdk` knowledge base and compare it with the
   self-chain statistic for statistic (recall, TTFT, citation correctness).
2. **S9 answer quality** - the citation failures are a model-behaviour gate: next candidates are a
   stricter prompt with an explicit "cite only from [1]..[k]" contract, and scoring faithfulness
   with a second model (the programme's stated method).
3. **S11 album** - the embedding harness is built and the 1,000-image pass ran at **45.03 ms/image**
   (MobileCLIP2-S3, AI cluster), but the first metrics are **rejected: the fixture is wrong** - its
   per-group variants are near-identical (intra-group cosine mean 0.970, min 0.898) and overlap the
   inter-group band (mean 0.901, max 0.966), so the planted duplicates are not separable from the
   non-duplicates (aHash precision 0.005 even at Hamming 0; embedding cosine 0.039 at >=0.999). The
   perfect NMI 1.000 is therefore treated as uninformative too. Fixture v2 (textured bases) was built
   and re-run at 44.86 ms/image; it moved the numbers without changing the conclusion - the two
   cosine bands still overlap (intra 0.959/0.881 against inter 0.909/0.981) because MobileCLIP2-S3 is
   trained on natural images and synthetic renders all land in one narrow cone. **S11 is recorded as
   blocked on material**, then unblocked the same day: `ffmpeg` was installed and 319 natural frames
   were sampled from the vendor's own 1080p sample videos (7 groups, 46.34 ms/image on the AI
   cluster). Clustering NMI 0.881 / purity 0.865; same-video cosine 0.954 against cross-video 0.470
   (bands separate, unlike the synthetic set); **verbatim duplicates: precision and recall 1.000
   with exact hashes**, with the embedding gate explicitly rejected for that job because consecutive
   frames 0.5 s apart are near-identical. S11 accepted on the split-gate basis.
4. **S10 embodied decision** - 100 generated scenes with ground truth, a decision-latency budget, the
   S7/S8 interference matrix re-used, `vla/` models optional.
5. **S13 audio** - optional; the vendor ASR path against a board-built `whisper.cpp`.

## 7. Evidence index

| package | content |
|---|---|
| `k3-evidence/onnxruntime/s12-fixture-20260922/` | 30-page fixture, dictionaries, tesseract baseline, seven own-chain runs (two invalid), CER/WER tables |
| `k3-evidence/onnxruntime/s9-rag-20260922/` | corpus + manifest, chunks.jsonl with vectors, retrieval and answer summaries |
| `k3-evidence/onnxruntime/s11-album-20260922/` | 1,000-image fixture with labels and digests |
| `k3-evidence/onnxruntime/vendor-app-20260922/`, `multisession-20260922/`, `scenario-*-20260922/`, `peak-20260922/` | the earlier scenarios' raw logs and TCM samples |

All packages carry a per-file sha256 registry with a self-check that comes out 0 mismatched /
0 missing / 0 unlisted.

---

# Addendum, 2026-09-23: the first full results after S9-S12 execution

## A1. S11 album - **accepted** (natural material)

The scenario was blocked on material, the blocker was named, and the unlock was executed on the
board: `ffmpeg` installed, **319 natural frames** sampled from the vendor's own 1080p sample videos
(2 fps, 256x256, **7 groups**), 20 verbatim duplicates planted, and MobileCLIP2-S3 run through the
C++ harness at **46.34 ms/image** on the AI cluster. Clustering NMI **0.881** / purity 0.865;
same-video cosine 0.954 (min 0.281) against cross-video 0.470 (max 0.763) - the bands separate, which
the two synthetic fixtures never managed. **Verbatim duplicates: precision and recall 1.000 with an
exact-hash gate**, and the embedding gate is explicitly rejected for that job because consecutive
frames 0.5 s apart are near-identical. Both synthetic fixtures are kept as the record of why
synthetic material cannot carry this scenario.

## A2. S9 RAG - stages measured, quality gate open, reference blocked

| stage | metric | result |
|---|---|---|
| retrieval | chunks / embedding cost | 2,226 from 38 files (531 KB), **17.7 ms/chunk** |
| retrieval | recall@1/5/10, verbatim / shuffled control | **1.000 / 0.980** |
| answer | TTFT median (P90) | 2,226 ms (3,887 ms) |
| answer | citation correct | **16/20** (first prompt), **14/20** (stricter prompt - worse) |
| answer | non-empty | 20/20 |
| reference | vendor `llm-sdk` | installed and driven: kb created, **38/38 uploaded and chunked**, batch vectorisation **accepted then silently dropped** - 0/38 vectorised, because the catalogue has no embedding model and the app that ships one is GUI-only |

## A3. Optimisation strategy, measured (the operator's parallel/invocation axis)

| lever | effect | verdict |
|---|---|---|
| decode under N slots | 6.13 -> 7.88 -> **9.43 t/s** (1/2/4) | **use slots** (+53.8 %) |
| prefill under N slots | 5.35 -> 10.4 -> 18-20 s | does *not* parallelise |
| retrieved context 1 -> 4 chunks | TTFT 914 -> **2,592 ms** | **keep k small** |
| prefix caching, verified | `prompt_n` 121 -> 1, prefill 1,037 -> **78 ms** | **cache only genuinely repeated prefixes**; read `prompt_n` per request |
| embedding, concurrent clients | 51.2 -> 55.2 chunks/s | +7.7 %: feed it as one stream |
| OCR batch-4 recogniser | whole page 213-316 -> **650-904 ms** | **rejected** (-70 %) |
| stricter citation prompt | 16/20 -> **14/20** | citation integrity must be enforced *outside* a 2B model |

## A4. S12 ingestion - accepted (unchanged)

Printed CER **0.0001** (negative control, band <= 0.01), tables **0.0053 against tesseract's 0.0188**,
scanned 0.0061 inside the stated band; **3-5 pages/s** on the AI cluster; five failed or invalid runs
kept with their causes.

## A5. Remaining, in order

1. **S10 embodied decision** - **measured, one scene short of its accuracy target**: fixture v2 (100
   labelled scenes, reading-order ground truth), four rejected variants and a perception path at
   **94/100 with zero segmentation failures and 46.6 ms median** on the X100. The VLM-only baseline
   (3/12, 560 ms) is recorded as the negative control it is. Remaining: attribute the six misses and
   decide whether the 95 % target or the fixture's tolerance is the thing to adjust.
2. **S13 audio** (optional) - vendor ASR against a board-built `whisper.cpp`.
3. **S9 reference** - only if the vendor supplies an embedding model to the SDK (or a headless path
   for Zenow); the self-chain numbers stand meanwhile.
