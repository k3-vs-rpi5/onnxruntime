# The scenario programme: a more complete acceptance scheme, and the next scenarios

Operator's brief (2026-09-22): the test scheme must be more complete because it is the foundation of
stable acceptance; more scenarios must be built at that scheme level - LLM RAG over a knowledge
base, embodied vision understanding plus decision, knowledge-base and photo-album construction - and
mature software may be used as the reference; **analysis and research first, no code this round**.

This document is that analysis. It (1) audits what the current scheme lacks, (2) proposes scheme v2,
(3) defines the new scenarios with their acceptance axes and reference implementations, (4) records
what this round's research found on the board, in the vendor documentation and in the vendor
archive, and (5) sets the order of work and what needs approval. Nothing here is measured yet; that
is the point.

## 1. Audit: what the current scheme does not cover

`SCENARIO-VALIDATION-20260922.md` gives a five-step ladder, five classes and an evidence contract. It
is good at *speed and engagement* and weak at everything else:

| gap | what is missing today | why it matters |
|---|---|---|
| **quality** | no accuracy gate: no mAP, no OCR CER/WER, no retrieval recall@k, no answer-faithfulness rubric | a scenario can be fast and useless; no current scenario would catch a model that answers wrongly |
| **statistics** | n=3 medians and "the spread is the band", but no confidence interval, no drift/thermal rejection rule, no outlier policy | acceptance needs a stated pass/fail band, not a habit |
| **reference (oracle)** | only the vendor's published rows; no independent implementation to compare *outputs* against | with RAG/album/decision scenarios, "correct" is not a published row, it is an answer |
| **fixtures** | ad-hoc inputs (one frame, one clip); no sealed, versioned datasets with ground truth | scenarios S9-S13 need corpora, labels and generators that do not change under the measurement |
| **long run** | three-minute bursts only (sustained YOLO 3x57 s was the exception) | knowledge bases and albums are batch jobs measured in minutes-to-hours; stability and memory growth matter there |
| **resource accounting** | TCM occupancy yes; DDR bandwidth, CPU side, memory peak, disk I/O only sporadically | the scenarios above are I/O- and memory-shaped, not only AI-core-shaped |
| **failure taxonomy** | invalid / blocked exists; no "wrong answer", "timeout", "partial result" classes | RAG and decision scenarios fail by giving bad answers, not by crashing |
| **acceptance levels** | no separation of smoke / parity / best-achievable per scenario | the same scenario must be runnable in 5 minutes (smoke) and in an hour (acceptance) |
| **maturity tracking** | scenarios are listed, not graded | an operator needs to know whether a scenario is "defined", "instrumented", "measured" or "stable" |

## 2. Scheme v2 (proposed)

**Ladder, extended from five steps to seven:**

| step | adds |
|---|---|
| L1 engagement | unchanged (TCM naming the workload, clean board before/after) |
| L2 correctness | **split**: L2a numerical identity (bit-exact where it must be), L2b **semantic/task quality** against labelled fixtures |
| L3 standalone | unchanged (per-stage cost, vendor flags, named statistic) |
| L4 composition | unchanged (the three deployment rules) |
| L5 comparison | unchanged (published row, or our own named statistic) |
| **L6 reference** | the same scenario run against an **independent implementation** - a vendor application, a host-side reference, or an upstream test suite - with the *outputs* compared, not just the speeds |
| **L7 stability** | a soak/batch run (>= 10x the smoke duration) with memory, temperature and TCM sampling, and a drift threshold |

**Statistics policy (applies to every scenario):** state n; headline = median of runs; report the
distribution when latency matters (P50/P90/P99) and the spread for throughput; a difference is a
result only if it exceeds the scenario's measured spread; a run whose temperature, governor or
background load is outside the declared window is **invalidated, not averaged**.

**Quality-gate library (each scenario picks its gates):**

| domain | gate | fixture |
|---|---|---|
| detection | mAP@50 and recall on a fixed labelled frame set | 200 frames with boxes (to be built once, sealed) |
| classification | top-1 agreement against the reference implementation | 500 images |
| OCR (text) | CER/WER against a reference transcription | 30 pages (printed + scanned) |
| embedding/retrieval | recall@1/5/10 against the reference embedding model | 300 query-document pairs |
| RAG answer | citation correctness + answer faithfulness (rubric scored by a second model, human spot-check) | 50 questions over the sealed corpus |
| VLM understanding | answer accuracy on a fixed Q/A set | 50 image-question pairs |
| decision (embodied) | decision accuracy against scenario ground truth | 100 generated scenes |
| album | cluster purity/NMI, duplicate-detection precision/recall | labelled image set with known groups/duplicates |

**Oracle policy (L6):** in preference order - (1) the vendor's own application for the same task
(`yolo-demo` already used this way; `zenow`, `seewise`, `yumeet`, `llm-sdk` are candidates), (2) an
independent implementation of the same algorithm on the host (for embeddings/clustering: the same
model on x86 CPU, not a different model), (3) an upstream test suite. An oracle that cannot be run
is recorded as "no oracle available" rather than replaced by a weaker comparison.

**Fixture policy:** every fixture is a sealed artefact (digest, provenance, licence) under
`/root/k3-evidence/`; synthetic fixtures come with their generator's seed and digest; fixtures never
change inside a scenario version.

**Regression policy:** each scenario carries an accepted baseline with its band; a new result outside
the band is a regression until explained; scenarios have three levels - *smoke* (<= 5 min),
*acceptance* (30-60 min), *soak* (>= 2 h, bi-weekly).

**Maturity levels:** `defined -> instrumented -> measured -> oracle-validated -> stable`, recorded in
the matrix; a scenario advances one level at a time and never by assumption.

## 3. The new scenarios (definitions, not measurements)

### S9 - LLM RAG over a knowledge base

* **Corpus**: this workspace's own reports and evidence indexes (we know the ground truth, and the
  corpus is already sealed by digest) plus a scanned-document fixture to exercise OCR ingestion.
* **Pipeline**: ingest -> chunk -> embed -> index -> retrieve -> (rerank) -> answer with citations.
* **Acceptance axes** (the same ones the vendor's own AI NAS document names for RAG - 文档解析速度、
  向量化速度、召回准确性、问答延迟): parse speed, vectorise speed, recall@k (L6 against a host-side
  reference embedding), answer latency (TTFT + total), citation correctness, faithfulness rubric.
* **Models available for it** (vendor archive, measured or measured-adjacent here): text embeddings
  `Bge-Small-Zh/En-V1.5`, `Qwen3-Embedding-0.6B`, `Jina-V5-small`, `Nomic-Embed-V2-MoE`; rerankers
  `Bge-Reranker-V2-M3`, `Qwen3-Reranker-0.6B`; the LLM side is the frozen llama.cpp stack.
* **Oracles**: vendor `zenow` (knowledge-base Q&A application, apt candidate 0.0.184) and `llm-sdk`
  (knowledge-base APIs with `kb_ids`, candidate 0.0.39); host-side reference embedding for recall.
* **Risks**: embedding-model support in this fork (`llama-embedding` exists and `llama-server
  --embeddings --pooling` is available - measured this round), index memory on a 31 GB board, and
  the fact that our Python ORT silently falls back to the CPU EP (a trap for any Python ingestion
  tooling).

### S10 - Embodied: vision understanding plus decision

* **Shape**: detector -> depth/pose -> VLM understanding -> decision rule, evaluated on a **generated
  scene set** with ground truth (object identity, position, target), so the decision has a right
  answer.
* **Metrics**: decision accuracy, end-to-end decision latency (budget stated per scenario), stage
  costs inside the loop, interference with a language service (we already know from S7/S8 what that
  costs).
* **Additional material found this round**: the archive carries a **`vla/` family** (`act`,
  `smolvla`, `lingbot-vla-v1`, `minicpm-robotmanip`, with its own `libs/` and `torch/`) and
  `Qwen3-VL-Embedding-2B` / `Qwen3-VL-Reranker-2B` - i.e. a real policy-model layer is available if
  the operator wants to go beyond a rule-based decision.
* **Oracles**: ground-truth scenes; a host-side reference pipeline (same models, x86 CPU) for the
  understanding stages; the VLM-only answer as the "no perception" baseline.

### S11 - Photo album construction

* **Shape**: image set -> embeddings (image encoder) -> clustering into albums/scenes -> duplicate
  detection -> natural-language search ("find the cat photos").
* **Acceptance**: cluster purity/NMI against known groups, duplicate precision/recall, search
  top-k accuracy, build time per 1,000 images.
* **Material**: the vendor zoo's `image_encoder.fp16` (MobileCLIP2-S3), `siglip2` (both were measured
  EP-complete here), `cn_clip_text/vision.onnx` from the archive, and `Qwen3-VL-Embedding-2B` for a
  stronger (and slower) reference.
* **Oracles**: host-side reference with the same embeddings; vendor `seewise` (video semantic search,
  candidate 1.0.16) for the search-shaped part; mature third-party album software (Immich,
  PhotoPrism) is **not** available in the Bianbu archive - recorded as "no oracle available" unless
  the operator wants to try a source build, which the storage/GPU assumptions make unlikely.

### S12 - Document to knowledge base (ingestion quality)

* **Shape**: PDF/scan -> layout + OCR -> markdown/text -> chunk -> index (the front half of S9).
* **Acceptance**: CER/WER per page type (printed, scanned, table), ingestion speed per page,
  structure preservation sampling.
* **Material**: our measured PP-OCRv6 tiny pair (det 35.99 ms / rec 2.36 ms, both EP-complete) plus
  the vendor `file2md` application as the oracle if it can be obtained (its apt candidate is
  currently **none** in the Bianbu archive we can see - recorded as a gap).

### S13 - Audio/meeting (optional, only because it is the last empty domain)

* **Shape**: ASR RTF, diarisation, summarisation (the vendor's `yumeet`, candidate 1.0.1, is the
  ready-made oracle; its own published ASR rows exist in the modelzoo).
* **Status**: we have never measured audio here; the modelzoo's sensevoice rows are therefore the
  only anchor. Included in the programme so the empty domain is visible, not because it is next.

## 4. Research findings (2026-09-22, no code written)

**Board capabilities** (measured by inspection, not benchmarked): Python 3.14.4, numpy 2.5.1,
onnxruntime 1.24.2+spacemit.a1 installed as a *Python* package (note: the Python path silently uses
the CPU EP, a known trap); llama.cpp tools present including `llama-embedding`, `llama-lookup`,
`llama-perplexity`; `llama-server --help` confirms `--embeddings`, `--pooling`, `--embd-normalize`;
`cmake` and `g++` are installed on the board (small native builds are possible); 68 GB free disk and
31 GB RAM.

**Vendor documentation inventory**: the `solutions/aicomputer_solution/` set contains applications
that *are* these scenarios - `zenow` (local knowledge-base assistant: LLM + Embed + Rerank),
`ainas` (AI NAS: album aggregation, image search, document OCR + vectorisation, local RAG with
knowledge-base management), `file2md` (document to markdown), `seewise` (video semantic search),
`yumeet` (meeting assistant: VAD/ASR/diarisation/translation/TTS + summary), `agentforce` (digital
employees with a knowledge base). **AI NAS documents the acceptance axes we should adopt for RAG
verbatim**: document-parse speed, vectorisation speed, retrieval accuracy, answer latency - plus a
storage-test matrix we do not need for this board.

**Package availability on this board**: `zenow` 0.0.184, `yumeet` 1.0.1, `llm-sdk` 0.0.39,
`seewise-2` 1.0.16 are in the Bianbu archive; `file2md`, `ainas`, `agentforce` are not. Third-party
(Immich, PhotoPrism, FAISS, Chroma, sqlite-vec) are **not** in the archive.

**Model inventory relevant to the new scenarios** (archive `model_zoo/`): `embed/` (Bge-Small Zh/En,
Jina-V5-small, Nomic-Embed-V2-MoE, Qwen3-Embedding-0.6B Q4_0/Q8_0, **Qwen3-VL-Embedding-2B**,
`cn_clip_text/vision.onnx`), `rerank/` (Bge-Reranker-V2-M3, Qwen3-Reranker-0.6B, Qwen3-VL-Reranker-2B),
`vla/` (ACT, SmolVLA, LingBot-VLA-v1, MiniCPM-RobotManip), plus the vision/VLM/ASR families we have
already used.

## 5. Order of work, and what needs approval

1. **Scheme v2 lands first** (this document + the updates to `SCENARIO-VALIDATION-20260922.md`):
   without the quality gates and the oracle policy, the new scenarios would just be more speed
   tables.
2. **S12 then S9**: ingestion first (it is the front half of RAG), RAG second; both can use corpora
   we already own, so no new fixture build is needed for the first pass.
3. **S11 album** next (embedding + clustering, closest to the material we already measured).
4. **S10 embodied decision** after that, because it needs a scene generator and a decision budget
   agreed with the operator; the `vla/` models are an option to be decided, not assumed.
5. **S13 audio** last, and only if the operator wants the domain covered.

Needs the operator: (a) approval to install the vendor applications used as oracles (`zenow`,
`llm-sdk`, and later `seewise`/`yumeet`) on the board - package installs are remote configuration;
(b) a decision on the embodied decision budget (what latency a decision may take) and whether policy
models from `vla/` are in scope; (c) whether a host-side reference implementation (x86 CPU, same
models) is acceptable as an oracle, since it is the cheapest way to get L6 for embeddings and
clustering; (d) confirmation that the fixture corpus may include this workspace's own reports (they
are already public in intent, but they are our own material).

Explicitly **not** in this round: no code, no board runs, no installs, no fixture generation, no
changes to the frozen stages.

## 6. GitHub open-source lightweight candidates (not only vendor material)

The operator's second instruction: look beyond the vendor's own documentation and applications, and
keep it light. Every candidate below was checked on GitHub on 2026-09-22 (language, licence, last
push); "portable" means it is pure Python, or C/C++ that the board's own `cmake`/`g++` can build -
the board has no Rust or Go toolchain by default, which is what rules the heavier stacks out.

| scenario | candidate | what it gives | licence / language | verdict |
|---|---|---|---|---|
| S9 RAG | **`asg017/sqlite-vec`** | vector index as a SQLite extension; brute-force + ANN; tiny | Apache-2.0 / C | **primary index** - single C file, builds on the board, no service |
| S9 RAG | **`simonw/llm`** | CLI: `embed-multi` into SQLite, prompt templates, plugins; scriptable, testable | Apache-2.0 / Python | **primary glue + oracle harness**: the whole RAG pipeline becomes a reproducible CLI |
| S9 RAG | `nmslib/hnswlib` | header-only HNSW index | Apache-2.0 / C++ | alternative index if sqlite-vec's brute force is too slow at 10^5 chunks |
| S9 RAG | `unum-cloud/USearch` | compact ANN index, C++ with bindings | Apache-2.0 / C++ | reserve |
| S9 RAG | `facebookresearch/faiss` | the reference ANN library | MIT / C++ | **host-side reference** for recall@k, not for the board (heavy build) |
| S9/S12 | `tesseract-ocr/tesseract` | classic OCR engine | Apache-2.0 / C++ | independent OCR oracle for the CER/WER gate, board-buildable |
| S12 ingestion | `docling-project/docling` | document -> structured text/markdown | MIT / Python | moderate weight (torch-based models optional); use only if PDF layout fidelity must be validated, else `pymupdf`-class parsing + our PP-OCR |
| S11 album | `JohannesBuchner/imagehash` | perceptual hashes for duplicate detection | BSD-2 / pure Python | **duplicate gate** - trivially portable and exactly the right instrument |
| S11 album | `idealo/imagededup` | CNN + hashing duplicate detection, evaluation helpers | Apache-2.0 / Python | reference implementation for the duplicate gate (host side) |
| S11 album | `photoview/photoview` | self-hosted photo library (Go, web UI) | AGPL-3.0 / Go | **not usable on the board** (no Go toolchain, no e2e ML), recorded as "no oracle"; our album scenario stays algorithmic (clustering + search), judged by NMI/top-k |
| S11 album | `jmathai/elodie` | EXIF-based organiser, no ML | Apache-2.0 / Python | only if the scenario needs an EXIF/metadata baseline |
| S10 embodied | `huggingface/lerobot` | policy training/eval harness | Apache-2.0 / Python+torch | host-side reference only; board-side stays with the vendor `vla/` models |
| S10 embodied | `Breakthrough/PySceneDetect` | shot/scene segmentation | BSD-3 / Python | useful for the video variant of the scene generator, not the decision gate |
| S13 audio | **`ggml-org/whisper.cpp`** | standalone ASR, C++, no Python | MIT / C++ | **independent ASR oracle** for yumeet/sensevoice comparisons; board-buildable |
| search (S9/S11) | `quickwit-oss/tantivy`, `meilisearch/meilisearch` | lexical BM25 / search service | MIT / NOASSERTION, Rust | **excluded from the board** (no Rust toolchain); they are the reason a hybrid lexical+vector baseline is noted but not planned |

Two conclusions from the survey. First, the **board-side stack can stay tiny**: `sqlite-vec` +
`simonw/llm` + `llama.cpp` embedding/reranking covers S9/S12, `imagehash` covers the S11 duplicate
gate, `whisper.cpp` covers S13 - all Apache/MIT/BSD, all C or Python. Second, the heavier or
non-portable candidates are not "to do later"; they are recorded as **unusable on this board**, and
where they would have been the only oracle (photoview/immich-class album servers), the scenario is
declared oracle-less and judged against its own labelled fixture instead.

## 7. Lean rules, and what a scenario-level acceptance actually looks like

The operator's caution - do not overbuild - is now a rule, not a preference:

1. **Reuse before build.** A scenario adds code only for the part no cited OSS or vendor tool
   provides; the preferred pipeline is a shell script that calls `llm`, `sqlite-vec`, `whisper.cpp`,
   `llama-server` and our existing instruments.
2. **One fixture, one oracle, one page per scenario.** No fixture programme, no framework, no
   abstract layers.
3. **A scenario is accepted or rejected on its card**, not in prose.

**Acceptance card template** (the scenario-level artefact; each card is one screen):

    scenario / class / maturity
    fixture        (what, size, digest, seed)
    pipeline       (stages and the tool used for each)
    metrics        (speed: which statistic, n; quality: which gate)
    oracle         (implementation + how its output is compared)  [or: no oracle, why]
    pass/fail      (numbers with bands, and what invalidates a run)
    evidence       (directory, digest registry, self-check)
    known limits   (what this card does not claim)

### S9 card (draft)

* fixture: this workspace's reports + evidence READMEs as the corpus (sealed by digest), 50
  question/answer pairs written by us, plus the scanned-page fixture shared with S12;
* pipeline: `llm embed-multi` (BGE-Small-Zh GGUF via `llama.cpp`) -> `sqlite-vec` index -> top-k ->
  optional `Qwen3-Reranker-0.6B` -> answer with `llama-server`, citations required;
* metrics: parse and vectorise speed per document; recall@1/5/10; answer latency (TTFT + total);
  citation correctness and faithfulness rubric;
* oracle: **host-side same-model embeddings** for recall (x86 CPU), and the vendor `zenow`/`llm-sdk`
  application run on the same corpus where the operator approves the install;
* pass/fail: recall@5 no worse than the host reference minus the measured band; every answer cites a
  retrieved chunk whose text contains the answer span; a run with TCM not 8/8 at start is invalid;
* evidence: `k3-evidence/onnxruntime/s9-rag-<date>/`.

### S10 card (draft)

* fixture: 100 generated scenes with ground truth (object identity/position/target), generator seed
  and digest sealed;
* pipeline: detector -> depth -> VLM ask -> decision rule; the `vla/` policy models are an *option*
  recorded, not assumed;
* metrics: decision accuracy, end-to-end decision latency against a budget the operator sets, stage
  costs inside the loop (we already know the sharing price from S7/S8);
* oracle: the ground truth itself, plus a host-side reference pipeline using the same models;
* pass/fail: decision accuracy >= the reference minus band, and latency within budget, both at
  least 3 repeats; wrong answers are a failure class of their own (new taxonomy row).

### S11 card (draft)

* fixture: 1,000 images in known groups with planted duplicates (our frames plus generated
  variants), sealed;
* pipeline: image embeddings (MobileCLIP2-S3 / siglip2) -> `sqlite-vec` -> clustering (numpy
  agglomerative) -> `imagehash` duplicate pass -> text query search;
* metrics: cluster purity/NMI, duplicate precision/recall, search top-k, build time per 1,000;
* oracle: host-side reference with the same embeddings (recall/NMI); `imagededup` host-side for the
  duplicate gate; **no server-class oracle available** (photoview excluded, see §6);
* pass/fail: NMI within band of the host reference; duplicate recall >= 0.95 with precision >= 0.90
  on the planted set; a card that cannot state its band is not accepted.

### S12 card (draft)

* fixture: 30 pages (10 printed, 10 scanned, 10 tables), reference transcriptions sealed;
* pipeline: PDF/scan -> parse -> our PP-OCRv6 -> text; `tesseract` as the independent engine;
* metrics: CER/WER per page class, pages/minute, structure-sampling score;
* oracle: `tesseract` (and `docling` only if layout fidelity becomes the question);
* pass/fail: CER within band of the tesseract reference on printed pages, and a stated (not
  assumed) gap on scanned ones.

### S13 card (draft, optional)

* fixture: the vendor's own audio samples from the modelzoo table (`001_zh_daily_weather.wav`,
  `004_zh_selling_sausages.wav`), sealed;
* pipeline: ASR via the vendor path, oracle `whisper.cpp` (board-built);
* metrics: RTF, WER against the reference transcription;
* pass/fail: RTF and WER both stated against the modelzoo rows and against whisper.cpp.
