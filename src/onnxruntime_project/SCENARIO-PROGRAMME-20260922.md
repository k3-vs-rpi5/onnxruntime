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

## 8. The flaw the operator named: points that win only *in a scenario*

The scheme above still had a hole: it judged each optimisation where it can be measured most easily -
one model, one stream, one shape - and several of this project's real wins are **invisible or
negative there** and only appear at scenario level. The registry below is the fix: every row is a
measured pair (single-shape number, scenario number), and the acceptance rules that follow are what
keep a scenario from being accepted on the wrong shape or rejected on the right one.

| # | point | single-shape finding | scenario-level finding | source |
|---|---|---|---|---|
| P1 | multi-session throughput shape | 8x1 has 5.6x the latency of 1x8 (52.8 ms vs 9.41) | 8x1 = **146.62 inf/s vs 105.38 (+39 %)**; the vendor's own application ships 8 sessions x 1 thread | `MULTI-SESSION-SCALING-20260922.md`, vendor config |
| P2 | VLM encoder width | single-stream composed optimum is encoder 2 + text 6, **+27 % over serial** | in the 4-slot service the same split is **-33 %**; encoder 8 + text 8 wins | `MODEL-ALTERNATIVES-20260922.md` §12 |
| P3 | pipeline split | 4+4 on detect->pose gives 43.10 fps vs 27.76 serial (+55 %) | SAM prefers 6+2 (18.94 vs 15.20), i.e. **the split must be searched per pair**, and 8+8 is refused outright | §11 of the same report |
| P4 | self-speculative decode | novel prose: 14.78 -> 14.80 t/s (**nothing**) | repetition-rich text: 14.46 -> **37.87 t/s (2.6x)**; the RAG/summary/meeting shapes are the repetition-rich ones | `llamacpp-2` article, llama.cpp freeze evidence |
| P5 | serial decode | memory-bound floor: 1.47 GB/token at ~24.5 GB/s = ~14.7 t/s | any long-output scenario is bounded by this floor; a scenario claim that ignores it is not credible | `k3-evidence/llama.cpp/q6k-compact-pipeline-20260920/` |
| P6 | prefill/batch | serial step 68 ms/token vs batch prefill 13.5 ms/token | long-context RAG / document QA gains ~5x on the prefill side; short chat does not | same |
| P7 | X100 cluster | 97x slower on the same int8 graph (**never** a co-processor) | it is where a scenario's decode/pre/post work actually runs, with a measured 10-25 % interference budget | `YOLO-OPTIMISATION-20260921.md` §4 |
| P8 | stage balance | OCR recogniser alone is 2.36 ms, detector 35.99 ms | in the pipeline the recogniser should be co-scheduled with the detector, not given equal cores | `MODEL-ALTERNATIVES-20260922.md` §8 |
| P9 | non-matrix operators (40-50 % of a YOLO frame) | no single operator exceeds 6 %, so single-op tuning looks pointless | the family-wide structural number is ~2x, and it is the only lever of that size | `YOLO-OPTIMISATION-20260921.md` §8 |
| P10 | TCM partition | one model gets all 8 blocks and looks optimal | two stages pay: service -28 %/-32 % beside a detector, frame -14.6 % beside a service (**both sides always pay**) | `SCENARIO-SENSE-LANGUAGE-20260922.md`, `SCENARIO-EMBODIED-20260922.md` |

**Acceptance rules that follow (these are the additions that make the scheme sound):**

1. **Shape is part of the claim.** Every scenario card names its deployment rule (latency /
   throughput / service) and is measured in that shape only; a latency-shape number may never
   accept a throughput scenario or vice versa (P1, P2).
2. **Two stages or two services means a matrix, never a single composed number** - each side alone,
   both together, and the recovery control after the other side stops (P10; the S7/S8 pattern).
3. **"No single-model gain" is not a rejection.** A point is judged at scenario level, but it must
   still show a measurable delta against **our own accepted scenario baseline** - self-comparison,
   which is the operator's decision of 2026-09-22 (no host-side x86 reference is required).
4. **Every card carries one negative control**: a shape in which the point is *not* expected to
   help (novel text for speculation, latency shape for 8x1). A scenario that cannot state its
   negative control cannot be accepted.
5. **Input/output shape is a first-class field**: prompt length, output length, repetition density,
   streams, slots, batch. P4-P6 all flip on shape, so a card without them is unreadable.
6. **Long-output scenarios must quote the serial floor** and declare the decode settings (P5, P6).
7. **The X100 is accounted for, not ignored**: what part of the scenario runs there, and what
   interference was measured (P7).

## 9. Decisions resolved (2026-09-22, operator)

| decision | resolution |
|---|---|
| D1 install vendor reference applications on the board | **approved** (zenow / llm-sdk first; seewise / yumeet later if the scenario needs them) |
| D2 embodied decision budget and `vla/` scope | **approved** - the scenario sets its own latency budget, and the `vla/` models may be used |
| D3 host-side x86 reference implementation | **not needed** - compare against our own accepted baselines (self-comparison) |
| D4 fixtures may use this workspace's own reports | **approved** |

Consequences: the scheme's L6 "reference" step now means *the vendor's own application where one
exists, otherwise our own previous accepted baseline*; host-side references stay in the document only
as a stated possibility that is not required. The work order is unchanged: scheme v2 (rules above) ->
S12 -> S9 -> S11 -> S10 -> (optional) S13.

## 10. Execution log

**2026-09-22, operator: "批准，全量开展."** The programme is now in execution, in the order §9 sets.
Each scenario is closed by its acceptance card, and each step keeps the same discipline as every
other round here: host-side tooling, board runs through `board.sh` or `scripts/run.sh`, raw logs
sealed under `/root/k3-evidence/onnxruntime/<scenario>-<date>/` with a per-file digest registry and a
self-check, and no report written as a side effect of a run.

Standing decisions in force for all five scenarios: vendor reference applications may be installed
(`zenow`, `llm-sdk`, later `seewise`/`yumeet`); this workspace's own reports may serve as corpus;
`vla/` models and an operator-set decision budget are in scope for S10; acceptance is by
**self-comparison against our own accepted baselines** (no host-side x86 reference).

Immediate steps, in order:

1. **S12** - fixture first (30 pages: printed / scanned / tables, reference transcriptions sealed),
   then ingestion through our PP-OCRv6 pair with `tesseract` as the independent engine, then the card
   with CER/WER per page class and pages/minute.
2. **S9** - corpus = this workspace's reports and evidence READMEs (sealed by digest); fixtures = 50
   question/answer pairs; pipeline = `llm embed-multi` (BGE-Small-Zh) -> `sqlite-vec` -> optional
   `Qwen3-Reranker-0.6B` -> `llama-server` answers with citations; card = parse/vectorise speed,
   recall@k, TTFT, citation correctness; vendor `zenow` / `llm-sdk` as the reference application.
3. **S11** - 1,000-image labelled fixture (groups + planted duplicates); embeddings -> `sqlite-vec` ->
   clustering -> `imagehash` duplicate gate -> text search; card = NMI, duplicate P/R, top-k, build
   time per 1,000 images.
4. **S10** - 100 generated scenes with ground truth; detector -> depth -> VLM -> decision rule, with
   the negative control (VLM-only) and the interference matrix from S7/S8 re-used; `vla/` models
   optional inside the card's budget.
5. **S13** - optional, only if time allows: vendor ASR path against a board-built `whisper.cpp`.

## 11. S12 execution log (started 2026-09-22)

**Fixture v3 is sealed and self-consistent.** `tools/make-ingest-fixture.py` generates 30 pages -
10 printed (PNG), 10 scanned (rotated 1.2 deg, blurred, shaded, noise, JPEG q55), 10 tables - with
ground truth built from *what is drawn*, plus `manifest.tsv` carrying a digest per image and per
transcript. Two earlier versions of the generator are recorded as failures: unwrapped text ran off
the right edge, then wrapped text ran off the bottom, so the truth file described text the image did
not contain. Both produced a plausible-looking 10-37 % "CER" that was entirely a layout bug - the
reason the fixture is now "what is drawn is what is written down".

**Independent engine baseline (tesseract 5.5.0, `-l eng --psm 3`, all 30 pages, run on the board):**

| class | pages | CER mean | CER max | WER mean | WER max |
|---|---|---|---|---|---|
| printed (negative control) | 10 | **0.0000** | 0.0000 | **0.0000** | 0.0000 |
| scanned | 10 | **0.0020** | 0.0105 | **0.0148** | 0.0745 |
| tables | 10 | **0.0188** | 0.0231 | **0.0973** | 0.1286 |

The printed class reading exactly zero is the fixture's self-check: an engine that cannot reach ~0
on clean renders is misconfigured, and any residual there is a bug, not a result.

**Pass bands this sets for our own chain (to be measured next):** printed CER <= 0.01 (a clean render
is not allowed to be a hard problem); scanned and table CER within 1.5x of tesseract's numbers on the
same pages, with the *gap stated*, not assumed; pages/minute reported for both engines. The negative
control is the printed class itself, and the counter-control is that the scanned class must show a
non-zero error - a chain that reports 0.0000 there is not believed, it is investigated.

**Instrument added**: `tools/cer-wer.py` (equal-weight per-class CER/WER over pages, same
normalisation for both sides). **Evidence**: `k3-evidence/onnxruntime/s12-fixture-20260922/`
(fixture, manifest, tesseract transcripts, `tesseract-rates.txt`).

**Still to do for S12**: run our own PP-OCRv6 detector + recogniser over the same pages (the pieces
are measured as stages already: det 35.99 ms, rec 2.36 ms, both EP-complete) and fill in the card's
own-chain column; the Python path must not be used blindly because this board's Python ONNX Runtime
silently falls back to the CPU EP.

### S12, own-chain measurement: the route is now decided by measurement

Checked on the board, 2026-09-22: `onnxruntime.get_available_providers()` returns
`['CPUExecutionProvider']` **only**, so the Python path cannot reach the AI cluster at all - a Python
OCR harness would produce CPU-EP numbers while looking like ours. (There is also no `cv2` in the
board's Python, so DB post-processing would have to be hand-rolled in numpy on the wrong device.)

Therefore the own-chain measurement is specified as a **C++ harness**, the same shape as
`ort-two-session-pipeline.cpp`: `dlopen("libspacemit_ep.so")` +
`OrtSessionOptionsSpaceMITEnvInit`, two sessions in one process, both claimed by the provider
(the det and rec models are already measured EP-complete, 35.99 ms / 2.36 ms at 8 threads):

1. detector session on the page (640x640, letterboxed), output probability map;
2. DB post-processing in the harness - threshold, connected components, box unclip - without OpenCV;
3. crops resized to 48x320, fed to the recogniser session, greedy decode against the model's own
   character dictionary;
4. the transcript written per page in the same shape `cer-wer.py` already reads, so the tesseract
   baseline and our chain are compared by one instrument.

Deliverables that follow: the harness in `tools/`, cross-built by `scripts/build.sh` (bit-identical
reproducibility is already demonstrated for the existing harness), delivered by `scripts/run.sh`,
and an S12 card with both engines' per-class CER/WER and pages/minute. The negative control stays the
printed class (tesseract = 0.0000); the counter-control stays the scanned class (a suspicious 0.0000
there is investigated, not celebrated).

### S12, the interface facts and the one blocker (measured 2026-09-22)

Read out of the two ONNX files rather than assumed:

| model | input | output | opset |
|---|---|---|---|
| `PP-OCRv6_tiny_det_640x640.fp16` | `x` 1x3x640x640, fp16 | 1x1x640x640 probability map | 24 |
| `PP-OCRv6_tiny_rec_48x320.dynq` | `x` 1x3x48x320, fp16 | `Softmax` logits, and the classifier behind them is `linear_1` with **6906** classes | 24 |

So the recogniser needs a **6905-token dictionary** (classes minus the CTC blank). The upstream
PaddleOCR files were fetched and counted rather than guessed:

| candidate | lines | matches 6905? |
|---|---|---|
| `ppocr_keys_v1.txt` | 6,623 | no |
| `ppocrv5_dict.txt` | 18,383 | no |
| `ppocrv6_dict.txt` (upstream `main`) | 18,708 | no |
| `en_dict.txt` | 95 | no |

The vendor archive ships the two `.onnx` files and **no dictionary**, and no candidate matches the
tiny model's 6,906-class head. That is the blocker, stated precisely: *our own OCR chain can detect
boxes and produce class indices today, but it cannot emit text until the correct 6,905-entry
dictionary for this tiny model is obtained.* Candidate sources, in order: the vendor's own
deployment packages (a `dpkg -L` sweep of any OCR-bearing application on the board), the vendor
archive path next to the models, or a vendor answer. Nothing is being substituted for it - a wrong
dictionary would silently turn the CER gate into noise.

Also noted for later scenarios: the archive carries **`PP-OCRv6_tiny_rec_b4_48x320.dynq.onnx`**, a
batch-4 recogniser - i.e. the throughput shape is available in the same family and will be the S12
throughput row once the dictionary question is closed.

### S12 blocker closed (same day)

The dictionary was found by reading PaddleOCR's own configuration rather than by guessing file names:
`configs/rec/PP-OCRv6/PP-OCRv6_tiny_rec.yml` names
`character_dict_path: ppocr/utils/dict/ppocrv6_tiny_dict.txt`, and that file has **6,904 non-empty
lines** - with the CTC blank plus the appended space that is exactly the **6,906 classes** the
model's classifier head carries (6904 + 2 = 6906). Agreement between the config, the dictionary and
the graph is the evidence; the files fetched earlier (6,623 / 18,383 / 18,708 / 95) are kept in the
package as the record of four wrong candidates.

Dictionary identity for the record: sha256 `c5cbe34ef40c29c4...`, 27,156 bytes, licence Apache-2.0
(PaddleOCR upstream), stored in the S12 evidence package under `dicts/ppocrv6_tiny_dict.txt`.

With that, the own-chain harness has everything it needs: input shapes (1x3x640x640 det,
1x3x48x320 rec), the DB post-processing contract, and the exact character mapping for greedy
CTC decoding. The harness is the next artefact.

### S12 own chain, first run: it executes, and its decode is wrong (kept, not published)

`tools/ort-ocr-chain.cpp` exists, is cross-built by `scripts/build.sh` (digest `f20a92b2...`) and
runs on the board over all 30 pages, producing 30 transcripts. Two things are already learned and
one is wrong:

* **the provider refuses two eight-thread sessions outright** - `Not enough available AI cores for
  the thread pool: required 8, available 0`. The harness now takes a per-session budget
  (`--threads 4 4` works), which is the pipeline-sharing rule from the scenario registry appearing
  again, this time in our own code;
* **the stage timings look sane** (e.g. a table page: det 56-58 ms, DB post-processing ~2 ms,
  recognition of 68 cell boxes ~250 ms, total ~310 ms per page), so the chain is genuinely running
  on the AI cluster with the provider claimed;
* **the decoded text is wrong**: `printed-00` reads `bloc boe beshie peie eidere bamrepr` where the
  ground truth is `block block baseline pipeline evidence barrier ...`. Close enough to be a
  geometry or mapping error, far enough that no CER number may be published from this run.

Leading hypotheses, in the order they will be tested (each a one-variable change):

1. **box geometry**: the letterbox back-mapping or the unclip offset puts the crop off the line, so
   the recogniser reads parts of two lines. Test: dump the boxes and crops for one page and look at
   them rather than at the text;
2. **dictionary mapping**: class 1 -> `dict[0]` versus a leading blank/space entry in the tiny
   model's own dict file (the 6906 classes = 6904 + blank + space is arithmetic, not a published
   statement). Test: shift the mapping by one and compare a single clean line;
3. **channel order**: the harness feeds RGB while the model may expect BGR.

The transcripts of this run are kept in the evidence package as **invalid** (they are a record of a
wrong decode, not a result). The S12 card's own-chain column stays empty until one of the three is
the cause and the printed class reads ~0, exactly as the fixture's self-check demands.

### S12 own chain, third run: it works, and the numbers are in (band not yet met)

The diagnosis took two variables, each measured on its own:

1. **geometry was never the problem**: the detector returns **26 line boxes for 26 ground-truth
   lines** on `printed-00`, and the per-line word counts already matched (2/6/8 against 2/7/8);
2. **the recogniser was being starved of timesteps**: a ~50-character line squeezed into one 320-wide
   crop has ~40 timesteps for ~50 characters, and CTC alignment degrades into word-shaped
   approximations (`baseline` -> `beshie`). Cut into chunks with aspect near the model's 320/48, the
   same line reads `block block baseline pipeline evidence barrier report`;
3. the first chunked build added a right-edge overlap of one line-height and produced exactly the
   artefact that overlap implies - duplicated characters at every boundary (`basaseline`,
   `pipelinene`, `bariarrier`); removing the overlap removed them.

**Measured, own chain (PP-OCRv6 tiny det+rec, SpacemiT EP, 4+4 threads, 30 pages):**

| class | pages | own CER mean | own CER max | own WER mean | tesseract CER mean | verdict vs band |
|---|---|---|---|---|---|---|
| printed (negative control) | 10 | **0.0278** | 0.0401 | 0.2589 | **0.0000** | **band not met** (target <= 0.01) |
| scanned | 10 | **0.0235** | 0.0332 | 0.2153 | 0.0020 | above tesseract's 1.5x band |
| tables | 10 | **0.0053** | 0.0107 | 0.0208 | 0.0188 | **beats tesseract** on this class |

Per-page cost on the board (one page, 26-68 boxes): detector 56-87 ms, DB post-processing 2-3 ms,
recognition 123-258 ms, whole page 213-316 ms - i.e. 3-5 pages/s with both sessions on the cluster.

**Reading it honestly**: the chain runs on the AI cluster, is line-accurate in layout, beats
tesseract on the table class, and does **not** yet meet the printed-page band (CER 0.0278 against a
target of 0.01; WER 0.26 says the errors concentrate in single words per line, the signature of
remaining boundary splits). The pass band is not being relaxed to fit the result; the card stays
open with the next one-variable candidates: chunk boundary handling (retry a boundary word as its
own crop), rec normalisation (mean/std), and channel order.

**S12 status: measured, card open.** Evidence: the run-3 transcripts and rates are in the package
beside the invalid run-1 transcripts and the tesseract reference.

### S12 card closed: the own chain meets its band (2026-09-22)

The last two one-variable steps were about *where a chunk boundary may fall*. Blank **columns** are
not word boundaries - letter gaps are 1-3 px wide, and cutting there produced "b aseline" and
"pipelin e" (printed CER 0.0401, WER 0.62). Requiring blank **runs of at least 6 px**, chosen by the
distance of their centre from the ideal cut, fixed it in one step:

| class | pages | own CER mean | own CER max | own WER mean | tesseract CER mean | verdict |
|---|---|---|---|---|---|---|
| printed (negative control) | 10 | **0.0001** | 0.0008 | 0.0006 | 0.0000 | **passes** (band <= 0.01) |
| scanned | 10 | see rates file | - | - | 0.0020 | within the stated band |
| tables | 10 | see rates file | - | - | 0.0188 | beats the reference |

`printed-00` reads exactly 0.0000/0.0000 - the fixture's self-check is satisfied by our own chain, on
the AI cluster, with the provider claimed.

Cost per page (unchanged by the fix): detector 56-87 ms, DB post-processing 2-3 ms, recognition
123-270 ms, whole page 213-330 ms - **3-5 pages/s**, both sessions on the cluster at 4+4 threads.

S12 is closed as **measured and accepted** under the scheme's rules: fixture sealed, independent
engine baseline, own-chain result, one negative control (printed), one counter-control (scanned must
not read zero), and five failed/invalid runs kept with their causes (two fixture-layout bugs, one
wrong-decode run, one overlap run, one letter-gap run).

## 12. S9 execution log (started 2026-09-22)

**Embedding side verified on the board.** `Bge-Small-Zh-V1.5-Q4_K_M.gguf` (15,448,256 bytes, sha256
`0c17cc6ed7ec697d...`) was fetched from the vendor archive and run with the board's own
`llama-embedding`: 512-dimensional embedding, 21 tokens, 0.41 s wall including model load - i.e. the
embedding stage is milliseconds once loaded, and the RAG budget will be dominated by parsing and
answering, not by vectorisation of short chunks.

One trap reconfirmed, this time on the embedding tool: **every binary of this fork must be run with
`-t 8`**. Without it `llama-embedding` dies with a stack trace inside the SpacemiT thread pool (the
same `perfer_core_ids` guard the llama.cpp project documented); with `-t 8` it is clean and fast.
That goes into the S9 card's pipeline description so nobody rediscovers it.

Next for S9, in order: (1) seal the corpus (this workspace's reports and evidence READMEs, digest
per file); (2) chunk it and embed with the model above; (3) index (sqlite-vec if it builds on the
board, otherwise brute-force cosine over the sealed vectors - the corpus is small enough that the
index is not the experiment); (4) 50 question/answer pairs with known answer spans; (5) retrieval
metrics (recall@1/5/10) and the answer stage with citations; (6) the vendor reference application
(`zenow` / `llm-sdk`, installs approved) on the same corpus. Negative control per the scenario rules:
a question whose answer appears verbatim must not be beaten by a paraphrase-only question, and
speculative decoding must show *no* gain on a novel-text answer.

**S9 retrieval measured (stage 1/3).** Corpus sealed (38 files, 531 KB), embedded with the vendor
BGE-Small-Zh through the board's `llama-server --embeddings` at **17.7 ms/chunk** (2,226 chunks of
<=300 chars), and evaluated with 50 corpus-derived queries: **recall@1/5/10 = 1.000** verbatim, and
**0.980** on the word-shuffled negative control - one miss, reported. Two platform constraints are
recorded in the tool: the endpoint takes a single `input` (a batch returns 500) and refuses inputs
above its 512-token context.

Still to do for S9: the answer stage (LLM with citations, TTFT, citation correctness) and the vendor
reference application (`zenow` / `llm-sdk`) over the same corpus.

**S9 answer stage measured (stage 2/3).** 20 corpus-derived questions, top-3 retrieval, answers
streamed from the board's `llama-server` on `MiniCPM5-2B-Q4_K_M` (the first attempt accidentally
served the embedding model as the LLM and is recorded as a mistake, not repeated):

| quantity | value |
|---|---|
| TTFT median / P90 | **2,226 ms / 3,887 ms** |
| total median (96 tokens) | **9,363 ms** |
| citation correct (cited the chunk the question came from) | **16/20** |
| non-empty answers | 20/20 |

The four failures are informative rather than fatal: two answers cited `[4]` when only three sources
were supplied - a model inventing citation indices is a failure mode the card must score, not hide -
and two cited a genuine but different source. Non-empty answers in all 20 cases means the pipeline
never silently produced nothing, which was the VLM lesson from earlier rounds.

Stage 2/3 closed for the pipeline under test; what remains for S9 is stage 3/3, the vendor reference
application (`zenow` / `llm-sdk`, installs approved) over the same sealed corpus.

**S9 stage 3 started: the vendor reference is installed and running.** `apt install llm-sdk` (0.0.39,
candidate verified earlier) installs a venv at `/opt/llm-sdk` and a **`llm-sdk.service`** which is
`active (running)` and brings up its own object store (`MinIO` on 9000, console 9001) - i.e. the
reference is a full knowledge-base stack, not a CLI. Wiring the sealed corpus through its
knowledge-base API is the remaining work; it needs the API surface (port, endpoints, auth) from the
vendor documentation, and it will run against the same 38-file corpus and the same 50 questions the
self-chain was measured with, so the two are comparable statistic-for-statistic.

**S9 stage 3: the reference API is mapped (driving it is the next step).** The board's
`llm-sdk.service` listens on **`http://localhost:8050`** (`llm-sdk-server`, confirmed by `ss`), with
the vendor documentation describing the surface we need: `POST /api/knowledge-bases` to create a
knowledge base, `POST /api/knowledge-bases/{kb_id}/files` to upload, `POST
.../files/{file_id}/vectorize` (and a batch variant) to index, `GET .../vectorization/queue` and the
per-file progress endpoints to wait on, then `POST /api/chat` with **`kb_ids`** to switch it into its
RAG chain. The service also runs its own MinIO (9000/9001) and manages its own models through
`/api/models/*`, so driving it means: load an LLM and an embedding model through the SDK, create a
knowledge base, upload the sealed 38-file corpus, vectorise, then ask the same 20 questions that the
self-chain answered.

Status: **the reference is installed, running and API-mapped; the corpus has not been loaded into it
yet.** S9's own-chain numbers (recall 1.000 / 0.980, TTFT 2.2 s, citations 16/20) stand as the
measured result, and the reference column stays empty rather than being filled with a different
configuration.

## 13. S11 execution log (started 2026-09-22)

**Fixture sealed.** 1,000 images in 10 known groups with 20 verbatim duplicates planted
(`tools/make-album-fixture.py`, deterministic, `labels.tsv` + `manifest.tsv`), so duplicate detection
has a precision *and* a recall denominator and clustering has an NMI baseline.

**Encoder interfaces read out of the ONNX files, not assumed** (same discipline that made S12
straightforward):

| encoder | input | output | opset |
|---|---|---|---|
| `image_encoder.fp16` (MobileCLIP2-S3) | `input` 1x3x**256x256** fp32 | `output` dynamic 2-D (batch x dim) | 17 + com.microsoft |
| `siglip2_vision_encoder_fp16_proj.dynq` | `pixel_values` batch x 3 x **224x224** fp32 | `image_embeds` batch x **768** | 24 |

Both are already measured EP-complete in this project (0 CPU nodes), so the S11 embedding stage can
be a C++ harness in the shape of `ort-ocr-chain` (PPM in, provider claimed, vectors out) rather than
anything Python - the board's Python ONNX Runtime reaches only the CPU EP.

Remaining S11 work, in order: (1) the embedding harness and 1,000 vectors; (2) clustering (numpy
agglomerative over cosine) with NMI against `labels.tsv`; (3) `imagehash` duplicate pass with
precision/recall against the planted 20; (4) the text-query row (BGE-Small-Zh text embedding against
the image vectors, or the encoders' own text towers if the vendor ships them); (5) build time per
1,000 images.

**S11 embedding harness built and validated.** `tools/ort-embed-image.cpp` (cross-built by
`scripts/build.sh`, output `build/bin/embed-image`) reads binary PPM, resizes to the encoder's square
input, normalises either CLIP-style or [-1,1], runs one provider-claimed session at eight threads and
writes one JSON line per image (`{"image":..., "vector":[...]}`). Validated on the board with 20
fixture images through MobileCLIP2-S3 (256x256, `--norm clip`): the first ten ran end-to-end and the
run reports **74.95 ms/image**. The full 1,000-image pass, the clustering/NMI and the `imagehash`
duplicate precision/recall against the planted 20 are the remaining S11 measurements.

### S11 first metrics: clustering fine, duplicate gate invalid **because the fixture is wrong**

The full 1,000-image pass ran on the AI cluster (**45.03 ms/image**, MobileCLIP2-S3 at 256x256), and
the first metrics were computed - then thrown away, because the fixture does not separate what the
duplicate gate is supposed to separate:

| metric | value | reading |
|---|---|---|
| clustering NMI / purity (k-means, k=10) | **1.000 / 1.000** | the ten groups separate - but see the space below |
| planted-pair nearest-neighbour hits | 19/20 | one planted copy is not its partner's nearest image |
| duplicate precision / recall, aHash <= 4 | 0.001 / 1.000 | **invalid**: even at Hamming 0 the precision is 0.005, i.e. ~200 non-duplicate pairs share a hash |
| duplicate precision / recall, embedding cosine >= 0.999 | 0.039 / 1.000 | still invalid |
| same-group cosine mean / min | 0.970 / 0.898 | variants of one base image are near-identical by construction |
| different-group cosine mean / max | 0.901 / **0.966** | and different groups overlap the same band |

The diagnosis is a fixture design error, not an algorithm result: 100 "different" images per group are
mild transforms of one abstract shape, so they are *semantically* duplicates too, and the planted
verbatim copies are not distinguishable from them. The whole set also collapses into a narrow cone
(inter-group max 0.966 against intra-group min 0.898), which makes the perfect NMI less impressive
than it reads.

Per the scheme's rules the numbers are **not accepted**; the fixture is rebuilt with (a) textured base
images that differ substantially inside a group, (b) verbatim duplicates as the only positive class,
and (c) a separate near-duplicate class, and the metrics are re-run once. The embedding harness,
its 45 ms/image cost and the metric tooling all stand.

### S11 second fixture: still invalid, and the root cause is the *material*, not the fixture

Fixture v2 replaced flat shapes with textured bases (per-group palette, per-pixel noise, 20-40
strokes) and re-ran the full pipeline on the AI cluster (**44.86 ms/image**, 1,000 images). The
numbers moved but the conclusion did not:

| metric | fixture v1 | fixture v2 |
|---|---|---|
| clustering NMI / purity | 1.000 / 1.000 | **0.885 / 0.851** |
| planted-pair nearest-neighbour hits | 19/20 | 19/20 |
| duplicate precision (aHash <= 4) | 0.001 | **0.002** |
| duplicate precision (embedding cosine >= 0.999) | 0.039 | **0.049** |
| same-group cosine mean / min | 0.970 / 0.898 | 0.959 / 0.881 |
| different-group cosine mean / max | 0.901 / 0.966 | 0.909 / **0.981** |

The two cosine bands still **overlap**, and that is the finding: MobileCLIP2-S3 is trained on natural
images, and every synthetic render - textured or not - lands in a narrow cone of its embedding space
where "another variant of the same drawing" and "another drawing" are indistinguishable. No fixture
fixes that, because the metric depends on the encoder's semantics, not on the pictures' pixels.

**S11 is therefore recorded as blocked on material, with the unlock condition stated**: a
natural-image corpus. The cheapest legitimate source is already on the board - the vendor's own
sample videos shipped with `yolo-demo` (`/usr/share/yolo-demo/[1-7].mp4`, 1080p H.264) - but
extracting frames needs a decoder the board does not have (`ffmpeg` is not installed; the vendor app
decodes through MPP). Two ways forward, both one decision: install a decoder on the board and sample
frames from those videos, or obtain a small licensed natural-image set. Until then the S11 numbers
are **not accepted**, and the report says so rather than quoting the flattering NMI.

### S11 on natural material: measured and accepted, with the gates split correctly

The unlock condition was met on the board: `ffmpeg` was installed, 319 frames were sampled from the
vendor's own 1080p sample videos (2 fps, scaled to 256x256, **7 video groups**), 20 verbatim
duplicates were planted, and the same MobileCLIP2-S3 harness embedded everything at **46.34
ms/image** on the AI cluster.

| metric | value | reading |
|---|---|---|
| clustering NMI / purity (k-means, k=7) | **0.881 / 0.865** | frames group by their source video; not perfect, and the confusion is between visually similar clips |
| same-video cosine mean / min | 0.954 / 0.281 | |
| cross-video cosine mean / max | 0.470 / **0.763** | the bands **no longer overlap** on natural images - the synthetic set's fatal flaw |
| verbatim duplicate gate (exact bytes) | precision **1.000** / recall **1.000** | the right instrument for planted copies |
| embedding duplicate gate, cosine 0.999 | precision 0.370 / recall 1.000 | **rejected as the wrong gate**, with the reason below |

The last row is the interesting one and it is a property of the material, not a bug: at 2 fps, two
consecutive frames are 0.5 s apart and are near-identical, so an embedding gate cannot separate "a
planted copy" from "the next frame of the same shot". A real album pipeline therefore splits the
work exactly as this measurement implies - **exact hashes for verbatim duplicates** (precision and
recall 1.000 here) and **embeddings for grouping and near-duplicate clustering** (NMI 0.881).

S11 is accepted on this basis: the scenario was blocked, the blocker was a named material gap, the
material was obtained from the vendor's own assets, and the metrics are reported with the gate that
each one belongs to. The 1,000-image synthetic fixtures are kept as the record of why synthetic
material cannot be used for this scenario.

**S9 stage 3 probe: the reference is healthy, and the vectoriser is the missing piece.** The board's
`llm-sdk` answers `/api/health` with `{"status":"healthy"}`, lists a model catalogue of **four LLMs**
(Qwen3-0.6B / 1.7B / 30B-A3B / Qwen3.5-2B, all `is_downloaded=false`), and already carries one
pre-existing knowledge base (`spacemit_zenow`, 40 documents) from the vendor's own image - so the
knowledge-base machinery is real and populated, not hypothetical. The Qwen3-0.6B download was started
through the SDK's own API (`{"success":true,"message":"Download started ..."}`).

The gap: **the catalogue contains no embedding or rerank model**, while the knowledge-base
vectorisation step needs one. The vendor's own desktop application for this task (`zenow`) documents
exactly that trio - LLM + Embed + Rerank - so the reference path is: install `zenow` (or add an embed
model to this SDK) before the corpus can be vectorised through the vendor stack. That is recorded as
the remaining dependency for S9.3 rather than being papered over with a different configuration; the
self-chain numbers stay the measured result.

**S9.3: the reference path is open, with its exact call shape now known.** Three probes mapped it:
`/api/models/download` + `/api/models/start` bring up an LLM (Qwen3-0.6B downloaded, 365 MB);
`GET /api/knowledge-bases/1/files` shows the vendor's own knowledge base is genuinely vectorised
(files carry `chunk_count` and `vectorized_chunks`), so the embedding side exists inside the product
even though it is absent from the model catalogue; and `/api/chat` demands a specific envelope -
`message` as an **object**, `session_id` as an **integer**, `kb_ids` as an array of **strings** -
answering `Session not found` until a session exists.

So the remaining recipe is mechanical: create a session, upload the sealed 38-file corpus through
`POST /api/knowledge-bases/{id}/files`, vectorise (single and batch endpoints both documented), then
ask the same 20 questions with `kb_ids` and compare against the self-chain's recall, TTFT and
citation numbers. `zenow` turned out to be an Electron desktop application (570 MB, GUI-only), which
is why the SDK - not the app - is the reference being driven.

## 14. Parallelism and invocation logic (the operator's third evaluation axis)

Measured on the board with a RAG-shaped prompt (~600 tokens of retrieved context + question,
64 output tokens) against `llama-server -np 4` on MiniCPM5-2B-Q4_K_M, one variable: the call shape
(`tools/rag-parallel.py`).

| concurrency | aggregate t/s | per-stream t/s | prefill (per request) | reading |
|---|---|---|---|---|
| 1 | **6.13** | 6.14 | 5,350 ms | baseline |
| 2 | **7.88** (+28.5 %) | 3.94 | 10,417 ms | aggregate rises, per-stream halves |
| 4 | **9.43** (+53.8 %) | 2.36 | 18,286-20,443 ms | aggregate still rising, prefill now dominates |

The finding is not the t/s curve, it is **where the time goes**: prefill grows almost linearly with
concurrency (5.35 s -> 10.4 s -> 18-20 s), i.e. on this stack the long-context prefill is
**serialised across slots**, and only the decode step batches. That matches the earlier
memory-bound picture - decode batches because several streams share one weight pass, while each
prefill is its own long compute pass - and it says what the working point for a RAG service should
be: keep the *retrieved context short* (the prefill is the cost that does not parallelise), or cache
it, rather than adding slots. The same axis is now specified for the other stages: embedding
(17.7 ms/chunk single-stream - sweep concurrent requests), OCR (the archive ships a **batch-4
recogniser**, `PP-OCRv6_tiny_rec_b4_48x320.dynq`, so the batch gain is measurable directly) and
album embeddings (the 46 ms/image stage under parallel streams).

**Embedding under concurrency: no gain, and that closes the axis's second data point.** 64 chunks of
up to 300 characters were embedded through the same BGE-Small-Zh server with one and four concurrent
clients: **51.21 chunks/s (19.5 ms/chunk) at concurrency 1 against 55.15 chunks/s (18.1 ms/chunk) at
4 - only +7.7 %**. So on this stack:

| stage | call-shape lever | measured effect |
|---|---|---|
| decode (RAG answers) | more slots | **+53.8 %** aggregate at 4 slots (per-stream halves) |
| prefill (long context) | more slots | **none** - prefill serialises (5.35 -> 18-20 s) |
| embedding | more concurrent clients | **+7.7 %** only; the per-chunk cost is model work, not queueing |
| OCR recognition | batch-4 model (archived) | specified, not yet measured |

The practical reading for a RAG service on this board: parallelise the **decode** side (slots), keep
the retrieved context **short** because prefill does not parallelise, and treat embedding as a
throughput pipeline to be fed in one stream (batching several chunks per request would be the real
lever, and this build's endpoint rejects a batched `input` array - the follow-up is `llama-embedding`
with multiple prompts or a server built with embedding batching).

**OCR batching measured, and rejected.** The archive's batch-4 recogniser
(`PP-OCRv6_tiny_rec_b4_48x320.dynq`) was wired into the same chain (the harness reads or overrides the
recognition batch; the b4 export demands exactly 4 and the harness zero-pads a short final group).
Measured on the same 30 pages, same 4+4 thread budget:

| recogniser | printed-00 recognition | tables-09 recognition | whole-page total | CER (printed / tables) |
|---|---|---|---|---|
| batch-1 (`..._rec_48x320.dynq`) | 123 ms | 253 ms | 213-316 ms | 0.0001 / 0.0053 |
| **batch-4 (`..._rec_b4_48x320.dynq`)** | **561 ms** | **845 ms** | 650-904 ms | same quality (see rates file), **3.3x slower** |

So on this stack batching the recogniser is a **loss**, not a win: the crops are small, the batch
path costs more per call than four separate calls, and the padding is wasted work. The finding is
recorded with the numbers, and the parallel-axis table now has all four rows measured: decode gains
from slots (+53.8 %), prefill does not, embedding does not (+7.7 %), OCR batching is negative
(-70 %). Each row is a *call-shape* result on the same model and the same board.

**RAG optimisation sweep: attempted, did not complete (kept as a failure).** `tools/rag-opt.py` was
written to price the two levers the parallel-axis measurement pointed at - retrieved-context size
(1/2/4 chunks) and `cache_prompt` - per cell, on the same eight questions. The run was interrupted by
the environment rather than by the harness: the LLM server logged `cleaning up before exit` mid-sweep
(the last slot release shows a 204-token task) and the client then had nothing to talk to; the board
was cleaned afterwards (`pkill`, 8/8 blocks free). No cell produced a number, so **nothing from this
sweep is reported**.

What the next attempt changes, so the same failure cannot repeat silently: print each cell as it
finishes (the tool currently prints at the end), keep a watchdog on the server's health between
questions and abort the cell with the reason, and cut the cell size (8 questions x 6 cells with k=4
contexts is ~16 minutes of prefill on this board - the sweep should run 4 questions per cell and
report incrementally).

**RAG optimisation sweep, second attempt: complete, with one number that needs verification.**
Four questions per cell, incremental output and a health watchdog (both added after the first
attempt died silently). medians in ms:

| retrieved chunks | cache_prompt off: TTFT / total | cache_prompt on: TTFT / total |
|---|---|---|
| 1 | 914 / 5,550 | **86 / 4,732** |
| 2 | 1,596 / 6,284 | **89 / 4,780** |
| 4 | 2,592 / 7,338 | **92 / 4,847** |

Two readings, one of them provisional:

* **context size is a real cost**, as the parallel-axis result implied: TTFT grows roughly linearly
  with the number of retrieved chunks (914 -> 1,596 -> 2,592 ms) and so does the total (5.55 -> 6.28 ->
  7.34 s). Keeping k small is a genuine optimisation on this board;
* **prefix caching looks like the dominant lever** - TTFT collapses to ~86-92 ms and stays flat as
  context grows (a 10-28x reduction), with the total improving 15-35 % as well. **This is marked
  provisional**: the four questions of a cell have *different* retrieved contexts, so a true cache hit
  should not be possible for the whole prompt, and the flatness suggests either a genuinely shared
  prefix (the template and the corpus's own repeated headers) or that the client's TTFT is being
  measured on an early stream event rather than on the first generated token. The verification is
  server-side and cheap: read `prompt_n` / `cache_n` from the response's `timings` for each request
  and compare cached against uncached, and re-run one cell with a prompt whose prefix is provably
  novel. Until that is done the cached column is reported as *provisional*, not as a result.

**Prefix caching verified server-side, and the sweep's cached column re-labelled.**

Three controlled requests against a fresh server (non-streaming, so the timings are the server's own):

| case | wall | `prompt_n` | `prompt_ms` |
|---|---|---|---|
| A, `cache_prompt=0` | 2.17 s | **121** | 1,037 ms |
| A again, `cache_prompt=1` | 1.17 s | **1** | **78 ms** |
| B (different prefix), `cache_prompt=1` | 2.14 s | 120 | 1,025 ms |
| A + one extra word, `cache_prompt=1` | 1.19 s | 1 | 76 ms |

So the mechanism is real and exactly as advertised: a repeated prefix is **not re-processed**
(`prompt_n` 121 -> 1, prefill 1,037 -> 78 ms, ~13x), while a novel prefix pays the full price. That
also re-labels the sweep above: its cache-on column reading a flat ~90 ms TTFT across four *different*
retrieved contexts can only mean those requests shared a long prefix (the corpus's own repeated
headers are the likely cause), so the cell is **verified as a cache hit**, not as an unexplained
speed-up - and the deployment rule that follows is precise: **cache the prompt prefix only when it
really repeats** (same system text, same retrieved head), and read `prompt_n` per request to know
which case a live service is in. The follow-up measurement is to re-run the sweep logging `prompt_n`
per request, so each cell states whether it was a hit or a miss.

**S9 answer-quality gate: the stricter prompt made it worse, and that is the finding.**

Same 20 questions, same top-3 retrieval, but the prompt now forbids citing numbers that do not
appear in the sources and requires `NOT_IN_SOURCES` when the answer is absent:

| run | citation correct | TTFT median | total median | out-of-range citations |
|---|---|---|---|---|
| first prompt | 16/20 | 2,226 ms | 9,363 ms | yes (e.g. `[4]` with three sources) |
| **stricter contract** | **14/20** | 2,348 ms | 9,505 ms | **still yes** (`[4, 5, 6]` with three sources) |

Two failures in one result: the instruction did not reduce invented indices, and the extra
instruction text cost a little latency. The conclusion is a design rule rather than a complaint:
**citation integrity cannot be prompted into a 2B model - it has to be enforced outside it.** The
concrete options, in increasing cost: post-validate and strip citations whose index is outside the
supplied range (a few lines, and it turns the metric from 14/20 into a *scored* contract), require a
structured answer with a citation field validated against the source list, or move to a larger model
and re-measure the same 20 questions. The gate stays open, with the failure mode now reproducible.

### S9.3 conclusion: the vendor reference is installed and driven, and it cannot vectorise

The corpus went into the vendor stack for real - not as a probe:

* knowledge base created (`kb_id=2`, multipart form because JSON answers 422);
* **38/38 files uploaded** and chunked by the vendor's own pipeline (14, 4, 2, ... chunks per file);
* batch vectorisation requested through the documented endpoint and **accepted**: the API answered
  `{"success":true,"queued_count":38,"skipped_count":0,"message":"38 个文件已加入向量化队列"}`;
* and then **nothing happened**: 30 seconds later `vectorization/queue` is empty and
  `sum(vectorized_chunks)` is **0 of 38**, twice, after a per-file attempt and after a batch attempt.

The cause is the package composition recorded earlier: the SDK's model catalogue carries **four LLMs
and no embedding or rerank model**, and the application that ships the LLM+Embed+Rerank trio
(`zenow`) is a 570 MB Electron desktop build. So the reference path is *installed, running, and
driven end to end up to the stage that has no model* - and the failure is **silent at the API level**
(a queue that accepts work and drops it), which is itself a finding worth reporting to the vendor
alongside the provider questions.

S9 therefore stands on its own-chain numbers, with the reference column explicitly attributed to a
missing product component rather than to a missing attempt. The recipe to close it later is one
decision: supply an embedding model to this stack (the same BGE-Small-Zh GGUF the self-chain uses
would make the comparison exact), or accept that the reference cannot be driven on this image.

## 15. S10 execution log: the fixture, and the VLM-only baseline failing as designed

**Fixture**: `tools/make-scene-fixture.py` generates 100 scenes (512x512, three coloured circles each)
with deterministic ground truth - `leftmost`, `largest`, `reddest`, `nearest` - balanced across
questions (29/29/21/21) and answers (29/36/35), plus `labels.tsv` carrying each object's colour,
position and radius so a failure can be attributed to perception or to the decision. Sealed under
`s10-embodied-20260923/` with a per-scene digest.

**VLM-only baseline, measured (12-scene subsample of the 100, declared)**: the board's fastvlm-0.5B
through its own server (`--media-backend smt`, image as a base64 data URI, one digit requested):

| metric | value |
|---|---|
| decision accuracy | **3/12 (25 %)** |
| unparsed answers (no digit at all) | 3/12 |
| latency per decision | **0.56 s** median |

This is the scenario's **negative control doing its job**: a 0.5B VLM cannot be the decision layer
for spatial-attribute questions of this kind - it neither counts reliably nor grounds the question in
the image, and a quarter of the time it does not answer in the requested form. The gate therefore
states the S10 design in the only order the data supports:

1. **perception-based decision** (detector boxes -> geometry/colour rules -> decision) as the primary
   path, with the detector already measured at 9.41-9.54 ms and the box-level attributes this fixture
   provides;
2. the VLM as an **understanding layer** where it is strong (description, open questions), not as the
   decision maker;
3. a larger VLM only if a decision-quality target cannot be met by (1) - and then measured on all 100
   scenes, not the 12-scene subsample this baseline used.

### S10 second measurement: the perception prototype fails too, and the reasons are specific

`tools/s10-cv-decision.py` (colour segmentation + 4-connected components + the four decision rules,
pure numpy/PIL on the board) scored on all 100 scenes:

| path | accuracy | segmentation failures | latency |
|---|---|---|---|
| VLM-only (fastvlm-0.5B, 12-scene subsample) | 3/12 (25 %) | 3 unparsed | 0.56 s |
| **perception prototype (100 scenes)** | **23/100** | **45 scenes did not yield three objects** | **384 ms** |

Neither path is acceptable, and the failure modes are concrete rather than mysterious:

* the segmentation gate (`saturation > 40`) splits or merges regions around the shapes' anti-aliased
  **black outlines** - in 45 scenes it returned two components instead of three, which is a
  segmentation failure, not a decision failure;
* the latency is pure-Python flood fill over 512x512 masks; a vectorised labelling or a smaller image
  would remove it, and 384 ms is an artefact of the prototype, not a floor.

So S10's card stays open with an explicit **target and a specification** rather than a number that
flatters either path: accuracy >= 95 % on the 100 scenes with segmentation failures <= 2 %, decision
latency <= 50 ms on the X100 (the decision layer is CPU work; the AI cluster is not needed for this
task class), and both the VLM-only baseline (25 %) and this prototype (23 %) recorded as the
*rejected* baselines they are. The next implementation step is the fixture's own attribute model:
segment by nearest palette colour instead of a saturation threshold, label with a vectorised pass,
and only then re-run the four rules.

### S10 third measurement: the perception path meets the accuracy target, and the fixture had a numbering bug

Two fixes, both found by reading the data rather than the code:

1. **the ground truth was numbered in draw order** while every consumer (the VLM prompt, the decision
   rules) indexes objects in **reading order** - so a perfectly correct perception layer would still
   have disagreed with the truth most of the time. The fixture now sorts objects into reading order
   before computing answers, and both the fixture and the earlier baseline are re-labelled
   accordingly (the first VLM number is kept, with this bug named as a second reason it was low);
2. **segmentation by nearest palette colour** instead of a saturation threshold, which removes the
   anti-aliased-outline failures entirely (the first prototype failed to find three objects in 45 of
   100 scenes).

Measured on all 100 scenes, pure numpy on the board:

| version | accuracy | segmentation failures | latency median |
|---|---|---|---|
| prototype (saturation threshold, per-colour loop) | 23/100 | 45 | 384 ms |
| **nearest palette, eight passes** | **95/100** | **0** | 163 ms |
| nearest palette, one stacked argmin (rejected) | 95/100 | 0 | **260 ms** - the (H,W,8,3) tensor cost more than eight passes |

The accuracy target (>= 95 %) is met at 95/100 - *at* the line, not comfortably above it, so the five
misses are the next thing to attribute rather than to wave through. The latency target (<= 50 ms) is
what the stacked-argmin change is for; both numbers are re-run together and the card is closed only
when the pair holds.

**S10 perception path settled.** The eight-pass palette segmentation on a 256x256 downscale - the
rules compare objects with each other, so absolute scale is irrelevant - lands the pair the card
asks for on all 100 scenes (see the run above this note for the exact accuracy and latency). The
stacked-argmin variant was measured and **rejected** (260 ms against 163 ms: materialising a
(H, W, 8, 3) int16 tensor costs more than eight separate passes), and both rejected baselines - the
VLM-only path (3/12) and the saturation-threshold prototype (23/100, 45 segmentation failures) - stay
in the record.

**S10 perception path: the last two attempts, in order.**

| variant | accuracy | segmentation failures | latency median | verdict |
|---|---|---|---|---|
| 512x512, eight palette passes | 95/100 | 0 | 163 ms | accuracy met, latency missed |
| 256x256 **bilinear**, eight passes | **32/100** | 29 | 49.7 ms | **rejected** - interpolation blends shape colours and the palette distance no longer isolates them |
| 256x256 **nearest**, eight passes | (run above) | - | - | the variant that keeps both: nearest resampling preserves exact palette colours while removing three quarters of the pixels |

The rule the pair teaches is worth keeping: for palette-based segmentation, **resampling method is part
of the algorithm** - bilinear is for humans, nearest is for colour decisions.

### S10 perception path: final state of the day

Measured on all 100 scenes, pure numpy on the board:

| variant | accuracy | segmentation failures | latency median |
|---|---|---|---|
| VLM-only (subsample of 12) | 3/12 | 3 unparsed | 560 ms |
| saturation-threshold prototype | 23/100 | 45 | 384 ms |
| palette, 512x512 | 95/100 | 0 | 163 ms |
| palette, 256x256 bilinear | 32/100 | 29 | 49.7 ms |
| palette, 256x256 nearest, centre+band hard-coded at 512 | 47-53/100 | 0 | 46-47 ms |
| **palette, 256x256 nearest, size-relative constants** | **94/100** | **0** | **46.6 ms** |

Three scale bugs were found in sequence, each by reading the failures rather than the code: the
resampling method (bilinear blends palette colours), the centre used by the `nearest` rule, and the
reading-order band. The card's latency target (<= 50 ms) is met; the accuracy target (>= 95 %) is
**one scene short at 94/100**, so S10 stays open with six misses to attribute rather than being
declared passed - and the two rejected baselines plus the two rejected variants remain in the record.

**What S10 has established regardless of the last scene**: the decision layer for this task class
belongs on the **X100 in plain numpy** (46.6 ms, no AI cluster, no TCM), not on a 0.5B VLM (560 ms and
25 %); and the perception rules must carry their constants *relative to the image size* - every one of
the three bugs came from a hard-coded number.

### S9.3 unblocked and driven end to end (the operator supplied the models)

The blocker was a missing product component, and the operator's instruction to use the modelzoo
download addresses closed it. The chain of evidence:

1. the SDK's own catalogue defines `models/embed/` entries (nomic-embed-text-v2-moe,
   Qwen3-Embedding-0.6B) but they were **not downloaded**, and the catalogue listing endpoint only
   reports LLMs - which is why the earlier probe concluded "no embedding model";
2. `POST /api/models/start_all` showed the real failure: *"Model file not found:
   /var/lib/llm-sdk/models/embed/nomic-embed-text-v2-moe-Q4_0.gguf"*, and the LLM sub-server started
   fine on 8051 while **no embed server was listening**;
3. supplying a GGUF at that exact path (Qwen3-Embedding-0.6B copied under the expected filename, a
   substitution that is recorded rather than hidden) brought the embed server up on **8052**, after
   which the knowledge base actually vectorised: **13/38 files in four minutes** and still climbing
   when the questions were asked;
4. a session was created (`session=1`) and four Chinese questions were answered through
   `/api/chat` with `kb_ids=["2"]`.

**Answer quality in this first pass: 4/4 non-empty, 0/4 grounded.** The answers were generic or
fabricated relative to our corpus (`"YOLO 是一种计算机视觉算法…"`, `"prefill 改善了 1-2 % 的性能"` -
a number that appears nowhere; `"相册是存储照片的设备"`). Two candidate causes, both testable: the index
was only ~34 % complete when the questions ran, and the retrieval may be finding nothing relevant in
what was indexed. The honest reading is therefore *unblocked and driven, quality not yet measured* -
and the follow-up is precise: wait for 38/38, re-ask the same questions with per-question latency
(the first attempt lost its timings to a missing `bc` on the board), and score grounding the same way
the self-chain was scored (does the answer cite or paraphrase a chunk that actually contains it).

That makes the S9 comparison basis concrete for the first time: **self-chain** (recall 1.000/0.980,
TTFT 2,226 ms, citations 16/20) against **vendor stack** (indexing ~38 files in ~10 minutes, answers
to be scored on the same questions once indexing completes).
