# Seventh round: VLM service shape - the decoder batches, the vision stage does not

Round 6's VLM work (§7.2 of `METHOD-ALIGNMENT-20260921.md`) settled what the published row measures;
it says nothing about what a *service* does under load. That is this round's question, and the answer
is not the same for the two halves of a VLM request.

Setup: the vendor's own model (`fastvlm-mm-0.5b-q4_1`), their `llama-server` with
`--media-backend smt -ctk f16 -ctv f16 -t 8 -c 1024`, four slots, one image (512² input) per
request, a different prompt per request so nothing is a prompt-cache hit. Board state for every
number: 2.0 GHz userspace governor, 33-36 °C, TCM 8/8 sampled during every run, no leftover process
afterwards and idle between runs (user CPU 0-1.6 %, 1-2 runnable tasks). Note that the 1-minute load
average still reads 3.4-3.8 right after a four-request sweep because the burst has not decayed yet,
which is why the state line uses instantaneous CPU and a process check rather than load average.
Raw: `/root/k3-evidence/onnxruntime/baseline-20260921/vlm-service-shape/`.

## 1. The text decoder batches, and it is the largest lever this board has for LLM work

Text-only requests through the native `/completion` endpoint with `ignore_eos`, so each request
produces exactly 64 tokens; the image is not in the request at all.

| simultaneous requests | aggregate | per stream | median latency |
|---|---|---|---|
| 1 | 43.68 t/s | 43.7 t/s | 1464 ms |
| 2 | 72.26 t/s | 35.2 t/s | 1769 ms |
| 4 | **126.92 t/s** | 35.4 t/s | 2014 ms |

**2.91x aggregate at four concurrent for -19 % per stream.** The mechanism is the one round 5 found
for a CNN batch (four rows share one pass over the weights) applied to decode: every token reads the
model's weights, so N sequences decoded together share that read. Latency grows sub-linearly - the
fourth stream costs the first one 25 % of its per-token rate and buys 1.8x aggregate.

## 2. The image path batches much less, and the encoder width matters at every load

Same shape, but each request carries the image, and the vision encoder's width is set where round 6
found it belongs - `vision_model.spacemit_ep_intra_thread_num` in the model directory's `config.json`.

| encoder threads | 1 request | 2 requests | 4 requests |
|---|---|---|---|
| 8 | **27.10 t/s**, 1179 ms | **37.03 t/s**, 1727 ms | **47.88 t/s**, 2627 ms |
| 4 | 25.16 t/s, 1271 ms | 33.14 t/s, 1930 ms | 41.89 t/s, 3008 ms |

(aggregate tokens/s, median request latency; 32 tokens per request)

Two readings:

* **eight encoder threads win at every load**: +7.9 % aggregate at one request, +9.9 % at two,
  +13.8 % at four, and 92-381 ms less latency. It is a one-line change to the model's `config.json`,
  no code;
* **the image path gains 1.77x at four concurrent against 2.91x for text.** Half of an image request's
  wall time is not decode, and that half does not batch.

## 3. What the difference is: the vision stage contends, and it re-runs every time

The server's own counters say where the non-batching half is. Under four-way load the per-slot
`prompt eval time` - the image-token ingest that follows the encoder - goes from **249 ms unloaded to
630 ms** with eight encoder threads and **727 ms** with four, while the per-stream decode rate only
falls to 35 t/s. So the contended resource is the vision stage on the same eight AI cores the decoder
is streaming weights on, not the decode itself.

At four concurrent, an image token costs **82.1 ms** of wall time against **31.5 ms** for a text
token - a 2.6x surcharge for being an image request. Part of that is the encoder's own 152 ms, and
part is that the media stage runs on **every** request: round 4 measured `smt_encode_media` firing 30
times in a 30-frame loop over the same image, with only the prompt tokens cached.

## 4. What this round changes

Accepted, no code change anywhere:

1. **A VLM service on this board is shaped by concurrency, not by core count**: one server, four
   slots, requests queued. Worth 2.91x on text-heavy work and 1.77x on image work at four, with
   sub-linear latency growth. This is the opposite of the ONNX EP's rule from round 5 (where
   concurrency was flat and splitting cores was negative) - the two stacks answer the same question
   differently, and both answers are measured;
2. **the vision encoder's width belongs in the model's `config.json` and should be 8** on this board:
   +14 % aggregate at four concurrent and 92-381 ms of latency at every level;
3. **the live lever for image-heavy VLM workloads is the vision stage's scheduling** - it contends
   with decode for the same eight cores and re-encodes every frame. Anything that queues, rate-limits
   or overlaps it is worth up to the 2.6x surcharge above, and is not measured yet. Named as the next
   question rather than guessed at.

Method note, kept because it invalidated a first table: the first sweep of this round did not set
`ignore_eos`, so requests stopped between 6 and 32 tokens and the resulting aggregate numbers
(20.56 / 22.41 / 30.40 t/s) were dominated by *how long each answer happened to be*, not by
throughput - 25-35 % below the corrected table. An aggregate tokens/s is only a measurement when
every request produces the same number of tokens.

## 5. Scheduling the vision stage: one rejection and two negatives

Section 3 named the vision stage as the contended resource. Three ways to take it off the critical
path were tried; none of them recovers throughput, and each result is worth keeping so the next
attempt starts elsewhere.

**Offload it to the X100 cluster - rejected by 30x.** The encoder is a 512² fp16 ViT. On the AI
cluster it is 243.02 ms at four threads and 152.01 ms at eight. Run it on the X100 with ONNX Runtime's
own CPU EP and the RVV kernels (`-e cpu -x N` on the same file): **7903.83 ms at four threads and
4561.01 ms at eight.** The two clusters are not interchangeable for this stage, and the 10-25 %
interference budget of round 5 never gets a chance to matter.

**Queue the arrivals - no throughput.** Four image requests, same server, arriving 0 / 400 / 800 ms
apart:

| stagger | aggregate | min latency | median latency |
|---|---|---|---|
| none | **48.23 t/s** | 2534 ms | 2604 ms |
| 400 ms | 47.09 t/s | 1515 ms | 2210 ms |
| 800 ms | 35.01 t/s | 1253 ms | 1643 ms |

Throughput does not improve - and the 800 ms row is *lower* only because 3 x 800 ms of deliberate
waiting is inside its wall clock, which is also why its per-request latencies are the best of the
three. Reading: queueing is a fairness knob (the last arrival gets a quiet server), not a throughput
lever. The contention is not an arrival-order problem.

**Make the encoder narrower to leave room for decode - no.** Completing the curve at four simultaneous
requests: 2 threads 34.65 t/s, 4 threads 41.89 t/s, 8 threads **47.88 t/s**. Monotone, so the encoder
should take all eight; there is no interleaving point that beats giving it the cluster and letting
decode wait.

**And the encode cannot be skipped.** There is no media-encode cache in this path: the vendor's own
source computes an FNV hash of the image bytes and uses it as the media item's identity for the
prompt-token cache, but the encoder itself runs per request - measured again here (a second identical
request's time to first token is 228.0 ms against 229.7 ms for the first) and in round 4
(`smt_encode_media` fired 30 times in a 30-frame loop over one image). No server flag turns it off.

What is left for image-heavy VLM work is therefore **service policy, not runtime configuration**:
send fewer frames (an embodied agent at 1 fps spends 152 ms of every second, 15 % of the cluster, on
an encoder that could be run at 2-3 fps), or accept the 2.6x image surcharge of section 3. That is a
product decision with a measured price on it, which is as far as this round can take it.

## 6. The harness defect this round found, and why the numbers survived it

While re-running the cells to write this document, **two `llama-server` processes survived their
cell's `SIGTERM` and kept port 8099.** That matters more than it sounds: if the next cell's server had
started against that port it would have failed to bind, the client would have talked to the *previous*
configuration, and the cell's tag would have said otherwise - a silent wrong answer, not a crash.

The evidence survived because every cell's server log carries two things that make it
self-checking: the `[SMT][vision] Spacemit EP enabled (... INTRA_THREAD_NUM=N)` banner of the
configuration that actually served it, and exactly one `listening on` line. All six image cells and
the three stagger cells check out, and none of the logs contains a bind failure - so each cell was
served by its own server, and the affected processes were leftovers created *after* their cell's data
was taken.

`tools/board-vlm-service-shape.sh` is the fixed harness: it stops any server with `SIGTERM`, waits,
escalates to `SIGKILL`, refuses to continue if one survives, refuses to start if the port still
answers, prints the banner into the cell record, and asserts exactly one server process before the
client runs. Re-measuring one cell through it reproduced the published value (34.71 t/s against
34.65), which is the check that the hardening did not change the measurement.

## 7. Second model: what generalises and what does not

The category rule is "one or two models per category", so the round's conclusions were re-measured on
the vendor's **Qwen3.5-0.8B** package (`Qwen3.5-0.8B.tar.gz`, 977,343,463 bytes, sha256
`be44a9d2...`; text `qwen3_5vl_0.8b-text-q41.gguf` `143617d3...`; vision-384
`51b71bdc...`; its own `config.json` `26a1971e...`, which ships `input_size: 384` and
`spacemit_ep_intra_thread_num: 4`), pushed to the board and measured with the same harness. Raw:
`vlm-second-model/` (29 files).

### Two things had to be fixed before it would run at all

* **the documented reasoning flags are not optional for this family.** Without
  `--reasoning-budget 0 --reasoning off` every request "succeeds" with 32 tokens generated and an
  **empty** answer: the model's output goes to the reasoning channel. With them, the same request
  answers correctly. A VLM service that skips these flags looks like it works and returns nothing;
* **the harness must derive the model directory from the config directory** (now fixed in the tool).
  Filling only the config path left the text model at its previous default, and the server caught it
  itself: `SMT hidden size (1024) does not match model embedding size (896)`.

### The encoder, the served stage and time to first token

| measurement | 4 threads | 8 threads |
|---|---|---|
| encoder alone, 384² (`perf_test`, our instrument) | 240.70 ms | **184.52 ms** |
| encoder alone, 224² / 768² (8 threads) | - | 62.04 ms / 786.18 ms |
| encoder alone, 384² at **one** thread | 600.37 ms | - |
| **served** `image prefill (smt)` stage, 384² | 973 / 965 ms | **969 / 963 ms** |
| served stage, 224² (8 threads) | - | 367 / 365 ms |
| time to first token (cached prompt) | 414.3 / 403.8 ms | **361.7 / 352.4 ms** |
| published row for this model, 384² | 340.42 | 245.61 |

Three readings:

* **the encoder width generalises, weakly**: eight threads beat four on time to first token by 52.6 ms
  here against 87.3 ms on fastvlm, and by 1-2.5 % of aggregate under load against 8-14 % there. Same
  direction, smaller prize;
* **the publisher's row still does not identify a statistic.** Our encoder-only number is 25-29 %
  *faster* than the row (on fastvlm that reading matched within 5-8 %), our time to first token is
  22-47 % slower, and the server's own stage is 2.9-4x slower still. Two models, two different gaps,
  one unnamed statistic: acceptance rule R10 is now confirmed on a second model rather than resting
  on one;
* **a new instrument finding: the served stage does not move with the encoder width in either model**
  (fastvlm 159/162 ms at four and eight threads; Qwen3.5 969/973 ms), yet time to first token does
  (-87 ms, -53 ms). So the banner's `SPACEMIT_EP_INTRA_THREAD_NUM` is not evidence that the vision
  session used that many threads - the setting reaches something in the path, but not the stage the
  log timestamps. The 224² probe is what proves the stage is encoder work at all: 969 -> 367 ms
  against an area ratio of 0.34.

### Under load, the second model is worse, not different

| measurement | fastvlm-0.5B | Qwen3.5-0.8B |
|---|---|---|
| text-only decode, 1 / 2 / 4 simultaneous | 43.68 / 72.26 / 126.92 t/s | 21.94 / 31.50 / 41.19 t/s |
| per-stream decode rate at 4 | 35.4 t/s | 13.3 t/s |
| gain at four simultaneous (text) | **2.91x** | **1.88x** |
| image requests, encoder 8, 1 / 2 / 4 simultaneous | 27.10 / 37.03 / 47.88 t/s | 11.47 / 13.33 / 15.40 t/s |
| image requests, encoder 4, 1 / 2 / 4 | 25.16 / 33.14 / 41.89 t/s | 11.22 / 13.20 / 15.25 t/s |
| gain at four simultaneous (image) | 1.77x | 1.34x |

The direction is the same everywhere - concurrency pays, the encoder should take eight threads, the
image path pays less than the text path - and the magnitudes are the model's own. On this model the
served frame at one request is 1172 ms of image ingest against 1278 ms of decode for a 32-token
answer, so the vision stage is now as large as the thing it contends with, which is why its
concurrency gain is the smaller of the two.

## 8. What the served image stage actually contains (no new material needed)

Section 7 left a contradiction: the server's `image prefill (smt)` line reads 969 ms while the
encoder measured alone is 184.52 ms, and the stage does not respond to the encoder's thread count.
Three instruments already in hand settle it.

**The server's own execution-provider trace.** Starting the server with
`SPACEMIT_EP_DEBUG_PROFILE` makes the media worker write its own Chrome trace
(`vlm-second-model/qwen08b-server-ep-trace.json`). In a session with one warm-up and four image
requests it contains exactly **five** subgraph executions of 191.9-198.9 ms, i.e. **one ~195 ms
encoder run per request**, matching the standalone measurement. The 969 ms stage is therefore not the
encoder.

**A text-only control.** The same server, same session: a 230-token text prompt with no image costs
**1523.35 ms of prompt eval (6.62 ms/token)**. So the language model's ingest of a long prompt costs
about 6.6 ms per token on this board, and an image contributes ~160 tokens.

**The resolution variants already in the package.** Qwen3.5-0.8B ships 224/384/768 vision graphs; the
same request against the 224 variant:

| variant | image tokens in the prompt | served stage | prompt eval | time to first token |
|---|---|---|---|---|
| 384² | 162 | 969 / 963 ms | 1172.34 ms (7.24 ms/token) | 361.7 / 352.4 ms |
| 224² | 67 | 367 / 365 ms | 560.21 ms (8.36 ms/token) | 225.1 / 213.1 ms |

Fitting the two points: **stage ~= encoder + 4.7 ms per image token** - 195 + 4.7x162 = 956 ms against
969 measured, 62 + 4.7x67 = 377 ms against 367. So the stage is the encoder *plus the language
model's ingest of the image tokens*, which is why the encoder's thread count barely moves it and why
it scales with the image area: the area sets the token count.

Two consequences, both deployment-level and both free:

1. **The per-image cost is dominated by the number of image tokens, not by the encoder.** On this
   model one image token costs about 4.7 ms of prompt ingest; the encoder is 195 ms of a 969 ms
   stage - 20 %. Any lever aimed at the vision encoder (threads, affinity) can move at most that
   fifth;
2. **the resolution variant is the lever, and it is a quality/cost dial rather than a speed trick.**
   Moving from 384² to 224² cuts the image tokens from 162 to 67 and the served stage from 969 ms to
   367 ms, worth 137 ms of time to first token and ~600 ms of every frame in a streaming pipeline.
   Which variant is acceptable is a quality question this project has not measured; the *price* of
   each is now known.

For contrast, the same instrument on fastvlm-0.5B reads differently: its stage (159/162 ms) *is* its
encoder (152.01 ms standalone) and its image-token ingest appears separately in prompt eval
(249.34 ms for 95 tokens, 2.62 ms/token). The two model families cost the same thing in different
places, which is why a single "vision stage" number cannot be compared across them - and why the
published VLM table, whose statistic is still unnamed, stays an order-of-magnitude reference under
rule R10.
