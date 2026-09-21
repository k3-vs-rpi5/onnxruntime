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
