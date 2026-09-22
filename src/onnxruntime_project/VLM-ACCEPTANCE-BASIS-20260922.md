# The VLM acceptance basis: a contract, because the published row has no statistic

This is item **G4** of `NEXT-ROUND-ANALYSIS-20260922.md`. It adds no measurement; it writes down
which number this project will hold itself to on the VLM path, and why the vendor's published row
cannot be that number. It exists because two models have now been measured end to end and the
published cell stays undefined against every reading that could be applied to it
(`METHOD-ALIGNMENT-20260921.md` §7.2, `VLM-SERVICE-SHAPE-20260921.md` §7).

## 1. Why the published row cannot be the acceptance basis

The vendor's VLM table is headed `LLM 8 Core + VisionEncoder N Core/ms` and never says which quantity
the cell contains. The candidate readings differ by 40 % on the same board, model, server and flags:

| reading | fastvlm-0.5B 512^2, N=4 / N=8 | against the published 256.47 / 164.50 |
|---|---|---|
| the published row | 256.47 / 164.50 | - |
| our encoder-only measurement | 243.08 / 150.02 | -5.2 % / -8.8 % |
| client-side time to first token | 317.0 / 229.7 | +23.6 % / +39.6 % |
| end to end, 24 tokens | 1013.6 / 937.9 | - |

The 4 -> 8 core step is identical in all three framings (**~87-92 ms**), so the *scaling* is
reproduced whatever the unit is; only the level is ambiguous, and it matches under one reading - the
vision encoder's own inference time. On the second model (`Qwen3.5-0.8B`, 384²) the same ambiguity
gives different gaps again: our encoder is 25-29 % *faster* than the row while our time to first
token is 22-47 % slower. One unnamed statistic, two models, three readings: acceptance rule **R10**
applies, and the project therefore accepts the row only as an order-of-magnitude reference.

Two instrument facts belong with this, both measured:

* the server's own `image prefill (smt) processed in N ms` line **is not the vision stage cost**: it
  read 159/162 ms on fastvlm at four and eight encoder threads and 969/973 ms on Qwen3.5 - flat in
  the knob that defines the table's two columns - while client-side time to first token moved by
  87 ms and 53 ms respectively. It must never be quoted as the stage cost;
* on the VLM path the encoder width is **not** an environment variable: it is
  `vision_model.spacemit_ep_intra_thread_num` in the model directory's `config.json`. The ASR
  examples in the same documentation set it on the command line; setting it there on the VLM path
  leaves the banner unchanged and silently makes two different "columns" identical.

## 2. The named statistics (the acceptance set)

Every VLM number this project publishes carries its statistic name and this metadata: model, image
size, prompt and token count, **cache state**, slot count, encoder width, decoder threads, and which
side measured it (client or server). Current values, all sealed:

| # | statistic | instrument | fastvlm-0.5B 512² | Qwen3.5-0.8B 384² |
|---|---|---|---|---|
| S1 | **encoder-only latency** at width W | our `perf_test` on the vision ONNX, 10 reps, 3 runs | 243.08 / 150.02 ms (4/8) | 240.70 / 184.52 ms (4/8) |
| S2 | **time to first token**, cached prompt, image attached, vendor server and payload | client-side | 317.0 / 229.7 ms (4/8) | 409 / 357 ms (4/8; two runs each: 414.3/403.8 and 361.7/352.4) |
| S3 | **end-to-end per request** at a stated token count | client-side | 1013.6 / 937.9 ms (24 tokens, 4/8) | ~2.45 s/frame at one request (32 tokens: 1172 ms image ingest + 1278 ms decode) |
| S4 | **cache semantics / per-new-frame cost** | server counters | encoder re-runs on every frame (30/30 in a 30-frame loop); first frame 680.72 ms against a cached frame 285.44 ms | same behaviour |
| S5 | **multi-slot service aggregate**, 4 slots, 4 simultaneous image requests | vendor `llama-server`, client loop | **47.88 t/s** (encoder 8 + text 8); 42.14 (enc 4 + text 8); 32.22 (enc 2 + text 6); 32.06 (ASR-style pinned) | 15.40 (enc 8); 15.25 (enc 4) |
| S6 | text-only decode aggregate, 4 simultaneous, no image | vendor `llama-server` | 126.92 t/s | 41.19 t/s |
| S7 | **quality gate** | tensor-dump equivalence (precision/layout changes) or identical answers on a fixed prompt set (scheduling changes) | - | - |

## 3. The rules that follow

1. **S1 is the only official-comparable statistic.** Every report quotes it next to the published row
   with the ambiguity above spelled out, and states which of the two readings it is using.
2. **There is no published end-to-end target.** S3/S4 are planning numbers for a customer scenario,
   never a "versus official" row, because the vendor has not published one with a named statistic.
3. **A "better" claim requires the same statistic before and after**, >= 3 rounds, medians, and - when
   the AI cluster is involved - the TCM sample naming the workload beside it. A change that improves
   S5 but worsens S2 is reported as a trade, not a win.
4. **Cache state is never omitted**, and no number is quoted without the pair it belongs to (cold and
   cached, or S1 beside S2).
5. **Nothing is called the vision stage cost except S1**, and no banner or server log line is used as
   evidence that a width took effect - the width is set in `config.json` and verified by S1 moving.

## 4. What a change must show to be accepted on this path

1. S1 before/after at both widths the table defines (4 and 8 threads), medians of >= 3 rounds;
2. S2 and S5 before/after at four slots, four simultaneous image requests, with the same prompt
   set - the service shape is what decides, and the single-stream optimum (2,6) already lost 33 %
   there once (`MODEL-ALTERNATIVES-20260922.md` §12, §14.2);
3. the quality gate of column S7;
4. the deployment statement: which shape is being claimed (latency, throughput, or multi-slot
   service), since the three deployment rules do not agree with each other.

## 5. Open, and deliberately not closed by this document

* the vendor's measurement conditions for the VLM rows (the statistic, the cache state, the OS
  revision) remain a question for them; until answered, S1 stays the comparison and the row stays
  context;
* a third model is not needed for the basis to hold - the plan's rule is one or two models per
  category, and the two measured here already disagree with the row in two different directions,
  which is what makes R10 evidence rather than a precaution;
* the decoder-side numbers inherit the LLM-decode caveat (`METHOD-ALIGNMENT-20260921.md` §7.1): the
  published `tg128` rows are ~10 % above what the vendor's own binaries produce on this board, so
  any VLM decode claim uses our own A/B and quotes the published row as context only.
