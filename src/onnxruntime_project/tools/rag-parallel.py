#!/usr/bin/env python3
"""RAG answer stage under concurrency: aggregate tokens/s, per-stream rate and TTFT.

    rag-parallel.py <port> <concurrency> <tokens> [prompt-bytes]

Builds a RAG-shaped prompt (~600 tokens of retrieved context plus a question) and fires
`concurrency` requests at the board's `llama-server` with `-np N` slots, measuring both the aggregate
rate and the per-stream rate. This is the invocation-logic axis: the same model and the same prompt,
only the call shape changes - and on this board a decode step is memory-bound, so batching several
streams into one weight pass is where the extra throughput comes from.
"""

import json
import sys
import threading
import time
import urllib.request


def one(port, prompt, tokens, results, idx):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps({"prompt": prompt, "n_predict": tokens, "temperature": 0.0,
                         "cache_prompt": False}).encode(),
        headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.load(resp)
    results[idx] = (time.time() - start, data.get("tokens_predicted", tokens),
                    data.get("timings", {}))


def main(argv):
    port = int(argv[1])
    conc = int(argv[2])
    tokens = int(argv[3]) if len(argv) > 3 else 64
    ctx_bytes = int(argv[4]) if len(argv) > 4 else 2400
    context = ("Source: measured on the board. " * (ctx_bytes // 30))[:ctx_bytes]
    prompt = (f"Use the context to answer.\n{context}\nQuestion: what was measured?\nAnswer:")
    results = [None] * conc
    threads = [threading.Thread(target=one, args=(port, prompt, tokens, results, i))
               for i in range(conc)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    total = sum(r[1] for r in results)
    per_stream = [r[1] / r[0] for r in results]
    prompt_ms = [r[2].get("prompt_ms") for r in results if r[2]]
    print(f"concurrency={conc} tokens_per_request={tokens} wall_s={wall:.1f}")
    print(f"aggregate_tps={total / wall:.2f} per_stream_tps_min={min(per_stream):.2f} "
          f"max={max(per_stream):.2f}")
    if prompt_ms:
        print(f"prefill_ms_min={min(prompt_ms):.0f} max={max(prompt_ms):.0f}")


if __name__ == "__main__":
    main(sys.argv)
