#!/usr/bin/env python3
"""RAG answer-stage optimisation sweep: retrieved-context size x prefix caching.

    rag-opt.py <chunks.jsonl> <embed-port> <llm-port> [questions]

The parallel-axis measurement showed the long-context prefill is the cost that does not parallelise
(5.35 s for ~600 tokens, growing linearly with slots). This sweep measures the two levers that act on
that cost directly - how much context is retrieved (k chunks) and whether the server may reuse a
cached prefix (llama-server's `cache_prompt`) - and reports TTFT and total per (k, cache) cell, on the
same questions, so the deployment choice is a number rather than a preference.
"""

import json
import pathlib
import random
import re
import statistics
import sys
import time
import urllib.request


def embed_one(port, text):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings",
                                 data=json.dumps({"input": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["data"][0]["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5) + 1e-9)


def ask(port, prompt, cache, tokens=64):
    body = {"prompt": prompt, "stream": True, "n_predict": tokens, "temperature": 0.0,
            "cache_prompt": bool(cache)}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    ttft = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("content") and ttft is None:
                ttft = time.time() - start
            if obj.get("stop"):
                break
    return ttft, time.time() - start


def main(argv):
    rows = [json.loads(l) for l in pathlib.Path(argv[1]).read_text().splitlines()]
    eport, lport = int(argv[2]), int(argv[3])
    n = int(argv[4]) if len(argv) > 4 else 8
    rng = random.Random(5)
    pool = [(i, s.strip()) for i, r in enumerate(rows)
            for s in re.split(r"(?<=[.!?;])\s+", r["text"]) if 60 <= len(s) <= 200]
    rng.shuffle(pool)
    sample = pool[:n]
    print("k  cache  ttft_median_ms  total_median_ms")
    for k in (1, 2, 4):
        for cache in (0, 1):
            ttfts, totals = [], []
            for idx, q in sample:
                qv = embed_one(eport, q)
                ranked = sorted(range(len(rows)), key=lambda i: -cosine(qv, rows[i]["vector"]))[:k]
                src = "\n".join(f"[{i+1}] {rows[r]['text'][:280]}" for i, r in enumerate(ranked))
                ttft, total = ask(lport, f"Answer using the sources, cite them.\n{src}\nQ: {q}\nA:", cache)
                if ttft:
                    ttfts.append(ttft)
                totals.append(total)
            print(f"{k}  {cache}      {statistics.median(ttfts)*1000:.0f}          "
                  f"{statistics.median(totals)*1000:.0f}")


if __name__ == "__main__":
    main(sys.argv)
