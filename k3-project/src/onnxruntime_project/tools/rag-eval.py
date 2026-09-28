#!/usr/bin/env python3
"""S9 retrieval evaluation: chunk a corpus, embed it, and measure recall against known chunks.

Runs on the board against a `llama-server --embeddings` instance (the embedding model is the
vendor's BGE-Small-Zh GGUF; the server is started separately so the model loads once).

    rag-eval.py <corpus-dir> <out-dir> [chunk-chars] [queries]

The queries are drawn from the corpus itself, in two shapes, because that is the negative control the
scenario requires: a **verbatim sentence** (the retrieval task is easy by construction) and the same
sentence with its words **shuffled** (same content words, broken order). A retrieval stack that does
well on the first and collapses on the second is a bag-of-words index, not a semantic one - and both
numbers are reported rather than only the flattering one.

Output: <out-dir>/chunks.jsonl (text + vector per chunk), <out-dir>/summary.txt with recall@1/5/10
for both query shapes plus timings.
"""

import json
import pathlib
import random
import re
import sys
import time
import urllib.request


def chunks_of(text, size):
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        if len(cur) + len(para) + 1 > size and cur:
            out.append(cur)
            cur = ""
        cur = (cur + " " + para).strip() if cur else para
    if cur:
        out.append(cur)
    # Hard cap: the board's embedding server refuses any input above its 512-token context
    # ("input (1046 tokens) is larger than the max context size (512 tokens)"), and a single long
    # paragraph would otherwise slip through the paragraph-based packing above.
    capped = []
    for ch in out:
        while len(ch) > size:
            capped.append(ch[:size])
            ch = ch[size:]
        if ch:
            capped.append(ch)
    return capped


def embed(port, texts):
    # One string per request: this build's `/v1/embeddings` answers a single input and returns 500
    # for a list (measured 2026-09-22), so batching is the client's job.
    out = []
    for text in texts:
        out.extend(_embed_one(port, text))
    return out


def _embed_one(port, text):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/embeddings",
        data=json.dumps({"input": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    return [item["embedding"] for item in data["data"]]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


def recall(vectors, queries, expected, k):
    hits = 0
    for q, want in zip(queries, expected):
        scored = sorted(range(len(vectors)), key=lambda i: -cosine(q, vectors[i]))[:k]
        hits += 1 if want in scored else 0
    return hits / len(queries)


def main(argv):
    corpus = pathlib.Path(argv[1])
    out = pathlib.Path(argv[2])
    size = int(argv[3]) if len(argv) > 3 else 600
    n_queries = int(argv[4]) if len(argv) > 4 else 50
    port = int(argv[5]) if len(argv) > 5 else 8100
    out.mkdir(parents=True, exist_ok=True)

    chunks, sources = [], []
    for path in sorted(corpus.rglob("*.md")):
        for ch in chunks_of(path.read_text(errors="replace"), size):
            chunks.append(ch)
            sources.append(str(path.relative_to(corpus)))
    print(f"corpus: {len(chunks)} chunks from {len(set(sources))} files")

    t0 = time.time()
    vectors = []
    for i in range(0, len(chunks), 16):
        vectors.extend(embed(port, chunks[i:i + 16]))
    embed_s = time.time() - t0
    print(f"embedded {len(vectors)} chunks in {embed_s:.1f} s "
          f"({embed_s / max(len(chunks), 1) * 1000:.1f} ms/chunk)")
    with (out / "chunks.jsonl").open("w") as fh:
        for c, s, v in zip(chunks, sources, vectors):
            fh.write(json.dumps({"source": s, "text": c, "vector": v}) + "\n")

    rng = random.Random(7)
    sentences = []
    for idx, ch in enumerate(chunks):
        for s in re.split(r"(?<=[.!?;])\s+", ch):
            if len(s) >= 60:
                sentences.append((idx, s.strip()))
    rng.shuffle(sentences)
    sample = sentences[:n_queries]
    verbatim = [s for _, s in sample]
    shuffled = [" ".join(rng.sample(s.split(), len(s.split()))) for _, s in sample]
    want = [i for i, _ in sample]

    t0 = time.time()
    vq = embed(port, verbatim)
    sq = embed(port, shuffled)
    query_s = time.time() - t0
    lines = [f"chunks={len(chunks)} files={len(set(sources))} chunk_chars={size} queries={len(sample)}",
             f"embed_total_s={embed_s:.1f} ms_per_chunk={embed_s / max(len(chunks), 1) * 1000:.1f}",
             f"query_embed_s={query_s:.2f}"]
    for label, qs in (("verbatim", vq), ("shuffled", sq)):
        for k in (1, 5, 10):
            lines.append(f"recall@{k} {label} = {recall(vectors, qs, want, k):.3f}")
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
