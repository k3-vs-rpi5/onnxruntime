#!/usr/bin/env python3
"""S9 stage 2: answer with citations, measure TTFT and citation correctness.

    rag-answer.py <chunks.jsonl> <embed-port> <llm-port> [questions] [top-k]

For each question (a sentence drawn from the corpus, so its source chunk is known) the script
retrieves the top-k chunks by cosine, builds a numbered-source prompt, streams the answer from the
board's `llama-server`, and records:

  * **TTFT** - wall time from request to the first streamed token (the user-visible latency);
  * **total** - wall time to the end of the answer, with the token counts the server reports;
  * **citation correctness** - whether the cited markers include the chunk the question actually came
    from, and whether the answer is non-empty. A model that answers without citing, or cites the
    wrong source, is a failure even when the prose looks right.
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
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


def stream_answer(port, prompt, max_tokens=96):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps({"prompt": prompt, "stream": True,
                                                  "n_predict": max_tokens, "temperature": 0.0,
                                                  "cache_prompt": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    ttft = None
    text, tokens = "", 0
    with urllib.request.urlopen(req, timeout=600) as resp:
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
            piece = obj.get("content", "")
            if piece and ttft is None:
                ttft = time.time() - start
            text += piece
            tokens = obj.get("tokens_predicted", tokens) or tokens
            if obj.get("stop"):
                break
    return text, ttft, time.time() - start, tokens


def main(argv):
    chunks_path = pathlib.Path(argv[1])
    eport, lport = int(argv[2]), int(argv[3])
    n_questions = int(argv[4]) if len(argv) > 4 else 20
    top_k = int(argv[5]) if len(argv) > 5 else 3

    rows = [json.loads(line) for line in chunks_path.read_text().splitlines()]
    rng = random.Random(11)
    pool = []
    for idx, row in enumerate(rows):
        for s in re.split(r"(?<=[.!?;])\s+", row["text"]):
            if 60 <= len(s) <= 200:
                pool.append((idx, s.strip()))
    rng.shuffle(pool)
    sample = pool[:n_questions]

    ttfts, totals, cited_ok, nonempty = [], [], 0, 0
    for want, question in sample:
        qv = embed_one(eport, question)
        ranked = sorted(range(len(rows)), key=lambda i: -cosine(qv, rows[i]["vector"]))[:top_k]
        sources = "\n".join(f"[{i + 1}] {rows[r]['text'][:280]}" for i, r in enumerate(ranked))
        # Stricter contract than the first round: only the supplied indices may be cited, and an
        # unanswerable question must say so. The first run cited [4] when three sources existed.
        prompt = ("Answer using only the sources below. Cite with the source number in square "
                  "brackets, and only numbers that appear below. If the sources do not contain the "
                  "answer, reply exactly: NOT_IN_SOURCES.\n\n"
                  f"{sources}\n\nQuestion: {question}\nAnswer:")
        text, ttft, total, tokens = stream_answer(lport, prompt)
        cited = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
        hit = (want in ranked) and bool(cited & {ranked.index(want) + 1})
        cited_ok += 1 if hit else 0
        nonempty += 1 if text.strip() else 0
        if ttft:
            ttfts.append(ttft)
        totals.append(total)
        print(f"q={question[:48]!r} ttft={ttft and round(ttft * 1000)} ms "
              f"total={round(total * 1000)} ms tokens={tokens} cited={sorted(cited)} hit={hit}")

    lines = [f"questions={len(sample)} top_k={top_k}",
             f"ttft_ms_median={statistics.median(ttfts) * 1000:.0f} "
             f"ttft_ms_p90={sorted(ttfts)[int(len(ttfts) * 0.9)] * 1000:.0f}",
             f"total_ms_median={statistics.median(totals) * 1000:.0f}",
             f"citation_correct={cited_ok}/{len(sample)}",
             f"nonempty_answers={nonempty}/{len(sample)}"]
    print("\n".join(lines))
    pathlib.Path("s9-answer-summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv)
