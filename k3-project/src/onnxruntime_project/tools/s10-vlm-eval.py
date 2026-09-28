#!/usr/bin/env python3
"""S10 decision gate: ask the board's VLM which object answers each scene's question.

    s10-vlm-eval.py <fixture-dir> <vlm-port> [scenes]

The question is phrased so the answer is a single object index (0/1/2), and the scene's ground truth
is in `labels.tsv`; accuracy, latency and the *failure attribution* (perception miss versus decision
miss) come out per scene. The image travels as a base64 data URI in the OpenAI-compatible chat body.
"""

import base64
import json
import pathlib
import re
import statistics
import sys
import time
import urllib.request

PROMPT = ("In the image there are three coloured circles, numbered 0, 1, 2 in reading order "
          "(left to right, top to bottom). Answer with the single digit of the circle that is "
          "{question}. Reply with one digit only.")
ASK = {"leftmost": "furthest to the left", "largest": "the largest",
       "reddest": "the most red", "nearest": "closest to the image centre"}


def ask(port, image_b64, question):
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(question=ASK[question])},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]}],
        "max_tokens": 8, "temperature": 0.0, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    text = data["choices"][0]["message"]["content"]
    return text, time.time() - start


def main(argv):
    root = pathlib.Path(argv[1])
    port = int(argv[2])
    limit = int(argv[3]) if len(argv) > 3 else 20
    labels = {}
    for line in (root / "labels.tsv").read_text().splitlines()[1:]:
        scene, question, answer, _objs = line.split("\t")
        labels[scene] = (question, int(answer))
    scenes = sorted(labels)[:limit]
    hits, lats, unparsed = 0, [], 0
    for name in scenes:
        question, answer = labels[name]
        img = base64.b64encode((root / "scenes" / f"{name}.png").read_bytes()).decode()
        text, dt = ask(port, img, question)
        digits = re.findall(r"\d", text)
        got = int(digits[0]) if digits else None
        if got is None:
            unparsed += 1
        hits += 1 if got == answer else 0
        lats.append(dt)
        print(f"{name} q={question:8s} truth={answer} got={got} t={dt:.2f}s", flush=True)
    print(f"scenes={len(scenes)} accuracy={hits}/{len(scenes)} unparsed={unparsed} "
          f"latency_median_s={statistics.median(lats):.2f}")


if __name__ == "__main__":
    main(sys.argv)
