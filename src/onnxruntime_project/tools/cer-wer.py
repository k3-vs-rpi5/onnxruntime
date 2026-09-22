#!/usr/bin/env python3
"""Character- and word-error rate for the S12 ingestion gate.

    cer-wer.py <fixture-root> <engine-output-dir> [label]

Reads the fixture's ground truth from `pages/<page>.txt` and the engine's transcript from
`<engine-output-dir>/<page>.txt`, normalises both the same way (lower-case, whitespace collapsed,
punctuation dropped), and reports per-page-class CER and WER with an equal-weight mean over pages -
never a pooled average, because the classes are the experiment.
"""

import pathlib
import re
import sys

CLASSES = ("printed", "scanned", "tables")


def normalise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\.\t ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def rates(truth, hyp):
    t, h = normalise(truth), normalise(hyp)
    cases = []
    for cls in CLASSES:
        cases.append((cls, edit_distance(t, h) / max(len(t), 1),
                      edit_distance(t.split(), h.split()) / max(len(t.split()), 1)))
    return cases


def main(argv):
    root = pathlib.Path(argv[1])
    engine = pathlib.Path(argv[2])
    label = argv[3] if len(argv) > 3 else engine.name
    per_class = {cls: [] for cls in CLASSES}
    for truth_path in sorted((root / "pages").glob("*.txt")):
        page = truth_path.stem
        cls = page.split("-")[0]
        hyp_path = engine / f"{page}.txt"
        if not hyp_path.exists():
            print(f"missing transcript for {page}")
            continue
        truth = truth_path.read_text()
        hyp = hyp_path.read_text()
        t, h = normalise(truth), normalise(hyp)
        per_class[cls].append((page, edit_distance(t, h) / max(len(t), 1),
                               edit_distance(t.split(), h.split()) / max(len(t.split()), 1)))
    print(f"engine={label}")
    print(f"{'class':10s} {'pages':>5s} {'CER mean':>9s} {'CER max':>8s} {'WER mean':>9s} {'WER max':>8s}")
    for cls in CLASSES:
        rows = per_class[cls]
        if not rows:
            continue
        cer = [r[1] for r in rows]
        wer = [r[2] for r in rows]
        print(f"{cls:10s} {len(rows):>5d} {sum(cer) / len(cer):>9.4f} {max(cer):>8.4f} "
              f"{sum(wer) / len(wer):>9.4f} {max(wer):>8.4f}")
        for page, c, w in rows:
            print(f"    {page:14s} CER {c:.4f}  WER {w:.4f}")


if __name__ == "__main__":
    main(sys.argv)
