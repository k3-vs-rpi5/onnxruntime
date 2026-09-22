#!/usr/bin/env python3
"""S11 metrics: clustering NMI/purity, duplicate precision/recall, and the search sanity check.

    s11-metrics.py <fixture-root> <embeddings.jsonl> [k]

Inputs: the album fixture (`images/`, `labels.tsv` with group and planted duplicate links) and the
vectors produced on the board by `ort-embed-image`. Duplicate detection uses a 64-bit average hash
computed here (no dependency beyond PIL/numpy) with a Hamming threshold, and is scored against the
planted duplicates; clustering is deterministic k-means over the L2-normalised vectors with k equal
to the number of known groups, scored as NMI and purity.
"""

import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image


def ahash(path, size=8):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)
    bits = (a > a.mean()).flatten()
    return sum(int(b) << i for i, b in enumerate(bits))


def hamming(a, b):
    return bin(a ^ b).count("1")


def nmi(clusters, groups):
    clusters = np.asarray(clusters)
    groups = np.asarray(groups)
    n = len(clusters)
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / n
        return -sum(pi * math.log(pi) for pi in p if pi > 0)
    hc, hg = entropy(clusters), entropy(groups)
    mi = 0.0
    for c in np.unique(clusters):
        for g in np.unique(groups):
            pij = np.mean((clusters == c) & (groups == g))
            if pij > 0:
                mi += pij * math.log(pij / (np.mean(clusters == c) * np.mean(groups == g)))
    return mi / max((hc + hg) / 2, 1e-9)


def kmeans(x, k, iters=25):
    idx = np.linspace(0, len(x) - 1, k).astype(int)
    cent = x[idx].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        for j in range(k):
            if (labels == j).any():
                cent[j] = x[labels == j].mean(0)
    return labels


def main(argv):
    root = pathlib.Path(argv[1])
    rows = [json.loads(l) for l in pathlib.Path(argv[2]).read_text().splitlines()]
    k = int(argv[3]) if len(argv) > 3 else 10

    names = [r["image"] for r in rows]
    x = np.asarray([r["vector"] for r in rows], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    labels = {}
    for line in (root / "labels.tsv").read_text().splitlines()[1:]:
        name, group, dup = line.split("\t")
        labels[name] = (group, dup)
    groups = np.asarray([int(labels[n][0][5:]) for n in names])

    # --- clustering ---
    clusters = kmeans(x, k)
    purity = 0
    for c in np.unique(clusters):
        members = groups[clusters == c]
        if len(members):
            purity += np.bincount(members).max()
    purity /= len(groups)

    # --- duplicates ---
    hashes = {n: ahash(root / "images" / f"{n}.png") for n in names}
    planted = [n for n in names if labels[n][1]]
    tp = fp = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            close = hamming(hashes[a], hashes[b]) <= 4
            truth = labels[a][1] == b or labels[b][1] == a
            if close and truth:
                tp += 1
            elif close and not truth:
                fp += 1
    seen = set()
    for n in planted:
        partner = labels[n][1]
        key = tuple(sorted((n, partner)))
        if key not in seen and hamming(hashes[n], hashes[partner]) <= 4:
            seen.add(key)
    recall = len(seen) / len({tuple(sorted((n, labels[n][1]))) for n in planted})
    precision = tp / max(tp + fp, 1)

    # --- neighbour sanity: planted pairs must be each other's nearest image ---
    sim = x @ x.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(1)
    pair_ok = sum(1 for n in planted if names[nn[names.index(n)]] in
                  (labels[n][1],) )

    out = [f"images={len(names)} groups={k}",
           f"clustering_nmi={nmi(clusters, groups):.3f}",
           f"clustering_purity={purity:.3f}",
           f"duplicate_precision={precision:.3f}",
           f"duplicate_recall={recall:.3f}",
           f"planted_pairs={len({tuple(sorted((n, labels[n][1]))) for n in planted})} "
           f"nearest_neighbour_hits={pair_ok}"]
    text = "\n".join(out) + "\n"
    (root / "metrics.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
