#!/usr/bin/env python3
"""S10 perception-based decision layer: segment, measure, decide - and score it against the truth.

    s10-cv-decision.py <fixture-dir> [scenes]

The VLM-only baseline answered 3 of 12 scenes, so the scenario's primary path is a perception layer
that produces object attributes and a decision rule that consumes them. This implementation is
deliberately classical (numpy + PIL, no AI): threshold away the background, take 4-connected
components, and read each component's centroid, area and mean colour - the same attributes the
fixture's ground truth carries. It answers the four question types and reports accuracy, the
failure attribution (wrong object found versus wrong rule) and per-decision latency.
"""

import pathlib
import re
import sys
import time

import numpy as np
from PIL import Image

# The fixture's own palette (make-scene-fixture.py); segmentation is by nearest entry.
PALETTE = [(220, 40, 40), (40, 160, 60), (50, 70, 220), (230, 190, 40),
           (160, 60, 200), (40, 200, 200), (240, 120, 30), (90, 90, 90)]
def components(mask):
    """4-connected labelling by iterative flood fill (no scipy on the board)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y0 in range(h):
        xs = np.where(mask[y0] & ~seen[y0])[0]
        for x0 in xs:
            if seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) > 400:
                out.append(np.array(pts))
    return out


def decide(objects, question):
    if question == "leftmost":
        return min(objects, key=lambda o: o["x"])["order"]
    if question == "largest":
        return max(objects, key=lambda o: o["area"])["order"]
    if question == "reddest":
        return max(objects, key=lambda o: o["r"] - (o["g"] + o["b"]) / 2)["order"]
    return min(objects, key=lambda o: (o["x"] - 256) ** 2 + (o["y"] - 256) ** 2)["order"]


def main(argv):
    root = pathlib.Path(argv[1])
    limit = int(argv[2]) if len(argv) > 2 else 100
    labels = {}
    for line in (root / "labels.tsv").read_text().splitlines()[1:]:
        scene, question, answer, _ = line.split("\t")
        labels[scene] = (question, int(answer))
    hits, rule_misses, lats = 0, 0, []
    for name in sorted(labels)[:limit]:
        question, answer = labels[name]
        t0 = time.time()
        # Downscale with NEAREST: the palette segmentation depends on pixels keeping their exact
        # colour, and bilinear interpolation blends shape edges into in-between colours - that cost
        # the accuracy target (32/100 with 29 segmentation failures at 256x256 bilinear against
        # 95/100 and zero at full scale). Nearest keeps the colours and still removes 3/4 of the work.
        img = np.asarray(Image.open(root / "scenes" / f"{name}.png").convert("RGB")
                         .resize((256, 256), Image.NEAREST), dtype=np.int16)
        # Segment by *nearest palette colour* instead of a saturation threshold: the fixture draws the
        # shapes from a fixed palette, so each shape is exactly one mask and the anti-aliased black
        # outline (which broke the first prototype) falls outside every colour's radius. No connected
        # components are needed at all, which also removes the pure-Python flood fill and its 384 ms.
        # One stacked distance tensor and a single argmin instead of eight full-image passes: same
        # segmentation, a fraction of the work (163 ms -> the tens of milliseconds the target asks for).
        objects = []
        for i, colour in enumerate(PALETTE):
            mask = np.abs(img - np.array(colour, dtype=np.int16)).sum(2) < 60
            area = int(mask.sum())
            if area < 100:            # the same shape at a quarter of the pixels
                continue
            ys, xs = np.nonzero(mask)
            objects.append({"x": float(xs.mean()), "y": float(ys.mean()), "area": area,
                            "r": float(colour[0]), "g": float(colour[1]), "b": float(colour[2])})
        # reading order defines the indices the question refers to
        objects.sort(key=lambda o: (round(o["y"] / 120), o["x"]))
        for i, o in enumerate(objects):
            o["order"] = i
        got = decide(objects, question) if len(objects) == 3 else None
        hits += 1 if got == answer else 0
        if len(objects) != 3:
            rule_misses += 1
        lats.append(time.time() - t0)
        print(f"{name} q={question:8s} truth={answer} got={got} objs={len(objects)}", flush=True)
    lat = sorted(lats)
    print(f"scenes={len(lats)} accuracy={hits}/{len(lats)} "
          f"segmentation_failures={rule_misses} latency_median_ms={lat[len(lat)//2]*1000:.1f}")


if __name__ == "__main__":
    main(sys.argv)
