#!/usr/bin/env python3
"""S10 fixture: 100 decision scenes with ground truth, for a vision-understanding-then-decide gate.

Each scene is a 512x512 image with three coloured shapes and a single question whose answer is
deterministic from the scene's own construction:

  * `leftmost`   - which shape sits furthest left,
  * `largest`    - which shape covers the most pixels,
  * `reddest`    - which shape has the highest red channel,
  * `nearest`    - which shape is closest to the image centre.

    make-scene-fixture.py <out-dir> [scenes]

Output: scenes/<id>.png, labels.tsv (question, answer, and the objects' own attributes so a failure
can be attributed to perception or to the decision), manifest.tsv with a digest per scene.
"""

import hashlib
import pathlib
import random
import sys

from PIL import Image, ImageDraw

S = 512
PALETTE = [(220, 40, 40), (40, 160, 60), (50, 70, 220), (230, 190, 40),
           (160, 60, 200), (40, 200, 200), (240, 120, 30), (90, 90, 90)]


def scene(seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (S, S), (245, 245, 240))
    d = ImageDraw.Draw(img)
    colours = rng.sample(PALETTE, 3)
    objects = []
    for i, colour in enumerate(colours):
        r = rng.randint(40, 90)
        x = rng.randint(r + 5, S - r - 5)
        y = rng.randint(r + 5, S - r - 5)
        d.ellipse([x - r, y - r, x + r, y + r], fill=colour, outline=(0, 0, 0), width=2)
        objects.append({"id": i, "colour": colour, "x": x, "y": y, "r": r})
    question = rng.choice(["leftmost", "largest", "reddest", "nearest"])
    if question == "leftmost":
        answer = min(objects, key=lambda o: o["x"])["id"]
    elif question == "largest":
        answer = max(objects, key=lambda o: o["r"])["id"]
    elif question == "reddest":
        answer = max(objects, key=lambda o: o["colour"][0] - (o["colour"][1] + o["colour"][2]) / 2)["id"]
    else:
        answer = min(objects, key=lambda o: (o["x"] - S / 2) ** 2 + (o["y"] - S / 2) ** 2)["id"]
    return img, question, answer, objects


def main(argv):
    out = pathlib.Path(argv[1])
    n = int(argv[2]) if len(argv) > 2 else 100
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    rows, labels = [], []
    for i in range(n):
        img, question, answer, objects = scene(1000 + i)
        name = f"scene{i:03d}"
        path = out / "scenes" / f"{name}.png"
        img.save(path)
        rows.append((name, hashlib.sha256(path.read_bytes()).hexdigest(), path.name))
        labels.append((name, question, answer,
                       ";".join(f'{o["id"]}:{o["colour"][0]},{o["colour"][1]},{o["colour"][2]}'
                                f'@{o["x"]},{o["y"]}r{o["r"]}' for o in objects)))
    with (out / "manifest.tsv").open("w") as fh:
        fh.write("scene\tsha256\tfile\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with (out / "labels.tsv").open("w") as fh:
        fh.write("scene\tquestion\tanswer\tobjects\n")
        for r in labels:
            fh.write("\t".join(map(str, r)) + "\n")
    print(f"{n} scenes with ground truth in {out}")


if __name__ == "__main__":
    main(sys.argv)
