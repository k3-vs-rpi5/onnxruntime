#!/usr/bin/env python3
"""Generate the S11 album fixture: 1,000 images in known groups, with planted duplicates.

Deterministic and self-describing, in the same spirit as the S12 page fixture: the point is a
labelled set, not pretty pictures. Ten base renders (reusing the *text* of the S12 pages via PIL's
own drawing would tie two fixtures together, so these are generated abstractly) are transformed into
100 variants each - brightness, blur, rotation, crop, JPEG quality - and 20 duplicates are planted
verbatim, so duplicate detection has both a precision and a recall denominator.

    make-album-fixture.py <out-dir> [groups] [per-group]

Output: images/<group>-<nnn>.png, labels.tsv (group, duplicate-of), manifest.tsv with a sha256 per
image.
"""

import hashlib
import pathlib
import random
import sys

from PIL import Image, ImageDraw, ImageFilter

W = H = 256


def base_image(seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), (rng.randint(180, 255), rng.randint(180, 255), rng.randint(180, 255)))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(6, 14)):
        x0, y0 = rng.randint(0, W - 40), rng.randint(0, H - 40)
        x1, y1 = x0 + rng.randint(20, 120), y0 + rng.randint(20, 120)
        colour = (rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 200))
        if rng.random() < 0.5:
            d.rectangle([x0, y0, x1, y1], fill=colour, outline=(0, 0, 0))
        else:
            d.ellipse([x0, y0, x1, y1], fill=colour, outline=(0, 0, 0))
    return img


def variant(img, rng):
    out = img.copy()
    if rng.random() < 0.6:
        out = out.rotate(rng.uniform(-12, 12), resample=Image.BICUBIC, fillcolor="white")
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))
    if rng.random() < 0.6:
        factor = rng.uniform(0.6, 1.4)
        out = out.point(lambda v: max(0, min(255, int(v * factor))))
    if rng.random() < 0.4:
        box = (rng.randint(0, 32), rng.randint(0, 32), W - rng.randint(0, 32), H - rng.randint(0, 32))
        out = out.crop(box).resize((W, H))
    return out


def main(argv):
    out = pathlib.Path(argv[1])
    groups = int(argv[2]) if len(argv) > 2 else 10
    per_group = int(argv[3]) if len(argv) > 3 else 100
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows, labels = [], []
    for g in range(groups):
        base = base_image(1000 + g)
        dup_pool = []
        for i in range(per_group):
            rng = random.Random(g * 100000 + i)
            img = variant(base, rng)
            dup_of = ""
            if i < 2:                      # keep the first two variants as duplicate sources
                dup_pool.append(img.copy())
            elif i < 4:                    # ... and plant verbatim copies of them
                img = dup_pool[i - 2].copy()
                dup_of = f"group{g:02d}-{i - 2:03d}"
            name = f"group{g:02d}-{i:03d}"
            path = images / f"{name}.png"
            img.save(path)
            rows.append((name, hashlib.sha256(path.read_bytes()).hexdigest(), path.name))
            labels.append((name, f"group{g:02d}", dup_of))
    with (out / "manifest.tsv").open("w") as fh:
        fh.write("image\tsha256\tfile\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with (out / "labels.tsv").open("w") as fh:
        fh.write("image\tgroup\tduplicate_of\n")
        for r in labels:
            fh.write("\t".join(r) + "\n")
    print(f"{len(rows)} images, {groups} groups, {sum(1 for r in labels if r[2])} planted duplicates")


if __name__ == "__main__":
    main(sys.argv)
