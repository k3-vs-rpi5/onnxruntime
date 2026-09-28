#!/usr/bin/env python3
"""Generate the S12 document-ingestion fixture: printed, scanned and table pages with ground truth.

Deterministic (seeded), self-describing (a manifest with per-file sha256) and deliberately plain:
the point of the fixture is to measure an ingestion chain, not to look like a real scan corpus. The
scanned class is the interesting one - small rotations, sensor noise, a brightness gradient and JPEG
artefacts are exactly what an OCR chain has to survive, and the *printed* class is the negative
control where any two engines should agree closely.

    make-ingest-fixture.py <out-dir> [pages-per-class]

Output: pages/<class>-<nn>.{png,jpg} + pages/<class>-<nn>.txt (ground truth) + manifest.tsv.
Note: the host has DejaVu/Noto-Mono but no CJK font, so this fixture is Latin+digits; a Chinese page
set needs a CJK font and is recorded as a limitation rather than faked.
"""
import hashlib
import pathlib
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
W, H = 1240, 1754  # A4 at ~150 dpi
WORDS = ("system network kernel memory buffer pipeline latency throughput vector tensor "
         "cluster schedule barrier cache block region context stream session profile "
         "measure verify evidence report baseline acceptance regression").split()
HEADERS = ("item", "value", "unit", "note")


def sentence(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def render_text_page(seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(SERIF_BOLD, 40)
    body_font = ImageFont.truetype(SERIF, 28)
    d.text((90, 80), f"Report {seed % 1000:03d}", font=title_font, fill="black")
    y = 160
    lines = [f"Report {seed % 1000:03d}"]
    # The ground truth is built from *what is drawn*: every rendered line is appended as it is placed,
    # and the loop stops at the page's line budget. Two earlier fixture versions failed exactly here -
    # unwrapped text ran off the right edge, and wrapped text ran off the bottom, so the "truth" file
    # described text the image did not contain and the measured CER was a layout bug, not OCR error.
    line = ""
    n_lines = 0
    while y <= H - 140:
        words = sentence(rng, rng.randint(6, 12)).split()
        if n_lines % 6 == 5:
            words += [f"({rng.randint(10, 99)}.{rng.randint(0, 99)}", "ms)"]
        for word in words:
            candidate = word if not line else line + " " + word
            if d.textlength(candidate, font=body_font) <= 1050:
                line = candidate
            else:
                d.text((90, y), line, font=body_font, fill="black")
                lines.append(line)
                y += 62
                n_lines += 1
                line = word
        if line:
            d.text((90, y), line, font=body_font, fill="black")
            lines.append(line)
            y += 62
            n_lines += 1
            line = ""
    return img, "\n".join(lines)


def render_table_page(seed):
    rng = random.Random(seed + 5000)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    head_font = ImageFont.truetype(SERIF_BOLD, 30)
    cell_font = ImageFont.truetype(MONO, 26)
    x0, y0, cw, ch, rows = 90, 140, 265, 62, 16
    lines = ["\t".join(HEADERS)]
    for c, name in enumerate(HEADERS):
        d.text((x0 + c * cw + 12, y0 + 14), name, font=head_font, fill="black")
    for r in range(rows):
        vals = [f"row{r + 1}", f"{rng.randint(1, 999)}.{rng.randint(0, 99)}",
                rng.choice(("ms", "MB", "t/s", "%")), rng.choice(WORDS)]
        lines.append("\t".join(vals))
        for c, v in enumerate(vals):
            d.text((x0 + c * cw + 12, y0 + ch * (r + 1) + 14), v, font=cell_font, fill="black")
    for r in range(rows + 2):
        d.line([(x0, y0 + ch * r), (x0 + cw * len(HEADERS), y0 + ch * r)], fill="black", width=2)
    for c in range(len(HEADERS) + 1):
        d.line([(x0 + c * cw, y0), (x0 + c * cw, y0 + ch * (rows + 1))], fill="black", width=2)
    return img, "\n".join(lines)


def degrade(img, seed):
    """The scanned variant: rotate, blur, shade, noise, then JPEG."""
    rng = random.Random(seed + 9000)
    img = img.rotate(rng.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor="white")
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
    pix = img.load()
    for y in range(0, H, 2):
        shade = 1.0 - 0.10 * (y / H)
        for x in range(0, W, 2):
            r, g, b = pix[x, y]
            n = rng.gauss(0, 9)
            v = max(0, min(255, int((r * shade) + n)))
            for dx in (0, 1):
                for dy in (0, 1):
                    if x + dx < W and y + dy < H:
                        pix[x + dx, y + dy] = (v, v, v)
    return img


def main(argv):
    out = pathlib.Path(argv[1])
    per_class = int(argv[2]) if len(argv) > 2 else 10
    pages = out / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in ("printed", "scanned", "tables"):
        for i in range(per_class):
            seed = {"printed": 100, "scanned": 200, "tables": 300}[cls] + i
            img, truth = (render_table_page(seed) if cls == "tables" else render_text_page(seed))
            ext = "jpg" if cls == "scanned" else "png"
            if cls == "scanned":
                img = degrade(img, seed)
            name = f"{cls}-{i:02d}"
            path = pages / f"{name}.{ext}"
            if ext == "jpg":
                img.save(path, quality=55)
            else:
                img.save(path)
            (pages / f"{name}.txt").write_text(truth + "\n")
            rows.append((name, cls, path.name, hashlib.sha256(path.read_bytes()).hexdigest(),
                         hashlib.sha256((truth + "\n").encode()).hexdigest()))
    with (out / "manifest.tsv").open("w") as fh:
        fh.write("page\tclass\tfile\tsha256_image\tsha256_text\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"{len(rows)} pages in {pages} (fixture root {out})")


if __name__ == "__main__":
    main(sys.argv)
