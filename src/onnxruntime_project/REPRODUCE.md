# How to reproduce everything, from a clean machine

This is the closing document of the scenario programme: one path from an empty checkout to every
number in `ACCEPTANCE-20260923.md`, with the acceptance band for each step and the pitfalls that cost
time while the work was done. It assumes the workspace layout of this repository
(`/root/k3vsrpi5` with the `repo` manifest synced) and one K3 board reachable as `.ssh-k3`.

## 0. What is needed

| item | where it comes from |
|---|---|
| workspace toolchain | `spacemit-toolchain-linux-glibc-x86_64-v1.2.4/` (already in the workspace) |
| sealed vendor package | `.local/k3-tools/riscv64/spacemit-ort.riscv64.2.0.6.tar.gz`; its sha256 is checked against `sources.lock.yaml` by `scripts/build.sh` |
| board | Bianbu 4.0.6, ONNX Runtime 2.0.6, llama.cpp `0.1.7~a0+4`; the board must be **idle with 8/8 TCM blocks free** before any run |
| board packages | `tesseract-ocr ffmpeg llm-sdk yolo-demo` (`materials.sh list` prints them) |
| models and fixtures | `scripts/materials.sh fetch` |

## 1. Materials: one command, then verify

    projects/onnxruntime/scripts/materials.sh list      # what each scenario needs, and its source
    projects/onnxruntime/scripts/materials.sh fetch     # downloads anything missing, writes materials.lock
    projects/onnxruntime/scripts/materials.sh verify    # digest check; prints verified=<n> missing=0 mismatched=0

Fixtures are **generated**, not downloaded, and their generators are the source of truth:

    tools/make-ingest-fixture.py <dir> 10        # S12: 30 pages (printed/scanned/tables) + ground truth
    tools/make-scene-fixture.py  <dir> 100       # S10: 100 decision scenes + ground truth
    tools/make-album-fixture.py  <dir> 10 100    # S11: synthetic album (kept as a rejected fixture)
    # S11 natural frames: apt install ffmpeg; ffmpeg -i /usr/share/yolo-demo/<v>.mp4 -vf fps=2,scale=256:256 out-%04d.ppm
    # S9 corpus: this workspace's own reports and evidence READMEs, sealed with a digest manifest

## 2. Build: identity-checked, bit-reproducible

    projects/onnxruntime/scripts/build.sh

It asserts the fork commit and the sealed package digest, cross-builds `bin/pipeline`,
`bin/ocr-chain` and `bin/embed-image`, generates the three matrix probes, and writes
`build/SHA256SUMS` + `build/MANIFEST.txt`. The harness digests are reproducible: `pipeline` builds to
`b836e876…` on this host.

## 3. Delivery and the generic run path

    projects/onnxruntime/scripts/run.sh verify                 # offline digest check of build/
    projects/onnxruntime/scripts/run.sh smoke                  # deliver + one 8-thread session on yolov8n
    projects/onnxruntime/scripts/run.sh ime-probe               # the AI-cluster ceiling instrument
    projects/onnxruntime/scripts/run.sh instrument <board-script> [args…]

`instrument` is the general door: it pushes any script from `tools/`, runs it on the board with
`OUT=/tmp/instrument-run`, and pulls the raw output into a fresh evidence directory. Every device run
in this programme went through one of these two paths or the monitor's `board.sh`.

## 4. Scenario recipes, with their acceptance bands

| scenario | command (board side unless noted) | acceptance band |
|---|---|---|
| **S12** ingestion | `ocr-chain <det> <rec> <dict> pages/*.ppm -o out --threads 4 4` then host `tools/cer-wer.py <fixture> <out>` | printed CER <= 0.01 (met, 0.0001); tables <= 1.5x tesseract (met, 0.0053 vs 0.0188 - better); **scanned <= 1.5x tesseract (NOT met: 0.0061 vs 0.0020 = 3.05x, recorded as failing)**; >= 3 pages/s |
| **S11** album | `embed-image <encoder> *.ppm --size 256 --norm clip -o emb.jsonl`, then host `tools/s11-metrics.py` | clustering NMI >= 0.8; verbatim duplicates: **exact-hash gate**, P/R 1.000; embedding gate explicitly rejected for duplicates |
| **S10** decision | `python3 tools/s10-cv-decision.py <fixture> 100 <size>` | 384x384: accuracy >= 95 %, latency 94 ms (see the Pareto note in the acceptance record: 256x256 meets <= 50 ms with 94 %) |
| **S9** RAG | board: `llama-server -m Bge-Small-Zh… -t 8 --embeddings --pooling mean` + `llama-server -m MiniCPM5-2B… -t 8`; `tools/rag-eval.py`, `rag-answer.py`, `rag-opt.py`, `rag-parallel.py` | recall@1 >= 0.98 on corpus-derived questions; TTFT < 3 s; citation correctness reported (16/20 measured); **enforce citation integrity outside the model** |
| **S9.3** vendor RAG | `apt install llm-sdk`; `curl :8050/api/models/start_all`; `tools/s9-vendor.py <corpus> <questions> 8050` | acceptance basis is grounding on numeric-answer questions; the measured state is 0/5, unaccepted |
| **optimisation levers** | `rag-parallel.py` (slots), `rag-opt.py` (context size x caching), `ocr-chain --rec-batch 4` | decode +53.8 % at 4 slots; prefill does not scale; prefix caching 13x when the prefix truly repeats (`prompt_n` 121 -> 1); OCR b4 rejected (-70 %) |

## 5. Verifying the evidence, not just the runs

Every package under `/root/k3-evidence/onnxruntime/` carries a `README.md` whose digest table can be
checked mechanically:

    python3 - <<'PY'
    import hashlib, pathlib, re
    for d in sorted(pathlib.Path('/root/k3-evidence/onnxruntime').iterdir()):
        readme = d / 'README.md'
        if not readme.is_file(): continue
        listed = {m.group(1): (m.group(2), int(m.group(3)))
                  for m in re.finditer(r'^\| `([^`]+)` \| `([0-9a-f]{64})` \| (\d+) \|$', readme.read_text(), re.M)}
        files = {p.relative_to(d).as_posix(): p for p in d.rglob('*') if p.is_file() and p.name != 'README.md'}
        bad = sum(1 for rel,(dig,size) in listed.items() if rel in files and
                  (hashlib.sha256(files[rel].read_bytes()).hexdigest() != dig or files[rel].stat().st_size != size))
        print(f'{d.name:42s} files={len(files):4d} listed={len(listed):4d} bad={bad}')
    PY

## 6. Pitfalls that cost time here (each measured, each in the tools)

1. **Python on this board reaches only the CPU EP** - any Python ONNX Runtime number is a CPU number;
   every AI-cluster claim comes from a C++ harness that loads the provider explicitly, or from the
   vendor's own tooling.
2. **Every binary of the llama.cpp fork needs `-t 8`**, or it dies inside the SpacemiT thread pool.
3. **TCM hygiene**: check 8/8 free before a run; a killed vendor application can leak all eight
   blocks, and the documented `spacemit-tcm-smi -c` is the remedy (only with nothing running).
4. **Two sessions asking for eight threads each are refused** (`required 8, available 0`); budget the
   threads (4+4) and search the split when the stages differ in cost.
5. **Embedding endpoint**: single `input` only (a list returns 500) and inputs above 512 tokens are
   rejected - chunk to ~300 characters.
6. **Palette segmentation**: resampling is part of the algorithm (nearest, not bilinear) and constants
   must be relative to the image size (centre, line band) - three separate bugs came from this.
7. **`llm-sdk` needs its own embed model** at `models/embed/nomic-embed-text-v2-moe-Q4_0.gguf`; the
   catalogue listing hides embed models, and vectorisation is **silently dropped** when the embed
   server is not up.
8. **Fixtures can be wrong**: two S12 layout bugs produced a plausible 10-37 % "CER", and the S11
   synthetic album made duplicates indistinguishable. Both are kept in evidence as failures.
