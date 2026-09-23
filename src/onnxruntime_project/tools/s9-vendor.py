#!/usr/bin/env python3
"""S9.3: drive the vendor knowledge-base stack with the sealed corpus and ask the same questions.

    s9-vendor.py <corpus-dir> <questions-file> [port]

Steps: create a knowledge base -> upload every file (multipart, built by hand because the client is
stdlib-only) -> vectorise each file -> wait for progress -> ask each question with `kb_ids` and
record the answer plus latency. The vendor stack asserts its own call shape (message as an object,
session_id as an integer, kb_ids as strings), which is why this is a script rather than a curl line.
"""

import json
import mimetypes
import pathlib
import sys
import time
import urllib.request
import uuid


def api(port, path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def upload(port, kb_id, path):
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/knowledge-bases/{kb_id}/files",
                                 data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def create_kb(port, name):
    """The vendor's create endpoint takes a multipart form (`name`, optional avatar), not JSON -
    JSON answers 422, which is how this was found."""
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="name"\r\n\r\n',
        name.encode(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/knowledge-bases", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def main(argv):
    corpus = pathlib.Path(argv[1])
    questions = [q.strip() for q in pathlib.Path(argv[2]).read_text().splitlines() if q.strip()]
    port = int(argv[3]) if len(argv) > 3 else 8050

    kb = create_kb(port, f"s9-corpus-{int(time.time())}")
    kb_id = kb.get("id") or kb.get("kb_id") or (kb.get("knowledge_base") or {}).get("id")
    print(f"kb_id={kb_id} (create returned: {list(kb)[:4]})", flush=True)

    files = sorted(corpus.rglob("*.md"))
    uploaded = []
    for f in files:
        try:
            r = upload(port, kb_id, f)
            fid = r.get("id") or (r.get("file") or {}).get("id")
            uploaded.append((f.name, fid))
        except Exception as exc:  # noqa: BLE001 - the point is to report, not to crash the sweep
            print(f"upload failed for {f.name}: {exc}", flush=True)
    print(f"uploaded={len(uploaded)}/{len(files)}", flush=True)

    for name, fid in uploaded:
        if fid is None:
            continue
        try:
            api(port, f"/api/knowledge-bases/{kb_id}/files/{fid}/vectorize", {})
        except Exception as exc:  # noqa: BLE001
            print(f"vectorise failed for {name}: {exc}", flush=True)
    time.sleep(30)
    try:
        prog = api(port, f"/api/knowledge-bases/{kb_id}/files")
        done = sum(1 for f in prog.get("files", []) if f.get("vectorized_chunks"))
        print(f"vectorised_files={done}/{len(uploaded)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"progress check failed: {exc}", flush=True)

    lats = []
    for q in questions:
        body = {"message": {"role": "user", "content": q}, "session_id": 0,
                "kb_ids": [str(kb_id)], "stream": False}
        t0 = time.time()
        try:
            r = api(port, "/api/chat", body)
            answer = r.get("answer") or r.get("content") or json.dumps(r)[:160]
        except Exception as exc:  # noqa: BLE001
            answer = f"ERROR {exc}"
        lats.append(time.time() - t0)
        print(f"q={q[:44]!r} t={lats[-1]:.2f}s a={str(answer)[:90]!r}", flush=True)
    if lats:
        lats.sort()
        print(f"questions={len(lats)} latency_median_s={lats[len(lats)//2]:.2f} "
              f"p90_s={lats[int(len(lats)*0.9)]:.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
