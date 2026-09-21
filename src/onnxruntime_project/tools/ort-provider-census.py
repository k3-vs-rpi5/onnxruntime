#!/usr/bin/env python3
"""Turn ONNX Runtime per-node profiles into the provider-coverage census.

The question this answers is the deployment one: of the frames we measured, which models hand work
back to the CPU EP, how much of the frame is that, and which operators are involved. A graph whose EP
claims every node is insensitive to ONNX Runtime's own intra-op pool; a graph with a fallback is not
(`OPTIMISATION2-20260921.md` §2 measured 2.32x on one of the eight models in that round), so this
census is the screening step before a model goes into a deployment.

    ort-provider-census.py <profile.json> [<profile.json> ...]

Input: the files `onnxruntime_perf_test -p <prefix>` writes (Chrome-trace JSON, one `cat=Node` event
per node execution), and - if present next to them - the run logs they were written with, whose
`Average inference time cost total` is the wall-clock frame.

Two things about that profile decide how it may be read, both measured on 2026-09-21:

* **the provider's own node spans are not wall clock.** They sum to more than the frame - 186.26 ms
  against a measured 25.91 ms on `vit_b_16`, 410 ms against 91 on the reference model at pool 1 -
  because a subgraph span covers work that the provider parallelises internally. The EP column is
  therefore reported as a span, never as a time, and the fallback share is computed against the
  *measured* frame from the run log, not against the sum of spans;
* **the CPU EP's node spans are read as real time**, because a CPU node's span is wall clock on the
  graph executor's thread. They reproduce the independent two-run estimate: `pp_liteseg` 21.67 ms of
  a measured 84.81 ms frame (25.6 %) against 26.8 % from the `-x 1`/`-x 8` pair in
  `OPTIMISATION2-20260921.md` §2.

Never quote these numbers as latency: they exist to say *where* a model's frame goes, not how long
it is.
"""

import collections
import json
import os
import re
import sys

CPU = "CPUExecutionProvider"

# Nodes the EP's own boundary inserts around a fallback region: a share attributable to the fallback
# but not an operator the graph declares.
BOUNDARY = ("Transpose", "DequantizeLinear")


def load(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def measured_frame_ms(path):
    """The wall-clock frame from the run log the profile was written with, if it is still there."""
    log = re.sub(r"\.json(_\d.*)?$", "", path) + ".log"
    if not os.path.isfile(log):
        return None, None
    with open(log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Average inference time cost total:"):
                return float(line.split(":")[1].split()[0]), log
    return None, log


def census(path):
    trace = load(path)
    events = [e for e in trace if e.get("cat") == "Node"]
    if not events:
        return None
    # The divisor is the number of inferences, and it must not be guessed from node names: a graph may
    # use the same node name several times per inference (Reshape appears 24x in the SigLIP2 encoder),
    # which would make the max-per-name count an arbitrary multiple. ONNX Runtime writes exactly one
    # `model_run` session event per inference, so count those; the node-name maximum is only a
    # fallback for a profile that carries no session events.
    per_iteration = sum(1 for e in trace
                        if e.get("cat") == "Session" and e.get("name") == "model_run")
    if not per_iteration:
        per_iteration = max(collections.Counter(e["name"] for e in events).values())
    by_provider = collections.defaultdict(float)
    per_op = collections.defaultdict(float)
    counts = collections.Counter()
    for event in events:
        provider = event["args"].get("provider")
        by_provider[provider] += event.get("dur", 0)
        if provider == CPU:
            op = event["args"].get("op_name", "?")
            per_op[op] += event.get("dur", 0)
            counts[op] += 1
    cpu_ms = by_provider.get(CPU, 0.0) / 1000 / per_iteration
    frame_ms, log = measured_frame_ms(path)
    tag = re.sub(r"\.json(_\d.*)?$", "", path.rsplit("/", 1)[-1])
    return {
        "tag": tag,
        "iterations": per_iteration,
        "span_ms": sum(by_provider.values()) / 1000 / per_iteration,
        "ep_span_ms": by_provider.get("SpaceMITExecutionProvider", 0.0) / 1000 / per_iteration,
        "cpu_ms": cpu_ms,
        "frame_ms": frame_ms,
        "cpu_share": (cpu_ms / frame_ms) if frame_ms else None,
        "log": log,
        "cpu_nodes": sum(counts.values()) / per_iteration,
        "cpu_graph_ops": {op: ms / 1000 / per_iteration
                          for op, ms in per_op.items() if op not in BOUNDARY},
        "boundary_ms": sum(ms for op, ms in per_op.items() if op in BOUNDARY) / 1000 / per_iteration,
    }


def main():
    rows = [row for row in (census(path) for path in sys.argv[1:]) if row]
    if not rows:
        print("no Node events in the given profiles", file=sys.stderr)
        return 1
    for row in rows:
        if row["frame_ms"]:
            share = (f"{row['cpu_share'] * 100:.1f} % of the measured {row['frame_ms']:.2f} ms frame")
        else:
            share = "share unknown (run log not found)"
        print(f"== {row['tag']}  CPU={row['cpu_ms']:.2f} ms  {share}  "
              f"CPU node executions/frame={row['cpu_nodes']:.1f}  "
              f"(provider spans sum to {row['span_ms']:.2f} ms, not wall clock)")
        for op, ms in sorted(row["cpu_graph_ops"].items(), key=lambda kv: -kv[1]):
            print(f"     {ms:7.2f} ms  {op}")
        if row["boundary_ms"]:
            print(f"     {row['boundary_ms']:7.2f} ms  (EP-boundary Transpose/DequantizeLinear)")
    print()
    print(f"{'model':44s} {'frame':>9s} {'CPU':>9s} {'share':>7s} {'nodes/frame':>11s}")
    for row in sorted(rows, key=lambda r: -(r["cpu_share"] or 0)):
        frame = f"{row['frame_ms']:.2f}" if row["frame_ms"] else "-"
        share = f"{row['cpu_share'] * 100:.1f}%" if row["cpu_share"] else "-"
        flag = "  <-- fallback" if (row["cpu_share"] or 0) > 0.02 else ""
        print(f"{row['tag']:44s} {frame:>9s} {row['cpu_ms']:9.2f} {share:>7s} "
              f"{row['cpu_nodes']:11.1f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
