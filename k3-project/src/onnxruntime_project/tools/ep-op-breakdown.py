#!/usr/bin/env python3
"""Full per-operator breakdown of an execution-provider trace, for one or more models.

The provider is closed, but it can write its own Chrome trace (`SPACEMIT_EP_DEBUG_PROFILE`), and that
trace names every operator inside its subgraphs. This turns those traces into the three views that
decide where optimisation could go:

* **family** - how much of the executed time is int8 matrix work (the only thing a TOPS figure
  applies to) and how much is activations, concatenation, transposition, quantisation and shape
  movement;
* **module** - the YOLO backbone / neck / head attribution, read off the operator names
  (`model.0` .. `model.22`);
* **operator** - the ranked single operators, so a family that looks expensive can be traced to the
  two or three nodes that carry it.

    ep-op-breakdown.py <trace.json> [<trace.json> ...]

Two cautions that belong with every number below, both established in this project:

* the profiler perturbs its own run (YOLOv8n: 13.07 ms instrumented against 9.54 ms not), so **shares
  are usable and absolute milliseconds are not** - the wall clock comes from the run log beside the
  trace, and the per-inference column is thread-summed node time, roughly wall x threads;
* the run count is taken from the subgraph executions in the trace, not guessed.
"""

import collections
import json
import os
import re
import sys

MATRIX = ("Conv", "Gemm", "MatMul")
QUANT = ("Quantize", "Dequantize", "PPQ")
MOVEMENT = ("Resize", "Slice", "Reshape", "Split", "Gather", "Tile", "Pad", "Expand")
ACTIVATION = ("Softmax", "Sigmoid", "Relu", "Mul", "Add", "Sub", "Div", "Hardswish", "Silu", "Exp")


def family_of(name):
    if any(k in name for k in MATRIX):
        return "conv/GEMM (int8 matrix work)"
    if "Transpose" in name:
        return "Transpose"
    if "Concat" in name:
        return "Concat"
    if any(k in name for k in QUANT):
        return "quant/dequant"
    if any(k in name for k in MOVEMENT):
        return "shape/data movement"
    if any(k in name for k in ACTIVATION):
        return "activation/softmax"
    return "other"


def module_of(name):
    match = re.search(r"/model\.(\d+)/", name)
    if not match:
        return "outside the model graph (EP boundary)"
    index = int(match.group(1))
    if index <= 9:
        return f"backbone (model.{index})"
    if index <= 21:
        return f"neck (model.{index})"
    return f"head (model.{index})"


def load(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def analyse(path):
    events = load(path)
    nodes = [e for e in events if e.get("cat") == "Node"]
    executions = len([e for e in events if e.get("cat") == "Session"
                      and "SpineSubgraph" in e.get("name", "") and not e["name"].endswith("_init")])
    if not nodes or not executions:
        return None
    tag = re.sub(r"-trace_[\d_]+\.json$", "", os.path.basename(path))
    log = re.sub(r"-trace_[\d_]+\.json$", ".log", path)
    if not os.path.isfile(log):
        log = os.path.join(os.path.dirname(path), f"{tag}.log")
    wall = None
    if os.path.isfile(log):
        with open(log, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("Average inference time cost total:"):
                    wall = float(line.split(":")[1].split()[0])
    family = collections.defaultdict(float)
    module = collections.defaultdict(float)
    per_op = collections.defaultdict(float)
    calls = collections.Counter()
    for e in nodes:
        duration = e.get("dur", 0)
        family[family_of(e["name"])] += duration
        module[module_of(e["name"])] += duration
        per_op[e["name"]] += duration
        calls[e["name"]] += 1
    total = sum(family.values())
    return {"tag": tag, "executions": executions, "wall": wall, "total_ms": total / 1000 / executions,
            "family": {k: (v / 1000 / executions, v / total) for k, v in family.items()},
            "module": {k: (v / 1000 / executions, v / total) for k, v in module.items()},
            "per_op": sorted(((v / 1000 / executions, calls[k] // executions, k)
                              for k, v in per_op.items()), reverse=True)}


def main():
    rows = [row for row in (analyse(path) for path in sys.argv[1:]) if row]
    if not rows:
        print("no node events found", file=sys.stderr)
        return 1
    for row in rows:
        wall = f"{row['wall']:.2f} ms" if row["wall"] else "unknown"
        print(f"===== {row['tag']}  subgraph executions={row['executions']}  wall={wall}  "
              f"thread-summed node time={row['total_ms']:.2f} ms/inference")
        print("  -- by family")
        for fam, (ms, share) in sorted(row["family"].items(), key=lambda kv: -kv[1][1]):
            print(f"     {share * 100:5.1f}%  {ms:8.3f} ms   {fam}")
        print("  -- by module")
        for mod, (ms, share) in sorted(row["module"].items(), key=lambda kv: -kv[1][1]):
            print(f"     {share * 100:5.1f}%  {ms:8.3f} ms   {mod}")
        print("  -- top 8 operators")
        for ms, count, name in row["per_op"][:8]:
            print(f"     {ms:8.3f} ms  {count:5d}x  {name}")
    print()
    header = f"{'model':22s} {'wall':>9s} {'node ms':>9s} {'matrix':>8s} {'non-matrix':>11s} {'top family':>26s}"
    print(header)
    for row in rows:
        matrix = row["family"].get("conv/GEMM (int8 matrix work)", (0, 0))[1]
        biggest = max((f for f in row["family"].items() if f[0] != "conv/GEMM (int8 matrix work)"),
                      key=lambda kv: kv[1][1], default=(("-"), (0, 0)))
        print(f"{row['tag']:22s} {row['wall'] or 0:9.2f} {row['total_ms']:9.2f} "
              f"{matrix * 100:7.1f}% {(1 - matrix) * 100:10.1f}% "
              f"{biggest[0] + f' {biggest[1][1] * 100:.0f}%':>26s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
