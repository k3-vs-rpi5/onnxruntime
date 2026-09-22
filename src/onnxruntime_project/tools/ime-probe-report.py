#!/usr/bin/env python3
"""Turn `board-ime-probe.sh` logs into the AI-cluster ceiling table.

The board script writes one perf_test log per (round, probe, thread count) row plus the claim and
control rows. This reads them back and does the arithmetic the roofline needs:

    GFLOP per inference = 2 * M * K * N / 1e9        (one MAC = two operations)
    achieved GOPS       = GFLOP / average_time

The headline per (probe, threads) is the **median of the independent rounds**, so a single fast or
slow run cannot become the number. The quoted comparison values are printed next to it only as
context, exactly as the reports do: the K1 `cpufp` figure (511.53 GOPS/core at 2 GHz, clock-scaled
here to 1.8 GHz) and the operator's stated ~15 TOPS for the K3 A100 cluster.

    ime-probe-report.py <log-directory> --probe p1024=1,1024,1024 --probe p2048=1,2048,2048
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

# `perf_test` prints the mean as "Average inference time cost total: <x> ms" and the latency
# distribution as "P50/P90 Latency: <x> s" when `-s` is on.
AVG = re.compile(r"^Average inference time cost total: ([0-9.eE+-]+) ms$", re.M)
P50 = re.compile(r"^P50 Latency: ([0-9.eE+-]+) s$", re.M)
P90 = re.compile(r"^P90 Latency: ([0-9.eE+-]+) s$", re.M)

QUOTED_K3_TOPS = 15.0          # the operator's stated K3 A100 dense-int8 specification
K1_GOPS_PER_CORE_AT_2GHZ = 511.53
K1_GOPS_PER_CORE_HERE = K1_GOPS_PER_CORE_AT_2GHZ * 1.8 / 2.0


def parse_log(path):
    text = path.read_text(errors="replace")
    out = {}
    for key, pattern in (("avg", AVG), ("p50", P50), ("p90", P90)):
        match = pattern.search(text)
        if match:
            out[key] = float(match.group(1))
    return out


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--probe", action="append", required=True,
                        metavar="TAG=M,K,N", help="e.g. p2048=1,2048,2048")
    args = parser.parse_args(argv)

    shapes = {}
    for spec in args.probe:
        tag, dims = spec.split("=", 1)
        m, k, n = (int(x) for x in dims.split(","))
        shapes[tag] = (m, k, n)

    root = Path(args.directory)
    rounds = {}
    for path in sorted(root.glob("r*-p*-t*.log")):
        head = path.stem.split("-")
        probe, threads = head[1], int(head[2][1:])
        rows = parse_log(path)
        if "avg" in rows:
            rounds.setdefault((probe, threads), []).append(rows)

    print("row                 rounds   avg median (ms)   achieved GOPS   vs 15 TOPS   per-core GOPS")
    for (probe, threads) in sorted(rounds, key=lambda key: (key[0], key[1])):
        rows = rounds[(probe, threads)]
        m, k, n = shapes[probe]
        gflop = 2 * m * k * n / 1e9
        med = statistics.median(row["avg"] for row in rows)          # milliseconds
        gops = gflop / (med / 1e3)
        spread = (max(row["avg"] for row in rows) - min(row["avg"] for row in rows)) / med
        print(f"{probe} t{threads:<2}          {len(rows):>2}       {med:>9.3f}      "
              f"{gops:>10.1f}      {gops / (QUOTED_K3_TOPS * 1e3) * 100:>8.1f} %   "
              f"{gops / threads:>9.1f}   (run spread {spread * 100:.1f} %)")
        if len(rows) > 1:
            for row in rows:
                print(f"    round: avg {row['avg']:.3f} ms"
                      + (f"  p50 {row['p50'] * 1e3:.3f}" if "p50" in row else "")
                      + (f"  p90 {row['p90'] * 1e3:.3f}" if "p90" in row else ""))

    print()
    for path in sorted(root.glob("census-*.log")):
        rows = parse_log(path)
        print(f"{path.name:<28} profiled"
              + (f"  avg {rows['avg']:.3f} ms" if "avg" in rows else "  FAILED (see the log)")
              + "  (per-node provider names: parse the census-*.json_*.json with"
                " ort-provider-census.py)")
    for path in sorted(root.glob("cpu-*.log")):
        rows = parse_log(path)
        print(f"{path.name:<28} " + (f"avg {rows['avg']:.3f} ms" if "avg" in rows
                                     else "FAILED (see the log)"))

    print(f"\ncontext: quoted K3 A100 peak {QUOTED_K3_TOPS} TOPS (not measured here); "
          f"K1 cpufp {K1_GOPS_PER_CORE_AT_2GHZ} GOPS/core at 2 GHz -> {K1_GOPS_PER_CORE_HERE:.0f} "
          f"GOPS/core at this board's 1.8 GHz, {K1_GOPS_PER_CORE_HERE * 8 / 1e3:.2f} TOPS for eight")


if __name__ == "__main__":
    main(sys.argv[1:])
