#!/usr/bin/env python3
"""Variance gate for repeated scenario runs: median, spread, comparability verdict.

    variance.py label=v1 label=v2 label=v3 [--max-spread 0.03] [--band LO:HI] [--monitor a.csv ...]

The scenario rules say a difference below the measured spread is never an improvement, so the spread
must be computed and written down rather than remembered. This prints the median and relative spread,
decides whether the row is reportable, optionally checks a band, and marks the whole row
INCOMPARABLE when any monitor CSV shows an excursion (a competing process or a hot zone) - because a
measurement taken on a busy board is not a measurement of the workload.
"""

import argparse
import csv
import statistics
import sys


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("values", nargs="+")
    parser.add_argument("--max-spread", type=float, default=0.03)
    parser.add_argument("--band")
    parser.add_argument("--monitor", nargs="*", default=[])
    parser.add_argument("--expect", default="python3,llama-server,perf_test,ocr-chain,embed-image,pipeline",
                        help="process names that ARE the workload; anything else above 10 % is interference")
    args = parser.parse_args(argv)

    pairs = [v.split("=", 1) for v in args.values]
    label = pairs[0][0]
    values = [float(v) for _, v in pairs]
    med = statistics.median(values)
    spread = (max(values) - min(values)) / med if med else 0.0

    verdict = []
    verdict.append(f"REPORTABLE (spread {spread * 100:.2f}% <= {args.max_spread * 100:.1f}%)"
                   if spread <= args.max_spread else
                   f"NOISY (spread {spread * 100:.2f}% > {args.max_spread * 100:.1f}%)")
    if args.band:
        lo, hi = (float(x) for x in args.band.split(":"))
        verdict.append("IN BAND" if lo <= med <= hi else f"OUT OF BAND [{lo}, {hi}]")

    excursions = []
    for path in args.monitor:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                hot = max(float(row.get(f"tz{i}") or 0) for i in range(4)) > 60000
                cpu = float(row.get("top_cpu") or 0)
                comm = row.get("top_comm", "")
                expected = any(name and name in comm for name in args.expect.split(","))
                if hot or (cpu > 10 and not expected and "ps" not in comm
                           and "board-monitor" not in comm):
                    excursions.append(f"{path}:{row.get('ts')}:{comm}={cpu}")
    if excursions:
        verdict.append("INCOMPARABLE (" + "; ".join(excursions[:3]) + ")")

    print(f"{label}: n={len(values)} median={med:.4g} spread={spread * 100:.2f}% "
          f"values={', '.join(f'{v:.4g}' for v in values)}")
    print("  " + " | ".join(verdict))
    return 1 if any(v.startswith("INCOMPARABLE") for v in verdict) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
