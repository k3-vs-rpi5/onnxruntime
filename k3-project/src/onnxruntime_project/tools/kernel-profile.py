#!/usr/bin/env python3
"""Aggregate the kernel function profiler across CPUs and rank it by real work.

    kernel-profile.py <profile-all-cpus.txt> [--exclude PATTERN ...] [--top N]

The kernel writes one `trace_stat/functionN` file per CPU, each with rows
`Function  Hit  Time  Avg  s^2` where Time is in microseconds and **includes time spent sleeping**.
Sleeping on this board means `schedule`, `do_idle`, `epoll_wait`, `futex_wait` and friends, which is
why a raw ranking is a ranking of idle time rather than of work. This tool sums each function over all
CPUs and reports both the raw view and a view with the sleep-dominated paths removed, so a kernel-time
question ("what does the kernel spend time on when it is *working*") has an answer.

This is a host-side, off-band tool (Rule 2 of TOOLING-RULES.md): it runs after the measurement, on
sealed output, where Python's cost cannot touch the measured system.
"""

import argparse
import re
import sys

SLEEP_PATHS = ("schedule", "do_idle", "epoll", "futex", "poll", "wait", "idle", "nohz",
               "cpu_startup_entry", "arch_cpu_idle", "psci", "suspend")

ROW = re.compile(r"^\s*(\S.*?)\s{2,}(\d+)\s+(\d+)\s+us\s+([0-9.]+)\s+us\s+([0-9.]+)\s+us\s*$")


def parse(path):
    hits, times = {}, {}
    cpu = None
    for line in open(path, errors="replace"):
        if line.startswith("=="):
            cpu = line.split()[-1]
            continue
        m = ROW.match(line)
        if not m:
            continue
        fn, h, t = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        hits[fn] = hits.get(fn, 0) + h
        times[fn] = times.get(fn, 0) + t
    return hits, times


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--exclude", nargs="*", default=list(SLEEP_PATHS))
    args = ap.parse_args(argv)

    hits, times = parse(args.path)
    if not times:
        print("no rows parsed - is this the profiler dump (function0..15 concatenated)?")
        return 2

    total = sum(times.values())
    print(f"functions={len(times)} total_kernel_time_s={total / 1e6:.1f} "
          f"(includes sleep; see the second table for the working view)")
    print()
    print(f"{'rank':>4} {'function':38s} {'hits':>9} {'total_s':>10} {'mean_us':>10}")
    for i, (fn, t) in enumerate(sorted(times.items(), key=lambda kv: -kv[1])[:args.top], 1):
        print(f"{i:>4} {fn[:38]:38s} {hits[fn]:>9} {t / 1e6:>10.1f} {t / hits[fn]:>10.1f}")

    work = {fn: t for fn, t in times.items()
            if not any(p in fn for p in args.exclude)}
    work_total = sum(work.values())
    print()
    print(f"working view (sleep-dominated paths removed): total_s={work_total / 1e6:.1f} "
          f"({work_total / max(total, 1) * 100:.1f} % of all kernel time)")
    print(f"{'rank':>4} {'function':38s} {'hits':>9} {'total_s':>10} {'mean_us':>10}")
    for i, (fn, t) in enumerate(sorted(work.items(), key=lambda kv: -kv[1])[:args.top], 1):
        print(f"{i:>4} {fn[:38]:38s} {hits[fn]:>9} {t / 1e6:>10.1f} {t / hits[fn]:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
