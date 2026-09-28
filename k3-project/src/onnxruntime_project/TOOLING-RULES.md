# Project tooling rules (operator-mandated, 2026-09-23)

**Rule 1 - no low-performance language for project tools.** Tools that run *on the board*, inside a
measurement window, or anywhere near the measured system are written in a compiled language with
predictable cost (C, C++, or Rust), not in Python or shell. Interpreters and shell pipelines have a
per-invocation and per-loop cost that is large compared with the quantities being measured, and a
monitor that perturbs the workload corrupts the very number it exists to protect.

**Rule 2 - Python and shell stay off-band.** They remain acceptable on the *host*, after the run, for
analysis over sealed raw output (metrics, digest registries, reports) - where their cost cannot enter
the measurement. A host-side tool that is run repeatedly during a board session should be bounded and
declared.

**Rule 3 - every on-board tool reports its own cost.** The monitor prints mean sample cost; the
harness prints per-stage milliseconds; the instrument that measures must be measurable.

## The evidence that produced the rule (all measured in this project)

| observation | number |
|---|---|
| `ps -eo pcpu,comm` inside the first monitor | appeared as the **top CPU consumer at 325 %** - the monitor was the interference |
| shell monitor sample cost | **100-120 ms per sample**, dominated by the `spacemit-tcm-smi` invocation, not by the file reads |
| Python decision layer (S10) | **~94 ms** for 100 scenes' worth of work per scene, including interpreter start-up, PIL decode and numpy passes - a C implementation removes the interpreter term |
| Python ONNX Runtime on this board | reaches only the CPU EP, so a tool written in Python cannot even exercise the thing being measured |
| contract already in force | every AI-cluster number in this project comes from a C++ harness or the vendor's own binary - for exactly this reason |

## Consequences already applied

* the on-board monitor was rewritten to **one** `spacemit-tcm-smi` call per sample, no `ps`, with
  interference judged from `/proc/pressure/cpu` (one file read) and per-sample cost printed;
* the next step named here: a **C sampler** (`tools/board-monitor.c`) that reads clocks, thermal
  zones, pressure and load directly at high frequency and calls the vendor `smi` only at a lower
  rate, so the common case costs microseconds rather than a hundred milliseconds;
* host-side analysis stays in Python by Rule 2, and every report states which side of the line a tool
  lives on.

## Status of the C sampler, measured (2026-09-23)
+
+| monitor | sample cost | note |
+|---|---|---|
+| shell version (fs reads + one `smi` + no `ps`) | **100-120 ms** | dominated by the `spacemit-tcm-smi` invocation |
+| C version `tools/board-monitor.c` (`3d6805c9…`), `fast-ms 500`, `smi-every 4` | **31-40 ms** | ~3x better, **not yet microseconds**; the `popen` of `smi` every 4th sample plus six sysfs reads still dominate |
+| **C, fast/slow split** (`4ec4823e…`) | **fast path 70 us mean / 108 us max**, slow path 39 ms once per 2 s | **target met**: the hardware-sensor and `smi` reads are amortised, the common sample is microseconds |
+| measured in the same window | PSI `some avg10` <= 0.2 %, thermal peak 34 C | the monitor no longer disturbs what it measures |
+
+Recorded rather than claimed: the rewrite is a 3x improvement and the remaining cost is named. The
+rule that matters is already in force - the monitor is C, it never shells out to `ps`, and it prints
+its own cost so the next pass is judged on a number.
