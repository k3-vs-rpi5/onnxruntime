# The AI cluster's own ceiling: measured on this board, not quoted

This is item **G1** of `NEXT-ROUND-ANALYSIS-20260922.md`, and it closes the last number the
optimisation analysis was borrowing: the denominator. Until today the roofline in
`YOLO-OPTIMISATION-20260921.md` §7 compared what YOLO extracts (912 GOPS on `yolov8n` at eight
threads) against the operator's **quoted ~15 TOPS** for the K3 A100 cluster and a **K1** `cpufp`
number clock-scaled from 2 GHz to 1.8 GHz. Both are other people's numbers about other silicon. This
round measures the ceiling here, through this provider, with an instrument - a pure-matrix QDQ probe
and nothing else.

Evidence: `/root/k3-evidence/onnxruntime/peak-20260922/` (230 files, per-file sha256 registry, self-check
0 mismatched / 0 missing / 0 unlisted). Instruments: `tools/make-ime-probe.py`,
`tools/board-ime-probe.sh`, `tools/ime-probe-report.py`.

## 1. The instrument, and the two forms that failed first

ONNX's `MatMul` does not admit `int8`, so a probe cannot be written as `MatMul(int8, int8)`: the
provider's int8 kernel is reached through the **QDQ** form its own shipped models use - a quantised
activation, a per-channel-symmetric quantised constant weight, both dequantised, a float `MatMul`,
then `QuantizeLinear` -> `DequantizeLinear`. That pattern was read out of `vit_b_16.q.onnx`'s PPQ
graphs rather than assumed. Two forms were tried and rejected, and both are kept in the evidence:

* the int8-`MatMul` form: rejected by ONNX Runtime's checker at load on every row, including the CPU
  EP. The host-side ONNX checker had accepted the file - the board is what refused it;
* the folded form (activation as a constant, no graph input): **segfaults on both EPs before any
  timing line**. The accepted form feeds the activation as a real graph input, which is also the
  deployment shape.

One route is closed with a measurement rather than a memory: `-C session.disable_cpu_ep_fallback|1`
cannot be used with this build's `perf_test`, because the tool adds the CPU EP itself and the session
is refused (`Conflicting session configuration`). The claim "the provider owns the graph" is carried
by the **per-node profile** instead, which is what this project has used since round 6.

## 2. The measurement

Three probes (1024^3, 2048^3, 4096^3 = 1.074 / 8.590 / 68.719 GMAC), the vendor's own
`onnxruntime_perf_test` with the published option set, three independent rounds per row; the table
below is the **median of the three** (run-to-run spreads are in `sweep4-three-sizes/ceiling-table.txt`,
0.3-4 % except where noted). `GOPS = 2 x MACs / time`.

| probe | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|
| 1024^3 | 2.990 ms / **718 GOPS** | 1.845 ms / 1164 GOPS | 1.200 ms / 1789 GOPS | 0.992 ms / **2164 GOPS** |
| 2048^3 | 19.249 ms / **893 GOPS** | 11.561 ms / 1486 GOPS | 7.966 ms / 2157 GOPS | 6.239 ms / **2754 GOPS** |
| 4096^3 | 155.376 ms / 885 GOPS | 169.157 ms / 813 GOPS | 106.688 ms / 1288 GOPS | 67.751 ms / **2029 GOPS** |
| CPU EP control (8 X100 cores) | - | - | - | 14.881 / 80.722 / 526.483 ms = **144 / 213 / 261 GOPS** |

Two of those columns are the same table doing the work they exist for: the CPU-EP control prices
what a silent fallback would look like (8-15x slower at the same shapes), and the per-width TCM
sample shows the provider hands one block to every compute thread it creates -
`intra=1 -> block 0`, `intra=2 -> 0 1`, `intra=4 -> 0 1 2 3`, `intra=8 -> 0 1 2 3 4 5 6 7` - with the
workload's own pid named in every line and 9 threads visible at `intra=8`. This is the contract's
AI-cluster evidence, not an inference from throughput.

## 3. What it says

1. **The measured runtime-path ceiling is 2754 GOPS (2.75 TOPS) at eight threads** - **18.4 %** of the
   quoted ~15 TOPS, and 75 % of the K1 `cpufp` figure clock-scaled to this board (3.68 TOPS). So the
   quoted peak is not merely unreached by YOLO; it is not reached by *anything this stack can express
   as a pure matrix graph*.
2. **One AI core reaches 893 GOPS** while holding exactly one TCM block - **1.94x** the clock-scaled
   K1 per-core figure (460 GOPS). The K1 number understates this part, which is why the denominator
   now comes from here.
3. **Scaling is 3.1x from one to eight threads** (2048^3: 893 -> 2754 GOPS; 3.0x on 1024^3), not 8x.
4. **Bigger is not better per FLOP.** The 4096^3 probe reaches 2029 GOPS at eight threads - 26 %
   *below* the 2048^3 probe - so the QDQ path is limited by the data movement and handling around the
   matrix units, not by matrix throughput. A ceiling quoted from a single shape would have hidden
   this; the instrument's own trend is what exposes it.
5. **The X100 cluster does not enter this conversation**: 144-261 GOPS on the same graphs.

For YOLO the arithmetic changes shape in a useful way. `yolov8n` end-to-end extracts **912 GOPS**
(8.7 GFLOP in 9.538 ms). Against the new numbers that is **33 % of what the provider itself manages
on a pure matrix graph**, and the pure matrix graph is at 18 % of the quoted platform peak. So the
gap has two named parts - the runtime path's own ~5.4x to the quoted peak, and the real graph's ~3x
to the runtime path - and neither is "the silicon is slower than the table".

## 4. Caveats, stated with the numbers

* the probe is the provider's QDQ dialect, so the rate **includes the quantise/dequantise data
  movement** around the `MatMul`. It is a *runtime-path* ceiling, not the bare `vmadot` ISA peak; the
  latter is what the vendor's `cpufp` measures and is not reachable through this stack. The correct
  reading of "2.75 TOPS" is therefore "the most this provider+cluster delivers on pure matrix work",
  which is exactly the denominator the model comparison needs;
* one sweep (`sweep3-three-sizes`) measured the 4096/1-thread row as **bimodal** (266.3 / 155.1 /
  268.2 ms, 42 % spread); the accepted sweep measured the same row at 155.31/155.38/155.38 ms (0.0 %
  spread) with identical models and flags. Both are kept in the evidence and the cause is unnamed;
* the probe is an **instrument**, not a model, and nothing about it is optimised, tuned or proposed
  for deployment.

## 5. What this changes in the plan

G1 is closed. The next items in `NEXT-ROUND-ANALYSIS-20260922.md` are unchanged and unblocked; the
roofline section of the YOLO report now cites a measured denominator, and any future claim about
"how far from the peak" work is should use this number rather than the quoted 15 TOPS. The remaining
question in the same family - whether the runtime's 5.4x to the quoted peak is data movement or
matrix-issue - cannot be answered through the documented interfaces, and is now a vendor question
with a number attached rather than an assumption.
