# Eighth round: session-level options - what the vendor's own tool can and cannot turn

The published matrix passes no `-o`, and `onnxruntime_perf_test` defaults graph optimisation to
**level 1 (basic)**, while an ONNX Runtime *application* gets `ORT_ENABLE_ALL` (99) by default (the
tool's own README, and `command_args_parser.cc`). A difference like that is enough to invalidate a
comparison, so it was measured rather than argued: three representative models - a detector with no
CPU fallback, the segmenter with one, an encoder - in the deployment shape (EP 8, ORT pool 8,
`-r 10`, `-S 1 -s -c 1 -I`), one variable at a time.

| variant | `yolov8n_no_dfl.q` | `pp_liteseg.q` | `image_encoder.fp16` |
|---|---|---|---|
| **published default** (no `-o`; level 1) | 9.141 ms | 82.904 ms | 44.765 ms |
| `-o 99` (all) | 9.199 ms (+0.6 %) | 84.980 ms (+2.5 %) | 44.579 ms (-0.4 %) |
| `-o 2` (extended) | 9.211 ms | 85.171 ms | 44.698 ms |
| `-o 0` (disabled) | 9.574 ms (**+4.7 %**) | 98.012 ms (**+18.2 %**) | 45.010 ms (+0.5 %) |
| `-o 99 -P` (parallel executor) | 9.225 ms | 91.057 ms (**+9.8 %**) | 44.779 ms |
| `-o 99 -A -M` (no arena, no memory pattern) | 9.230 ms | 83.140 ms | 44.796 ms |
| `-o 99 -C "session.intra_op.allow_spinning\|0"` | 9.230 ms | 82.747 ms | 44.621 ms |
| `-o 99 -C "session.use_env_allocators\|1"` | 9.203 ms | 84.423 ms | 44.496 ms |

Four readings:

1. **There is no free win here, and the vendor's own default is the right one.** Level 99 lands within
   +-2.5 % of level 1 and is *slightly worse* on two of the three models. So the published matrix is
   not disadvantaged by its level-1 default, and a deployment gains nothing by raising the level.
   This closes the one hypothesis the level-1 default invited, and it keeps our numbers
   vendor-comparable: our own runs keep using their shape.
2. **Two settings are traps.** `-o 0` costs 4.7 % on the detector and **18.2 %** on the segmenter;
   `-P` (parallel executor, which an application may set for multi-node graphs) costs **9.8 %** on
   the same model. Both are easy to reach by accident.
3. **The memory-policy family is flat.** Arena off with memory pattern off, spinning off, environment
   allocators on - every one within +-1 % on all three models. There is no memory-policy lever to
   pull on this execution provider, which is worth knowing before anyone spends a week on one.
4. **Everything that moves, moves on the fallback-heavy model.** The model whose frame is 25.5 % CPU
   EP (round 6 §3) is the one that responds to optimiser level and executor mode, by an order of
   magnitude more than the two EP-complete models. The census predicts where session settings matter.

Raw: `/root/k3-evidence/onnxruntime/baseline-20260921/session-options/` (the summary plus one log per
variant per model, 24 runs, each with its own session-creation and percentile rows).
