# ONNX Runtime project: plan

Scope: measure and evidence what the SpacemiT ONNX execution provider (`libspacemit_ep.so`) does on
this board, the way the llama.cpp project measures its vendor backend - host-side tooling around a
sealed vendor component, no modification of the component.

This round (2026-09-20) is **planning only**: the pinned fork is checked out, the sealed release
package is retrieved and identified, the directory layout exists, and
`TECHNICAL-ANALYSIS.md` records what is known and what is missing, and `DEEP-ANALYSIS.md` turns the
llama.cpp campaign's measured AI-cluster rules, the vendor's K3 tables and two real provider traces
into a ranked optimisation agenda with VLM and embodied-AI workloads in front. No source file in
`src/onnxruntime` has been touched, and no provider behaviour has been changed.

## What is in place

| item | state |
|---|---|
| pinned fork `src/onnxruntime` | `d76541ad26ca93d2d52e5d2db2aacaff862df5cf` (ref `upstream-1.24.2`), clean |
| sealed release package | `spacemit-ort.riscv64.2.0.6.tar.gz` under `.local/k3-tools/riscv64/`, sha256 checked against `sources.lock.yaml` |
| board-side provider | deb packages `spacemit-onnxruntime` / `python3-spacemit-ort` 2.0.6, library identities recorded in the analysis |
| vendor documentation | `.local/docs-ai/{en,zh}/compute_stack/ai_compute_stack/onnxruntime*.md` |
| probe tooling already in the workspace | `hardware/k3-monitor/experiments/onnx-ep-{matrix.sh,profile.py,analysis.py}` and the `onnx_ep_bench` workload |
| package contents beyond the libraries | `include/` (19 headers incl. `spacemit_ort_plugin.h`), `plugin/` examples + test models, `samples/` (LLM/vision engines, imagenet, concurrency), `bin/` (`onnxruntime_perf_test`, `onnx_test_runner`), `scripts/build_plugin.sh` - see the analysis |

## Deliverables (in order, each independently verifiable)

1. **Identity lock, extended.** `sources.lock.yaml` gains the deb-installed library digests for the
   board's provider, so a package upgrade cannot pass unnoticed the way a vendor force-push cannot
   for the fork.
2. **`scripts/build.sh`.** Cross-builds the project's own workload binary and probe against the
   board's shipped `libonnxruntime.so` + `libspacemit_ep.so` (the `board-onnxrt-copy` dependency in
   `hardware/k3-monitor` already does this by hand), sealed with `SHA256SUMS` and verified with the
   monitor's `payload_verify.py`.
3. **`scripts/run.sh`.** Host-to-device delivery and the measurement matrix: CPU EP baseline on the
   X100 cluster, provider on the AI cluster, identical-output check, `SPACEMIT_EP_DEBUG_PROFILE`
   trace, `SPACEMIT_EP_DUMP_SUBGRAPHS` / `SPACEMIT_EP_DUMP_TENSORS` captures.
4. **A100 engagement evidence.** Per the workspace contract, provider work on the AI cluster is
   claimed only with the runtime's own registration plus `spacemit-tcm-smi` occupancy naming the
   workload's pid, read together with the throughput arithmetic the workload's shape implies - never
   with per-CPU counters or per-process CPU time.
5. **Sealed evidence package and a report entry**, with every raw artefact named by its digest.
6. **Plugin track (optional, after 1-5).** The published `spacemit_ort_plugin.h` path
   (`AddOperator` / `AddDispatch`) is the only allowed way to change operator compute on this stack;
   it becomes interesting once the profile trace shows an operator worth replacing.

`DEEP-ANALYSIS.md` proposes D1-D8 in priority order and is the reference for what to run first:
D1 (EP thread/TCM/session census) and D2 (subgraph coverage audit) are the two that decide whether
the VLM encoder path is structurally sound before any tuning.

## Acceptance for step 1 (this round)

`PLAN.md` and `TECHNICAL-ANALYSIS.md` exist under `src/onnxruntime_project/`, the package digest
matches `sources.lock.yaml`, the fork checkout is clean at the locked commit, and nothing under
`src/onnxruntime` differs from the pin.

## Out of scope

- The provider's internals: it ships stripped and its source is not published. Work uses documented
  interfaces only (`Ort::SessionOptionsSpaceMITEnvInit`, the provider options, the dump/profile
  switches).
- Changing the model, its quantisation or the board's packages.
- Claiming any number without the run's own counters and a physics check, as everywhere in this
  workspace.

## Open questions for a human

1. Which workload is the acceptance target - the smoke model already on the board
   (`squeezenet1.1-7.onnx`, Conv-heavy) plus one transformer, or a specific customer model?
2. Is the deliverable the provider matrix as it stands (comparison and profile), or should the
   project also build the fork's own `onnxruntime_perf_test` / `onnx_test_runner` (the demo path in
   the vendor documentation) and measure with those?
3. Should the project aim at the two-cluster question the llama.cpp campaign closed (X100 + A100
   split), or stay with "which cluster does the provider use, and how well"?
