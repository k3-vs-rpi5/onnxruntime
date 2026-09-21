# tools/

Host-side tooling for this project: the analysis and reporting helpers that turn board-side raw
output (provider traces, subgraph/tensor dumps, monitor packages) into evidence. The provider
profiling helper already exists in the monitor repository
(`hardware/k3-monitor/experiments/onnx-ep-profile.py`); this directory is where this project's own
scripts land as `PLAN.md` steps 2-4 are executed.

The first two landed scripts are the baseline harness:

* `board-matrix.sh` - the provider/CPU-EP latency matrix with the vendor's own
  `onnxruntime_perf_test` and its published option set, plus the subgraph dumps and profile traces.
* `board-evidence.sh` - one long provider run sampled for the contract-grade AI-cluster evidence:
  `spacemit-tcm-smi` occupancy naming the workload's pid, the process's thread set, and per-thread
  affinity.
* `board-session-shape.sh` - the round-6 control pair for a graph whose EP claims only part of its
  nodes: ONNX Runtime's own intra-op pool at one thread against eight, with the EP's width pinned in
  both, a TCM sample taken inside every run, and both the provider's trace and ORT's per-node
  profile attached. The TCM sample is the point: it is what shows whether two runs being compared
  were the same AI configuration, and the first pair run without it was not.

All three run on the board; the host pushes them and pulls the logs. Results of the 2026-09-21
rounds are in `../BASELINE*-20260921.md` and `../OPTIMISATION*-20260921.md`.
