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
* `board-coverage-census.sh` + `ort-provider-census.py` - the screening pair that answers "does this
  model hand work back to the CPU EP, and how much of the frame is that". The board script profiles a
  model list in the deployment shape; the host script parses the profiles next to their run logs and
  prints per-provider shares, per-operator cost and the EP-boundary nodes. It reads the CPU side as
  time and the provider's spans as spans only - the profile's provider spans sum to more than the
  frame, because a subgraph span covers work the provider parallelises internally.
* `board-session-options.sh` - the session-level option sweep (graph optimisation level, executor
  mode, memory arena and pattern, spinning, environment allocators) for one model, one variable at a
  time, in the deployment shape.
* `board-vlm-service-shape.sh` (+ `vlm-service-shape-client.py`, `vlm-decode-shape-client.py` in the
  evidence directory) - one VLM service-shape cell: N simultaneous image requests against one server,
  with a server lifecycle that cannot lie. It stops any previous server with TERM, waits, escalates to
  KILL, refuses to continue if one survives, refuses to start if the port still answers, prints the
  configuration banner into the cell record and asserts a single server process. Two servers survived
  their SIGTERM in round 7; had the port stayed busy, the next cell would have measured the previous
  configuration under a new name.
* `make-ime-probe.py` + `board-ime-probe.sh` + `ime-probe-report.py` - the AI-cluster ceiling
  instrument (`AI-CLUSTER-PEAK-20260922.md`). The host generator writes a pure-matrix **QDQ** probe
  (the form the provider's own shipped models use, because ONNX's `MatMul` does not admit `int8`),
  the board script runs the vendor's `perf_test` option set over three sizes and 1/2/4/8 threads with
  three rounds each, a CPU-EP control, a per-node profile and an in-flight TCM sample that also maps
  one block per compute thread, and the host tool turns the logs into the median table. Two probe
  forms that failed first (int8 `MatMul`; constant activation) are recorded in the evidence rather
  than deleted, and so is one sweep in which the 4096/1-thread row was bimodal.
* `board-multisession-scaling.sh` - the throughput-rule check (`MULTI-SESSION-SCALING-20260922.md`):
  `ort-two-session-pipeline`'s `same` mode at 1x8, 2x4, 4x2 and 8x1 on two detector models, three
  rounds per shape, and one longer engagement run per shape sampled with `spacemit-tcm-smi`. Two
  harness defects were found and fixed here (see the report): `setsid` may fork and make `$!` the
  wrong pid, and the TCM table's PID column may name a *thread* of the workload rather than the
  process, so the sampler matches any of the workload's thread ids.

The board scripts run on the board and the host pushes them and pulls the logs. Results of the
2026-09-21 rounds are in `../BASELINE*-20260921.md`, `../OPTIMISATION*-20260921.md` and
`../METHOD-ALIGNMENT-20260921.md` §7.
