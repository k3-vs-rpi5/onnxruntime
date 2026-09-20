# tools/

Host-side tooling for this project: the analysis and reporting helpers that turn board-side raw
output (provider traces, subgraph/tensor dumps, monitor packages) into evidence. The provider
profiling helper already exists in the monitor repository
(`hardware/k3-monitor/experiments/onnx-ep-profile.py`); this directory is where this project's own
scripts land as `PLAN.md` steps 2-4 are executed.
