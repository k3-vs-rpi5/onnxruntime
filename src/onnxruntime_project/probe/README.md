# probe/

Sources for board-side probes and workloads that link against the board's shipped
`libonnxruntime.so` + `libspacemit_ep.so` (cross-built from the host with the workspace toolchain,
delivered by `scripts/run.sh`). They exist to answer questions the sealed provider does not answer
itself: thread counts, affinity behaviour, subgraph claims, and per-operator time from the
documented trace option. The existing `onnx_ep_bench` workload in `hardware/k3-monitor` is the
starting point.
