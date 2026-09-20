# ONNX Runtime project: technical analysis (planning pass, 2026-09-20)

## 1. What the pinned fork actually contains

`src/onnxruntime` is `spacemit-com/onnxruntime` at `d76541ad` (ref `upstream-1.24.2`), and
`sources.lock.yaml` digests the three files that distinguish it from upstream ONNX Runtime:

| file | what it is |
|---|---|
| `cmake/riscv64.spacemit.toolchain.cmake` | cross-toolchain file for the target |
| `tools/scripts/build_riscv64.spacemit.sh` | the fork's opinionated build recipe |
| `onnxruntime/test/util/test_spacemit_ep.cc` | a test utility that *uses* the provider |

The provider itself is **not** in this tree: `libspacemit_ep.so` has no source here and none is
published. Everything below therefore treats the provider as a sealed component reached through
documented interfaces, exactly as the workspace contract requires for vendor runtimes.

## 2. The sealed component and its identity

| artefact | identity |
|---|---|
| release package | `spacemit-ort.riscv64.2.0.6.tar.gz`, sha256 `bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6` (`sources.lock.yaml`); retrieved into `.local/k3-tools/riscv64/` this round |
| package commit | `5a5b59149c701931aeb32c36bc0181b4edf34dcc` |
| board packages | `spacemit-onnxruntime` 2.0.6 and `python3-spacemit-ort` 2.0.6 (deb, dated 2026-07-27) |
| installed runtime | `/usr/lib/libonnxruntime.so.1.24.2+spacemit.a1`, sha256 `076d9e8364dfa39f7b975f91b3c96eaa39634ce966d3abceeebce5703ce6639e` (20.4 MB) |
| installed provider | `/usr/lib/libspacemit_ep.so.2.0.6`, sha256 `d8fa3633536db78e7956312e985574ee9162d1363f9cef81426aba686a97d8af` (5.96 MB, **stripped**) |
| python package | `/usr/lib/python3.14/dist-packages/spacemit_ort` |

**The release tarball and the board's deb are different binaries under the same version number** -
measured this round by extracting the tarball's libraries and hashing them against the installed
ones:

| library (both claim 2.0.6, commit `5a5b5914`) | release tarball | board deb |
|---|---|---|
| `libonnxruntime.so.1.24.2+spacemit.a1` | `93bb75601d9eceb5aca192fa70c0c3e18b94a70b9f57acdc9b34c2ff426e09e3` | `076d9e83...` |
| `libspacemit_ep.so.2.0.6` | `dcc9503031bca22cf2b33a692f7b4c01d0fbb4a24c34f6e60c7faaddb78274ae` | `d8fa3633...` |

So three identities have to be named in every result - the fork commit, the release tarball and the
deb-installed pair the board actually loads - and none of them implies the others. The digests above
are recorded here first; `PLAN.md` step 1 moves the deb pair into `sources.lock.yaml` so a package
upgrade cannot pass unnoticed.

### 2.1 What the tarball adds beyond the libraries

The package is not only binaries: it carries the published extension surface this project is allowed
to use.

| path in the tarball | contents |
|---|---|
| `include/` (19 headers) | `onnxruntime_c_api.h` / `_cxx_api.h`, `provider_options.h`, `spacemit_ort_plugin.h`, `spine_llm_engine.h`, `spine_vision_engine.h`, `onnxruntime_lite_custom_op.h`, ... |
| `plugin/` (17 files + test models) | the plugin API's worked examples (`qgelu_op.cc`, `gelu_op.cc`, `layernorm_op.cc`, `plugin_main.cc`), `PLUGIN.md`, test ONNX models and their generator |
| `samples/` (28 files) | `spine_llm_engine.cc`, `spine_vision_engine.cc`, `multimodal_qa_demo.cc`, `imagenet_test.cc`, `concurrency_demo.cc`, `demo_custom_ort_op.cc`, `create_onnx_test_runner_cases.py`, ... |
| `bin/` | `onnxruntime_perf_test`, `onnx_test_runner`, `imagenet_test`, `run_demo` |
| `scripts/` | the vendor's own build recipes: `build_plugin.sh`, `build_samples_riscv64.sh`, `linux_riscv64.toolchain.cmake`, `run_plugin_demo.sh` |
| `python/` | the `spacemit_ort` wheel |

`PLUGIN.md` documents a **two-track plugin system**: `AddOperator` (a brand-new operator the plugin
fully owns: `Compile`, `ReCompile`, `KernelDispatch`) and `AddDispatch` (a dispatch-only overlay that
replaces the compute of an *existing* EP operator). A plugin is one `.so` built against
`spacemit_ort_plugin.h` with two exported entry points and an ABI version, loaded at runtime - no EP
recompilation. That is the one place where this project could change *compute* behaviour without
touching the sealed provider, and it is explicitly named as an allowed interface in the workspace
contract.

## 3. Documented interfaces (the only allowed window)

From `.local/docs-ai/{en,zh}/compute_stack/ai_compute_stack/onnxruntime.md` and `onnxruntime_ep_ops.md`:

* entry point `Ort::SessionOptionsSpaceMITEnvInit` (a provider-registration call that replaces the
  usual `SessionOptionsAppendExecutionProvider` path);
* thread shape: `SPACEMIT_EP_INTRA_THREAD_NUM` (with the documented note that `2 x intra_thread_num`
  compute threads are launched), `SPACEMIT_EP_INTRA_THREAD_AFFINITY` (semicolon list, count must
  match), `SPACEMIT_EP_INTER_THREAD_NUM`, `SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD`;
* observability: `SPACEMIT_EP_DEBUG_PROFILE` (Chrome trace), `SPACEMIT_EP_DUMP_SUBGRAPHS`,
  `SPACEMIT_EP_DUMP_TENSORS`;
* capability control: `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER`, `SPACEMIT_EP_DISABLE_OP_NAME_FILTER`,
  `SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE`, `SPACEMIT_EP_DENSE_ACCURACY_LEVEL`.

The accelerated set (per the op documentation) is dense `Conv`/`ConvTranspose`/`Gemm`/`MatMul`, the
QDQ family (`DynamicQuantizeMatMul`, `MatMulInteger`, `DynamicQuantizeLinear`, `QuantizeLinear`,
`DequantizeLinear`) and pooling. That shapes the workload choice: a Conv-heavy model (the smoke model
already on the board) exercises the dense paths, while a transformer ONNX would exercise
`MatMul`/`Gemm` and the QDQ paths - the same two families the LLM work lives in.

## 4. What this workspace has already measured

`hardware/k3-monitor/README.md` records the placement experiment (monitored packages, not hand
timings): **32.05 inferences/s with the CPU EP on the X100 against 81.95 with the provider on the AI
cores, identical output**. The tooling that produced it is already in the workspace:

* `experiments/onnx-ep-matrix.sh` - the provider matrix (cpu/provider, thread counts, affinity);
* `experiments/onnx-ep-profile.py` - ranks operators inside a `SPACEMIT_EP_DEBUG_PROFILE` trace
  (`cat=Node` operators, `cat=Session` subgraph executions, `cat=Api` session calls);
* `experiments/onnx-ep-analysis.py` - recomputes the monitor's verdict from its own timeline;
* the `board-onnxrt-copy` host dependency - the existing recipe for building a board-side binary
  against the board's shipped `libonnxruntime` + provider.

So the project does not start from zero: a first placement number and an operator-profiling tool
exist. What does not exist is a sealed, reproducible package: the matrix has been run by hand, the
trace tool has no recorded trace in an evidence tree, and no A100 engagement evidence has been
collected for the provider.

## 5. The measurement plan this analysis implies

1. **Baseline first.** CPU EP on the X100 with explicit affinity, same model, same thread count, as
   the reference the provider must beat - and the identical-output check that makes the comparison
   meaningful.
2. **Operator-level attribution through the provider's own trace.** `SPACEMIT_EP_DEBUG_PROFILE`
   gives per-operator and per-subgraph rows; `SPACEMIT_EP_DUMP_SUBGRAPHS` says which operators the
   provider claimed. The two together answer "what ran where" without touching the sealed library.
3. **A100 engagement, evidenced the contract's way.** For a provider run claimed to use the AI
   cluster: the provider's own registration of its threads (documented option and observed thread
   count), `spacemit-tcm-smi` occupancy naming the workload's pid while it runs, and the throughput
   arithmetic the model's shape implies. Per-CPU counters and per-process CPU time are not evidence
   here - measured on this board, the Linux-side instruments do not see the AI cluster at all.
4. **Numeric equivalence.** `SPACEMIT_EP_DUMP_TENSORS` against the CPU EP on the same inputs, so an
   operator-level or accuracy-level change is caught the way `capture-diff` catches kernel changes
   in the llama.cpp project. `SPACEMIT_EP_DENSE_ACCURACY_LEVEL` is the knob whose effect this
   measures.
5. **Package identity in every result.** A provider number without the deb/tarball digest is not
   reproducible, because the package is closed and can be upgraded independently of the fork.
6. **Plugin-based op work, when an operator is the question.** `AddDispatch` overlays the compute of
   an existing EP operator and `AddOperator` adds a new one; both leave the sealed EP in place. The
   package's own `plugin/` examples and `scripts/build_plugin.sh` are the starting point, and the
   `EP_DISABLE_OP_TYPE_FILTER` / `EP_DISABLE_OP_NAME_FILTER` options are how a built-in is taken out
   of the way when the overlay is meant to replace it.

## 6. Blind spots and risks

* **Stripped provider, no source.** No symbol-level profiling; only the documented dumps and the
  driver-side TCM view. Anything not covered by those stays unmeasured. The plugin API (2.1) is the
  single exception: it lets an operator's *compute* be replaced or added without touching the EP,
  so op-level work on this stack is possible - but only through that header and the ABI it checks.
* **Version drift.** The fork pins `upstream-1.24.2`; the board's packages are 2.0.6 (built from
  `5a5b5914`), and `.local/k3-tools/riscv64/ort-2.0.3` is an older package still lying around. Three
  identities, one board: every measurement names which one it used.
* **`2 x intra_thread_num` compute threads.** The documentation states this doubling; if it is what
  happens, the thread-count knobs do not mean what the names suggest, and any affinity experiment has
  to count threads rather than trust the option.
* **The two-cluster question is open.** The provider may use the AI cluster only, the X100 only, or
  both; the earlier placement result says the AI-cluster path is much faster, but a split
  (weights read by both clusters) is not documented as implemented anywhere.
* **Contract constraints carry over unchanged**: host-to-device delivery, no board-side fetching, no
  credential material in git, sealed local evidence.

## 7. Next step

Execute `PLAN.md` steps 1-3: extend the identity lock with the deb-installed digests, then write
`scripts/build.sh` and `scripts/run.sh` so the provider matrix, the profile trace and the
subgraph/tensor dumps run as one sealed, repeatable package instead of by hand.
