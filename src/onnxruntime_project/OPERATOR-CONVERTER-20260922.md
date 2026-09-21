# An operator converter for provider capability: the case, the rules, and the first target

The question: can we build a converter that rewrites a graph so the execution provider claims all of
it? Measured evidence says yes, that it is worth a lot on specific models, and that the vendor's own
tooling does not do it.

## 1. The fallbacks are mostly *capability* failures, not missing operators

The provider documents **80+ accelerated operators** (`onnxruntime_ep_ops.md`), including `Conv`,
`ConvTranspose`, `Gemm`, `MatMul`, the whole QDQ family, `AveragePool`, `MaxPool`, `Resize`, `Cast`,
`Pow`, `Sqrt`, `Sigmoid`, `Gelu`, `Softmax`, `Concat`, `Split`, `Transpose`, `Gather`, `Slice`,
`Pad`, `Tile`, `LayerNormalization`, `Attention`, `RotaryEmbedding` - and its own fused
`YoloDecode` function operator.

Now the three fallback inventories this project has measured with its census instrument:

| model | fallback share | the operators that fall back |
|---|---|---|
| `yolo26n-depth.fp16` | **18.4 %** of a 67.62 ms frame | `Resize` 6.34 ms, `Cast` 2.14, `Pow` 2.00, `Exp` 0.11, `Mul` 0.07 |
| `pp_liteseg.q` | **25.5 %** of an 84.81 ms frame | `QuantizeLinear` x3, `AveragePool` x3, `Cast` |
| `yolo26n.q` | **8.2 %** of a 17.96 ms frame | decode head: `Tile`, `TopK`, `GatherElements`, `Div`, `Mod`, `Concat`, ... (16 executions) |

Every operator in the depth model's list except `Exp` is **in the provider's own supported list**.
So the provider is not refusing these operator *types*; it is refusing these *instances* - an
attribute combination, an input form, a dtype pair that its capability check does not accept. That is
exactly the class of problem a graph rewriter can fix, and it is why the same `Resize` and `Cast`
appear in an EP-complete model (`yolov8n-pose`, 0 % fallback) and a failing one (`yolo26n-depth`,
18.4 %): the operator is supported, the *shape of its use* is not.

## 2. The vendor's own simplifier does not do this - measured, not assumed

`onnxslim 0.1.87` (the graph simplifier shipped alongside the vendor's `xslim 2.1.1` quantiser) was run
on the depth model: op counts unchanged (`Transpose 4 -> 4`), file size unchanged (9.85 MB), and the
re-census of the simplified model reads **15 CPU node executions and 18.7 % - identical to the
original within run-to-run noise**. A generic constant-folding/fusion simplifier has no notion of the
provider's capability check, so it cannot close this gap. `xslim` itself is a precision tool
(`--fp16`, `--dynq`, `--opset`) and, as round 3 showed, its dynamic-quantisation path lands on the
provider's slow fp32 route.

## 3. The converter, as a design

**Input**: a model plus the census report for it (which nodes fell back, their names, durations and -
from the ONNX profile - their input shapes and dtypes).

**Knowledge base**: the documented operator list and per-operator constraints, encoded as rules -
`Conv` wants `kernel_shape` and a constant `W`; `Gemm` wants `transA=0`, `alpha=1`, `beta=1`; the QDQ
form wants `A` asymmetric per-tensor and `B` symmetric per-channel; and so on.

**Rule library**, ordered by the prize each one has already been measured to carry:

| # | rule | applies to | measured prize |
|---|---|---|---|
| 1 | `Resize`: try the attribute combinations the provider accepts (`sizes` vs `scales`, `coordinate_transformation_mode` in {half_pixel, pytorch_half_pixel, asymmetric}, `mode` in {nearest, linear}, `antialias=0`) | `yolo26n-depth` | **6.34 ms/frame** |
| 2 | `Cast`: retype `int64` chains down to `int32`, or eliminate the cast by re-typing its producer | `yolo26n-depth`, `pp_liteseg` | 2.14 + 0.65 ms |
| 3 | `Pow`: rewrite a non-constant or integer exponent into a constant-exponent form, or into a `Mul`/`Sqrt` chain (`x^2 -> x*x`) | `yolo26n-depth` | 2.00 ms |
| 4 | `Exp` (genuinely absent from the supported list) -> `Pow(e, x)` | `yolo26n-depth` | 0.11 ms |
| 5 | `AveragePool` with `count_include_pad`/`ceil_mode` the provider refuses -> the accepted form, or a depthwise `Conv` with a constant average kernel | `pp_liteseg` | part of 25.5 % |
| 6 | QDQ boundary: align `QuantizeLinear` scale/zero-point granularity and re-place the Q/DQ pair so the region becomes claimable | `pp_liteseg` | the rest of 25.5 % |
| 7 | decode head -> the provider's own `YoloDecode` function operator | `yolo26n` | 8.2 % |

**Verification protocol** - the same rules this project applies everywhere, in this order:
1. numeric gate first: `SPACEMIT_EP_DUMP_TENSORS` before and after the rewrite, compared tensor by
   tensor; rules 1-3 and 5-6 should be bit-identical or within a stated tolerance, and rule 4
   (`Exp`) is explicitly a tolerance-declared change, not an equivalence;
2. re-census: the fallback must actually disappear (that is the whole point, and one run proves it);
3. timing: the measured prize above is the expectation to confirm, with `-r 20` and the deployment
   shape (one session, EP 8 threads, ORT pool 8);
4. task-level check for anything tolerance-declared, since "fewer CPU nodes" is not a quality claim.

**Artefacts**: a host-side tool (`tools/ep-capability-converter.py`), a per-model conversion report,
and the converted models as new files with their own sha256 plus the exact rewrite script - the
original is never overwritten, and each conversion is reversible by construction.

## 4. First target, and what could go wrong

**First target: `yolo26n-depth.fp16`.** Its 18.4 % fallback is dominated by three operators the
provider claims elsewhere in other models (`Resize` 6.34 ms, `Cast` 2.14, `Pow` 2.00), so a
capability-driven rewrite has the best chance of a fast, large win: closing the whole fallback would
take the frame from 67.62 ms toward ~55 ms, and closing only `Resize` is worth 6.34 ms of it.

Honest risks:

* the provider's inner constraints are closed, so a rule is a *hypothesis* until a census confirms it;
  each probe costs one push + one run (~1 minute), which is cheap enough to iterate;
* some fallbacks are structural rather than cosmetic - `yolo26n`'s decode head is a subgraph, and
  replacing it with `YoloDecode` means authoring that replacement, whose input contract the
  documentation states only as "Attributes: None";
* a rewrite that changes numerics needs the tolerance gate and a task check, and a rewrite that
  silently changes *semantics* (e.g. a Resize coordinate mode) is worse than no rewrite at all - the
  numeric gate exists for exactly that.

## 5. Requirements for the conversion logic (operator-directed, 2026-09-22)

**Query first, then act.** No rule is written before its evidence exists, and evidence has three
grades, in this order:

1. **documented constraint** - `onnxruntime_ep_ops.md` states it (e.g. `Resize` supports only
   `asymmetric`/`half_pixel` and `nearest`/`linear`; `AveragePool` requires all pads zero when
   `count_include_pad != 1`; `Split`/`Slice`/`Tile`/`Pad`/`Reshape`/`Unsqueeze`/`Squeeze` require their
   secondary inputs to be constants; QDQ wants A asymmetric per-tensor and B symmetric per-channel);
2. **cross-model evidence** - the same operator appears in several shipped models, some claimed and
   some refused, and the difference identifies the rule. This is stronger than the documentation where
   the two disagree, and it is how the `Resize` case below was resolved;
3. **vendor example code** - how the vendor's own example prepares or feeds the model.

A probe is only ever used to **confirm** a rule derived from the above, never to fish for one. Every
rule records: its evidence, the graph it was applied to, the numeric-gate result, the re-census
result, and its status (`accepted` / `rejected` / `needs more query`).

### 5.1 Evidence registry: what the capability check actually accepts

| operator / form | documented | measured | status |
|---|---|---|---|
| `Resize` nearest + asymmetric (+ `nearest_mode=floor`) | supported | claimed in fp16 and fp32, in every model that uses it (32 nodes) | confirmed |
| `Resize` **linear + half_pixel**, inside a QDQ/int8 graph | supported | **claimed**: `pp_liteseg`'s 7 nodes are inside the provider's subgraph (its ORT profile contains no Resize node at all), and appear in the provider's own subgraph dump | confirmed |
| `Resize` **linear + half_pixel**, standalone fp32 or fp16 | supported | **refused**: a standalone probe is executed by the CPU EP, and the same node in an fp16 graph is refused (1 of 66 Resize nodes in this project's inventory) | **documentation and behaviour disagree; the rule is not writable yet** |
| `Resize` linear + `align_corners` | not supported | refused (2 nodes) | confirmed, needs a semantics-preserving replacement |
| `Exp` | absent from the supported list | refused (1 node, 0.11 ms) | confirmed |
| `Pow`, `Gather`, `Mul`, `Add` individually | supported | claimed in isolation (probe set: axes 2/3, int32/int64, negative axis, broadcast weights) | confirmed |
| `Gather`-based separable interpolation (the natural replacement for linear Resize) | - | **the provider segfaults** on the graph, at every size tried, while each primitive passes in isolation | **rejected - and a robustness finding: the provider accepts a graph it cannot execute** |

### 5.2 Rule status after this round

| rule | evidence | numeric gate | re-census | status |
|---|---|---|---|---|
| `resize-attrs` (drop the irrelevant `nearest_mode` from a linear Resize) | inventory: claimed instances carry the ONNX default, the refused one carries `floor` | **bit-identical** | fallback unchanged (15 CPU nodes) | **rejected** - the discriminating property is elsewhere |
| `resize-canonical` (`scales` -> `sizes`, drop `cubic_coeff_a`) | claimed node uses the `sizes` form with empty roi/scales | **bit-identical** | unchanged | **rejected** |
| `resize-fp32` (fp16 linear Resize through fp32 casts) | 21 claimed instances are fp32 | 0.50 fp16 ulp | unchanged, and one more CPU node | **rejected** |
| `resize` (linear Resize -> Gather/Mul/Add) | - | 0.24 fp16 ulp on the CPU EP | the provider **segfaults** | **rejected** |

The `Resize` family therefore stands at *needs more query*, and the next query is named: a
Q/D/Q-wrapped probe of the same `half_pixel + linear` Resize - if the QDQ route is what makes the
provider claim it, the rule becomes "route the linear resize through the quantised path", and its
price is measured (6.34 ms on this model) against a stated precision cost.

This is the discipline working as intended: four candidate rules were killed by evidence rather than
shipped as guesses, and the one that survives will arrive with its provenance attached.
