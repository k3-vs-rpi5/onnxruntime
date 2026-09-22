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

### 5.0 Rules of engagement for a complex transformation (sharpened 2026-09-22)

Operator conversion is not a one-shot edit; it is a chain of artefacts, and this project's first
attempts at it produced two mistakes of its own before producing any finding. The discipline that
came out of them:

1. **opset discipline.** A rewrite must emit the form the *emitted* operator requires for the model's
   declared opset, and the two do not move together: `ReduceSum` takes `axes` as an input from opset
   13, while `ReduceL2` kept it as an attribute until opset 18. Feeding `axes` as an input into an
   opset-17 model produced a graph that the provider crashed on; using the ReduceL2 threshold for the
   ReduceSum being emitted produced a graph the checker rejected outright. Both were caught by the
   checker and the numeric gate **before** any board run, which is where they belong;
2. **three claims, never one.** "The rewrite is semantically equivalent", "the provider claims it" and
   "it is faster" are separate facts with separate instruments (numeric gate, coverage census,
   unprofiled timing). A rule is a *result* only when all three are recorded, and a rule that is
   equivalent and claimed but slower is a **rejected rule with a named mechanism**, not a finding
   about the provider;
3. **reduce before blaming.** When the provider crashes, the next step is to reduce the trigger to
   primitives and to the smallest graph that still fails - and to check the artefact first. A crash
   that survives every probe is recorded as **blocked-unexplained** with its reproduction attached,
   never as "the provider is broken";
4. **anomalies are observations.** A 3.5x slowdown in the direction of the intended optimisation is
   explained by instrumenting the mechanism (thread-summed per-operator time), not by restating the
   goal.

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

## 6. The mechanism, found by probe: context, not the operator

The named query was a Q/D/Q-wrapped probe. It answered something better than expected, and the answer
came from a control rather than from the QDQ wrap itself:

| probe (192 -> 768, `half_pixel + linear` Resize) | provider attribution |
|---|---|
| Resize alone, `sizes` or `scales` form, fp32 or fp16 | **CPU EP, 100 %** |
| Resize alone at `pp_liteseg`'s own channel counts and sizes (C=128 1->16, C=19 64->128) | **CPU EP, 100 %** |
| Resize **after a Conv** | **claimed** - the whole graph is one provider subgraph, zero CPU nodes |
| Resize after a Conv, wrapped in QDQ | **claimed** as well |

So the discriminator is neither dtype, nor attributes, nor the input form, nor the QDQ wrap: it is
**whether the Resize sits in a fragment the provider is already claiming**. A standalone resize is
declined and lands on the CPU; the same operator next to a convolution is claimed and executed. That
explains every observation in this project at once - `pp_liteseg`'s seven linear resizes are claimed
because they live inside its large QDQ/conv regions, and the depth model's are refused because the
head is cut into fragments by an unsupported node.

### 6.1 The rule that follows, and its net effect

If the fragmentation is the cause, removing the unsupported node is the rule: `Exp` is absent from the
supported list while `Pow` is on it, and the depth head's chain is `Mul(Pow(Exp(clip(x)), a), c)`.
`Exp -> Pow(E, x)` was applied and verified:

| check | result |
|---|---|
| numeric gate | **0.03 fp16 ulp** - effectively exact |
| re-census | CPU node executions **15 -> 8**, CPU share **18.4 % -> 8.0 %**, CPU time **12.46 -> 6.14 ms**; the `Pow`/`Exp`/`Mul` chain left the CPU list entirely |
| unprofiled timing | **64.52 / 64.59 ms before against 74.93 / 74.83 ms after - 16 % slower** |

**Rule rejected on net effect.** The mechanism is confirmed exactly as predicted - the fragment merged
and seven CPU node executions disappeared - and the model still got slower, because the reconnected
provider region costs about 16 ms more than the mixed region it replaced. "Fewer CPU nodes" is not the
same claim as "faster", and this is the measurement that proves it. It also reframes the remaining
work: the question is no longer which operators to convert, but **why the provider's execution of that
head region is 16 ms slower than the CPU's for the same work** - a question for the vendor, with a
16 ms price tag and a reproducible probe (one Conv + one Resize) behind it.

## 7. Two rules against the models that motivated the tool, analysed rather than declared

The rationale for the converter was always the models with the largest fallbacks. Two of them have now
been rewritten, verified and measured, and neither is a one-line result - which is the point.

### 7.1 `yoloe-v8s-seg.dynq`: the rewrite worked, and the provider is 6.9x slower than the CPU

The census names the fallback precisely: three `Einsum` nodes at 44.12 / 12.06 / 2.70 ms, i.e.
**58.88 ms of a 157.55 ms frame (37.4 %)**, and `Einsum` is **not** on the supported-operator list.
Their equation is `bchw,bkc->bkhw` - a per-pixel contraction over channels - which lowers exactly to
`Reshape` + `MatMul` + `Reshape` (`out[b,k,p] = sum_c x1[b,c,p] * x2[b,k,c]`, so the weight side
multiplies on the left; the first version of this rule added a `Transpose` and the model would not
load, which the gate caught before any timing).

| check | result |
|---|---|
| numeric gate | **1.79e-07** on both outputs (float32 rounding) |
| re-census | CPU node executions **12 -> 7**, CPU share **37.9 % -> 0.1 %** (0.77 ms) - the three `Einsum` nodes left the CPU exactly as intended |
| unprofiled timing | **157.55 ms -> 544.35 ms** (3.5x slower) |
| mechanism (provider trace) | the three new `MatMul` nodes cost **~405 ms/frame** on the provider against **58.9 ms/frame** for the equivalent `Einsum` nodes on the CPU EP - **6.9x**, over 0.7 GFLOP of arithmetic, i.e. **1.7 GOPS effective** |

The rule is correct, it did what it was designed to do, and it is rejected on net effect - with the
interesting part named: **for this arithmetic the provider's path is far slower than the CPU EP's**,
which is the opposite of the assumption the conversion idea rests on, and it is a specific question
for the vendor (a `MatMul` probe at these shapes: `[1,80,512] x [1,512,6400]` and its siblings).

### 7.2 `superpoint_512x512_top512_batch1.q`: one unsupported operator, and a crash not yet explained

The census names one operator: `ReduceL2` at 0.434 ms inside a 2.30 ms fallback (13.8 % of a 15.99 ms
frame). `ReduceL2` is **not** on the supported list, and it lowers to `Sqrt(ReduceSum(x*x))` - three
supported operators, bit-identical on the CPU EP (`max|diff| = 0`) once the opset form was right.

What happened next is recorded as a sequence, because the sequence is the evidence:

| step | observation |
|---|---|
| opset-naive rule | provider **segfault** - and the cause was *ours*: `axes` fed as an input into an opset-17 model |
| opset-correct rule | numeric gate **0**, ONNX checker passes, CPU EP runs it - and the provider **still segfaults** |
| primitives | `Mul`, `ReduceSum` with axes as an input, `Sqrt` and their composition all pass on the provider |
| same shape, fp32 | passes (`[1,256,64,64]`, 4.07 ms) |
| same shape, wrapped in QDQ | passes (14.95 ms) |
| axis | reducing over axis 3 instead of axis 1 **also crashes** - the axis is not the trigger |
| when | `-v` shows *session initialisation succeeding*; the crash is in the **first inference** |
| subgraph dump | nothing is written before the crash |

So the rule is correct, the provider accepts a graph it cannot execute, and every reduction a black box
can perform - primitives, shape, dtype context, axis - passes beside it. The rule is therefore
**blocked-unexplained**, not rejected: the trigger is in the interaction between the rewritten node
and this graph, and naming it needs either the provider's internals or a longer reduction than this
pass could afford. The reproduction travels to the vendor with the probes that pass beside it, which
is what makes it actionable rather than a complaint.

### 7.3 `mobilesam_decoder_sim.dynq`: the hypothesis was disproved by the query, before any rule

This model has the most fallback in the catalogue after `yoloe` - **20.3 %** of a 28.98 ms frame, with
**270 CPU node executions per frame** and a largest single node of only 0.32 ms. That signature says
"fragmentation", and the unsupported operators named by the query are `Shape` (36), `Expand` (2),
`Not` (1) and `OneHot` (1). The obvious hypothesis was that the `Shape` nodes are removable plumbing -
a shape that is statically known can be replaced by a constant initializer, which is exactly the kind
of input the provider requires for `Reshape`/`Slice`/`Expand`.

**The query killed it, with a number.** Counting every `Shape` node against the shapes that shape
inference can actually prove:

| `Shape` inputs | count |
|---|---|
| fully static dimensions | **0** |
| partially dynamic | 14 |
| no or empty shape | 22 |

and the reason is structural, not a tooling gap: the model's inputs include
`point_coords [1, num_points, 2]` and `point_labels [1, num_points]`. It is a **promptable** decoder,
so the dimension most of that plumbing computes is a genuine runtime variable - one that a service
changes per request. A shape-folding rule therefore has **zero applicable nodes** here, and the 8.59 ms
of fallback is not plumbing that can be simplified away; it is dynamic-shape handling the provider does
not claim, in a graph the vendor ships for exactly that purpose.

Two conclusions, both recorded rather than carried:

1. for this candidate the only exact rules left are tiny (`Not(x)` -> `Equal(x, false)` - `Not` is not
   on the supported list while `Equal` is - and `OneHot` -> `Gather` from a constant table). Neither
   touches the 8.59 ms, because that time is spread over 270 nodes: **there is no rule to write**, and
   a rule that reroutes the region through the provider is exactly the trade section 7.1 measured as a
   loss;
2. the addressable version of this candidate is a different artefact - a **static-prompt** decoder,
   i.e. a model change - which is out of the current contract and is recorded here as the number (20.3 %
   of the frame) rather than as a pending optimisation.

That is the third outcome the discipline allows, alongside *rejected with mechanism* (7.1) and
*blocked-unexplained* (7.2): a hypothesis **disproved by query**, before any artefact was built.
