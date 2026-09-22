#!/usr/bin/env python3
"""Rewrite an ONNX graph so the execution provider can claim the nodes it currently refuses.

The provider documents 80+ accelerated operators, yet several models fall back to the CPU EP
(`yolo26n-depth` 18.4 % of its frame, `pp_liteseg` 25.5 %, `yolo26n` 8.2 %). Of the depth model's
fallback operators, every one but `Exp` is *on* that supported list - so what the provider refuses is
the shape of the use, not the operator. This tool applies capability-driven rewrites to close that
gap, one rule at a time, and never in the dark: every rule is proposed, applied, and then verified by
the project's own instruments.

Usage:

    # what does the fallback actually look like? (no rewrite)
    ep-capability-converter.py inspect <model.onnx>

    # apply one or more rules, writing a new file
    ep-capability-converter.py convert <model.onnx> <out.onnx> --rules resize,cast,pow

The verification that follows a conversion is *not* this tool's job: the project's protocol is
numeric gate first (`SPACEMIT_EP_DUMP_TENSORS` comparison), then a coverage census to confirm the
fallback disappeared, then timing. A conversion that has not been through that is a hypothesis.
"""

import argparse
import collections
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DEFAULT_RULES = ("resize", "cast", "pow")


def tensor_type_name(elem_type: int) -> str:
    return TensorProto.DataType.Name(elem_type) if elem_type else "?"


def graph_index(model):
    """Producers, consumers and declared types, so a rule can reason about context."""
    producers = {}
    consumers = collections.defaultdict(list)
    for node in model.graph.node:
        for out in node.output:
            producers[out] = node
        for inp in node.input:
            consumers[inp].append(node)
    types = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = [d.dim_value or d.dim_param or "?" for d in value.type.tensor_type.shape.dim]
        types[value.name] = (tensor_type_name(value.type.tensor_type.elem_type), dims)
    for initializer in model.graph.initializer:
        types[initializer.name] = (tensor_type_name(initializer.data_type), list(initializer.dims))
    return producers, consumers, types


def describe(node, types):
    attrs = ", ".join(
        f"{a.name}={helper.get_attribute_value(a)!r}" for a in node.attribute
    ) or "-"
    ins = ", ".join(f"{i}{types.get(i, ('?', []))[0]}{types.get(i, ('?', []))[1]}" if i else "-"
                    for i in node.input)
    outs = ", ".join(f"{o}{types.get(o, ('?', []))[0]}{types.get(o, ('?', []))[1]}" if o else "-"
                     for o in node.output)
    return f"{node.op_type:16s} attrs[{attrs}]  in[{ins}]  ->  out[{outs}]"


def inspect(model_path, ops=("Resize", "Cast", "Pow", "Exp", "Mul", "AveragePool", "QuantizeLinear")):
    model = onnx.load(model_path, load_external_data=False)
    for name, array in _initializers(model):
        pass
    producers, consumers, types = graph_index(model)
    print(f"== {model_path}: {len(model.graph.node)} nodes, opset "
          f"{[f'{o.domain or 'ai.onnx'}:{o.version}' for o in model.opset_import]}")
    counts = collections.Counter(n.op_type for n in model.graph.node)
    print("   op inventory:", ", ".join(f"{k}x{v}" for k, v in counts.most_common(12)))
    for op in ops:
        nodes = [n for n in model.graph.node if n.op_type == op]
        if not nodes:
            continue
        print(f"\n   -- {op} ({len(nodes)} node(s))")
        for node in nodes:
            print(f"      {node.name or '(unnamed)':28s} {describe(node, types)}")
            for inp in node.input:
                if inp in producers:
                    print(f"         produced by: {describe(producers[inp], types)}")
            for out in node.output:
                for consumer in consumers.get(out, [])[:3]:
                    print(f"         consumed by: {consumer.op_type} ({consumer.name or 'unnamed'})")
    return 0


def _initializers(model):
    for initializer in model.graph.initializer:
        yield initializer.name, numpy_helper.to_array(initializer)


def _initializer_map(model):
    return {name: numpy_helper.to_array(init) for name, init in
            ((i.name, i) for i in model.graph.initializer)}


def _resize_target_sizes(node, initializers, in_h, in_w):
    """Return the output H,W a Resize asks for, from its constant `scales` or `sizes` input.

    These exports use `scales` in input slot 2 (the ONNX Resize signature puts `scales` before
    `sizes`), which is why the first version of this rule matched nothing - a reminder that a rule
    must be derived from the graph it is meant to rewrite.
    """
    scales = initializers.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
    if scales is not None and scales.size == 4:
        return int(np.floor(in_h * float(scales[2]))), int(np.floor(in_w * float(scales[3])))
    sizes = initializers.get(node.input[3]) if len(node.input) > 3 and node.input[3] else None
    if sizes is not None and sizes.size == 4:
        return int(sizes[2]), int(sizes[3])
    return None


def _axis_weights(in_size, out_size, mode, dtype):
    """Exact linear-interpolation indices and weights for one axis.

    `half_pixel` (PyTorch's align_corners=False) and `align_corners` both reduce to "gather two
    neighbours and blend", so the rewrite is exact up to floating-point rounding order rather than an
    approximation of the resampling maths.
    """
    positions = (np.arange(out_size, dtype=np.float64) + 0.5) / (out_size / in_size) - 0.5
    if mode == "align_corners":
        positions = np.arange(out_size, dtype=np.float64) * (in_size - 1) / (out_size - 1)
    positions = np.clip(positions, 0.0, in_size - 1)
    index0 = np.floor(positions).astype(np.int32)
    index1 = np.minimum(index0 + 1, in_size - 1).astype(np.int32)
    weight1 = (positions - index0).astype(dtype)
    weight0 = (1.0 - (positions - index0)).astype(dtype)
    return index0, index1, weight0, weight1


def rule_resize_linear(model, tag="capconv", only=None):
    """Rewrite linear `Resize` into separable `Gather` + weighted sums.

    The provider claims nearest-neighbour `Resize` (measured: the two `asymmetric/nearest` nodes in
    `yolo26n-depth` are on its side of the partition) but hands back every linear one. `Gather`, `Mul`
    and `Add` are all on its supported list, and the indices and weights are constants, so this
    replaces an unsupported *use* with supported ops and no approximation of the resampling maths.
    """
    initializers = _initializer_map(model)
    rewritten = []
    for node in list(model.graph.node):
        if node.op_type != "Resize":
            continue
        if only and only not in (node.name or ""):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("mode", b"nearest") not in (b"linear", "linear"):
            continue
        coordinate = attrs.get("coordinate_transformation_mode", b"half_pixel")
        coordinate = coordinate.decode() if isinstance(coordinate, bytes) else coordinate
        if coordinate not in ("half_pixel", "align_corners"):
            continue
        # Input shape must be known in the graph's declared types.
        producers, consumers, types = graph_index(model)
        in_type, in_dims = types.get(node.input[0], ("?", []))
        if len(in_dims) != 4 or not all(isinstance(d, int) for d in in_dims):
            continue
        _, _, in_h, in_w = in_dims
        sizes = _resize_target_sizes(node, initializers, in_h, in_w)
        if sizes is None:
            continue
        out_h, out_w = sizes
        # Blend in the graph's own dtype. Measured on the depth model: blending in fp32 and casting
        # back *doubles* the disagreement with the original resampler (4.9e-4 against 2.3e-4), because
        # ONNX Runtime's fp16 Resize also upsamples in fp16 - so matching the reference means matching
        # its precision, not exceeding it.
        use_fp16 = in_type == "FLOAT16"
        np_dtype = np.float16 if use_fp16 else np.float32
        work_type = TensorProto.FLOAT16 if use_fp16 else TensorProto.FLOAT
        name = node.name or node.output[0]
        base = f"{tag}_{name.strip('/').replace('/', '_')}"
        idx0h, idx1h, w0h, w1h = _axis_weights(in_h, out_h, coordinate, np_dtype)
        idx0w, idx1w, w0w, w1w = _axis_weights(in_w, out_w, coordinate, np_dtype)
        new_nodes = []
        new_inits = []
        for suffix, array, dims in (
            ("idx0h", idx0h, [out_h]), ("idx1h", idx1h, [out_h]),
            ("idx0w", idx0w, [out_w]), ("idx1w", idx1w, [out_w]),
        ):
            new_inits.append(numpy_helper.from_array(array, f"{base}_{suffix}"))
        for suffix, array in (
            ("w0h", w0h.reshape(1, 1, out_h, 1)), ("w1h", w1h.reshape(1, 1, out_h, 1)),
        ):
            new_inits.append(numpy_helper.from_array(array, f"{base}_{suffix}"))
        for suffix, array in (
            ("w0w", w0w.reshape(1, 1, 1, out_w)), ("w1w", w1w.reshape(1, 1, 1, out_w)),
        ):
            new_inits.append(numpy_helper.from_array(array, f"{base}_{suffix}"))
        new_nodes += [
            helper.make_node("Gather", [node.input[0], f"{base}_idx0h"], [f"{base}_g0h"], axis=2,
                             name=f"{base}/gather_h0"),
            helper.make_node("Gather", [node.input[0], f"{base}_idx1h"], [f"{base}_g1h"], axis=2,
                             name=f"{base}/gather_h1"),
            helper.make_node("Mul", [f"{base}_g0h", f"{base}_w0h"], [f"{base}_h0w"],
                             name=f"{base}/mul_h0"),
            helper.make_node("Mul", [f"{base}_g1h", f"{base}_w1h"], [f"{base}_h1w"],
                             name=f"{base}/mul_h1"),
            helper.make_node("Add", [f"{base}_h0w", f"{base}_h1w"], [f"{base}_h"],
                             name=f"{base}/add_h"),
            helper.make_node("Gather", [f"{base}_h", f"{base}_idx0w"], [f"{base}_g0w"], axis=3,
                             name=f"{base}/gather_w0"),
            helper.make_node("Gather", [f"{base}_h", f"{base}_idx1w"], [f"{base}_g1w"], axis=3,
                             name=f"{base}/gather_w1"),
            helper.make_node("Mul", [f"{base}_g0w", f"{base}_w0w"], [f"{base}_w0out"],
                             name=f"{base}/mul_w0"),
            helper.make_node("Mul", [f"{base}_g1w", f"{base}_w1w"], [f"{base}_w1out"],
                             name=f"{base}/mul_w1"),
            helper.make_node("Add", [f"{base}_w0out", f"{base}_w1out"], [node.output[0]],
                             name=f"{base}/add_w"),
        ]
        index = list(model.graph.node).index(node)
        del model.graph.node[index]
        for offset, new_node in enumerate(new_nodes):
            model.graph.node.insert(index + offset, new_node)
        model.graph.initializer.extend(new_inits)
        rewritten.append(f"{name} [{in_h}x{in_w}->{out_h}x{out_w}, {coordinate}] -> "
                         f"{len(new_nodes)} supported ops")
    return rewritten


def rule_resize_nearest_mode(model, tag="capconv", only=None):
    """Drop a non-default `nearest_mode` from `linear` Resize nodes.

    Evidence, from the project's own inventory rather than from a probe:

    * the provider documents `Resize` as supporting `coordinate_transformation_mode` in
      {asymmetric, half_pixel} and `mode` in {nearest, linear};
    * seven `half_pixel + linear` Resize nodes in `pp_liteseg` carry ONNX's default
      `nearest_mode=round_prefer_floor` and are all claimed;
    * the depth model's `half_pixel + linear` node carries `nearest_mode=floor` and is not.

    `nearest_mode` only affects `mode=nearest`, so removing it from a `linear` node cannot change the
    resampled values - it is a capability-check normalisation, not a numerical shortcut, and it is
    verified as bit-identical before its speed is quoted.
    """
    rewritten = []
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        if only and only not in (node.name or ""):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        mode = attrs.get("mode", b"nearest")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        if mode != "linear":
            continue
        nearest_mode = attrs.get("nearest_mode")
        if nearest_mode in (None, b"round_prefer_floor", "round_prefer_floor"):
            continue
        kept = [a for a in node.attribute if a.name != "nearest_mode"]
        del node.attribute[:]
        node.attribute.extend(kept)
        rewritten.append(f"{node.name or node.output[0]}: dropped nearest_mode="
                         f"{(nearest_mode.decode() if isinstance(nearest_mode, bytes) else nearest_mode)}"
                         f" from a linear Resize (no effect on linear resampling)")
    return rewritten


def rule_resize_canonicalise(model, tag="capconv", only=None):
    """Make a `Resize` node look like one the provider already claims.

    Derived by comparing a claimed instance against a refused one, not by guessing:

    | | `pp_liteseg` Resize.0 (**claimed**) | `yolo26n-depth` /model.23/Resize_2 (**refused**) |
    |---|---|---|
    | attributes | `coordinate_transformation_mode=half_pixel, mode=linear` | those plus `cubic_coeff_a`, `nearest_mode` |
    | inputs | empty `roi`, empty `scales`, `sizes` as int64 | empty `roi`, `scales` as float |

    Both forms mean the same thing - ONNX defines the `scales` form's output shape as
    `floor(input_shape * scales)` - so this normalisation cannot change the resampled values, and it is
    verified bit-identical before its speed is quoted. `align_corners` nodes are deliberately left
    alone: the provider documents only `asymmetric` and `half_pixel`, so those need a different
    treatment and a different proof.
    """
    initializers = _initializer_map(model)
    rewritten = []
    for node in list(model.graph.node):
        if node.op_type != "Resize":
            continue
        if only and only not in (node.name or ""):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        mode = attrs.get("mode", b"nearest")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        coordinate = attrs.get("coordinate_transformation_mode", b"half_pixel")
        coordinate = coordinate.decode() if isinstance(coordinate, bytes) else coordinate
        if coordinate == "align_corners":
            rewritten.append(f"{node.name or node.output[0]}: SKIPPED (align_corners is documented "
                             f"as unsupported; needs a different treatment)")
            continue
        scales = initializers.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        if scales is None or scales.size != 4:
            continue
        input_shape = graph_index(model)[2].get(node.input[0], ("?", []))[1]
        if len(input_shape) != 4 or not all(isinstance(d, int) for d in input_shape):
            continue
        sizes = [int(np.floor(dim * float(scale))) if index >= 2 else int(dim)
                 for index, (dim, scale) in enumerate(zip(input_shape, scales))]
        base = f"{tag}_{(node.name or node.output[0]).strip('/').replace('/', '_')}"
        shape = numpy_helper.from_array(np.array(sizes, dtype=np.int64), f"{base}_sizes")
        empty_roi = numpy_helper.from_array(np.array([], dtype=np.float32), f"{base}_roi")
        empty_scales = numpy_helper.from_array(np.array([], dtype=np.float32), f"{base}_scales")
        model.graph.initializer.extend([shape, empty_roi, empty_scales])
        kept = [a for a in node.attribute if a.name not in ("cubic_coeff_a", "nearest_mode")]
        del node.attribute[:]
        node.attribute.extend(kept)
        # Mirror the accepted node exactly: empty roi, empty scales, constant int64 sizes.
        x_name = node.input[0]
        del node.input[:]
        node.input.extend([x_name, f"{base}_roi", f"{base}_scales", f"{base}_sizes"])
        rewritten.append(f"{node.name or node.output[0]}: scales{list(scales)} -> sizes{sizes}, "
                         f"dropped cubic_coeff_a/nearest_mode (mode={mode})")
    return rewritten


def rule_resize_fp16_through_fp32(model, tag="capconv", only=None):
    """Run an fp16 `linear` Resize in fp32, cast in and out.

    This rule is derived from a complete inventory of every `Resize` node in every model this project
    holds - 66 of them across the zoo, the vendor's model-zoo repositories, the downloaded candidates
    and the provider's own dumped subgraphs:

    | mode | coordinate | dtype | count | claimed |
    |---|---|---|---|---|
    | nearest | asymmetric | float | 32 | yes |
    | **linear** | **half_pixel** | **float** | **21** | **yes - including nodes inside the provider's own subgraph dump** |
    | nearest | asymmetric | float16 | 10 | yes |
    | linear | align_corners | float16 | 2 | no (documented as unsupported) |
    | **linear** | **half_pixel** | **float16** | **1** | **no - the depth model's node** |

    So the discriminator is the **dtype**, not the attributes or the scales/sizes form: attribute
    normalisation and a `scales`-to-`sizes` rewrite were both tried and both left the fallback
    untouched, and a Gather-based reimplementation crashed the provider outright. Wrapping the node in
    casts is the rewrite the evidence actually supports: both `Cast` and `Resize` are supported ops,
    and the resampling then happens in the precision the provider demonstrably handles.
    """
    initializers = _initializer_map(model)
    types = graph_index(model)[2]
    rewritten = []
    for node in list(model.graph.node):
        if node.op_type != "Resize":
            continue
        if only and only not in (node.name or ""):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        mode = attrs.get("mode", b"nearest")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        coordinate = attrs.get("coordinate_transformation_mode", b"half_pixel")
        coordinate = coordinate.decode() if isinstance(coordinate, bytes) else coordinate
        if mode != "linear" or coordinate != "half_pixel":
            continue
        in_type = types.get(node.input[0], ("?",))[0]
        if in_type != "FLOAT16":
            continue
        base = f"{tag}_{(node.name or node.output[0]).strip('/').replace('/', '_')}"
        x_in, x_out = f"{base}_x32", f"{base}_y32"
        index = list(model.graph.node).index(node)
        model.graph.node.insert(index, helper.make_node(
            "Cast", [node.input[0]], [x_in], to=TensorProto.FLOAT, name=f"{base}/cast_in"))
        model.graph.node.insert(index + 2, helper.make_node(
            "Cast", [x_out], [node.output[0]], to=TensorProto.FLOAT16, name=f"{base}/cast_out"))
        node.input[0] = x_in
        node.output[0] = x_out
        # The output declaration of the original graph still names node.output[0]; the outer cast
        # produces it, so nothing else needs to change.
        rewritten.append(f"{node.name or ''}: fp16 half_pixel+linear Resize -> fp32 Resize "
                         f"(cast in/out), the form 21 claimed instances use")
    return rewritten


def rule_exp_to_pow(model, tag="capconv", only=None):
    """Replace `Exp` with `Pow(e, x)`, to reconnect a fragmented claimed region.

    Evidence: `Exp` is **absent** from the provider's supported-operator list while `Pow` is on it
    with no stated constraints; and a probe built to isolate the resize question showed that a
    `half_pixel + linear` Resize is claimed **when it sits next to a claimed operator** and refused
    when it stands alone - i.e. what the provider refuses is not always the operator itself but the
    small fragment it ends up in. In the depth model, `Exp` is the unsupported node that cuts the head
    into fragments, so removing it is the rule that the mechanism points at.

    Numerically this is an identity (up to rounding), and it is a *declared* substitution rather than
    an equivalence - the numeric gate records the difference and the rule is not accepted without it.
    """
    initializers = _initializer_map(model)
    types = graph_index(model)[2]
    rewritten = []
    for node in list(model.graph.node):
        if node.op_type != "Exp":
            continue
        if only and only not in (node.name or ""):
            continue
        in_type = types.get(node.input[0], ("?",))[0]
        np_dtype = np.float16 if in_type == "FLOAT16" else np.float32
        base = f"{tag}_{(node.name or node.output[0]).strip('/').replace('/', '_')}"
        exponent = numpy_helper.from_array(
            np.array(np.e, dtype=np_dtype).reshape([1] * 0 + [1]), f"{base}_e")
        model.graph.initializer.append(exponent)
        index = list(model.graph.node).index(node)
        model.graph.node.insert(index, helper.make_node(
            "Pow", [f"{base}_e", node.input[0]], [node.output[0]], name=f"{base}/pow"))
        del model.graph.node[index + 1]
        rewritten.append(f"{node.name or ''}: Exp -> Pow(E, x) (Exp is not on the supported list, "
                         f"Pow is)")
    return rewritten


RULES = {"resize": rule_resize_linear, "resize-attrs": rule_resize_nearest_mode,
         "resize-canonical": rule_resize_canonicalise, "resize-fp32": rule_resize_fp16_through_fp32,
         "exp-to-pow": rule_exp_to_pow}


def convert(model_path, out_path, rules, only=None):
    model = onnx.load(model_path)  # external data must resolve for a real conversion
    report = []
    for rule in rules:
        if rule not in RULES:
            print(f"unknown rule: {rule}", file=sys.stderr)
            return 2
        report += [f"  {rule}: {line}" for line in RULES[rule](model, only=only)]
    if not report:
        print("no rewrite applied - nothing in this graph matched the selected rules")
    else:
        print("rewrites:")
        print("\n".join(report))
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"written: {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect", help="describe the operators that tend to fall back")
    inspect_parser.add_argument("model")
    convert_parser = sub.add_parser("convert", help="apply capability-driven rules")
    convert_parser.add_argument("model")
    convert_parser.add_argument("out")
    convert_parser.add_argument("--rules", default=",".join(DEFAULT_RULES))
    convert_parser.add_argument("--only", default=None,
                                help="restrict a rule to nodes whose name contains this string")
    args = parser.parse_args()
    if args.command == "inspect":
        return inspect(args.model)
    return convert(args.model, args.out, [r.strip() for r in args.rules.split(",") if r.strip()],
                   only=args.only)


if __name__ == "__main__":
    sys.exit(main())
