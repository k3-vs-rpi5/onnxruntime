#!/usr/bin/env python3
"""Generate a pure-matrix QDQ probe model: the AI cluster's own int8 ceiling, as an instrument.

Why this shape. ONNX's `MatMul` type constraints do not admit `int8`, so a probe cannot be written
as `MatMul(int8, int8)`: the provider's int8 kernel is reached through the **QDQ** form its own
shipped models use - a quantised activation, a per-channel-symmetric quantised constant weight, both
brought back to float by `DequantizeLinear`, a float `MatMul`, then `QuantizeLinear` ->
`DequantizeLinear` on the output (the pattern read out of `vit_b_16.q.onnx`'s PPQ graphs on
2026-09-22). Two artefacts of the first attempt are kept in the record: the int8-`MatMul` model is
rejected by ONNX Runtime's checker, and that is *only* visible on the board.

Two activation forms exist, because the first one crashed the board (2026-09-22, recorded in the
evidence): with `--folded` the activation is a *constant* initializer (nothing is fed, and the
runtime is free to constant-fold the whole matmul at load); the default form feeds the activation as
a real graph input, which is the deployment shape and the one the measurements use.

    make-ime-probe.py [--folded] <path.onnx> <M> <K> <N>

MACs per inference = M x K x N; the achieved rate is 2 x MACs / time.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(path, m, k, n, folded=False):
    rng = np.random.default_rng(7)
    x = rng.integers(-127, 127, size=(1, m, k), dtype=np.int8)
    w = rng.integers(-127, 127, size=(k, n), dtype=np.int8)
    x_scale = np.array(0.01, dtype=np.float32)
    x_zero = np.array(0, dtype=np.int8)
    w_scale = (rng.random(n).astype(np.float32) * 0.01 + 0.001).astype(np.float32)
    w_zero = np.zeros(n, dtype=np.int8)
    y_scale = np.array(0.02, dtype=np.float32)
    y_zero = np.array(0, dtype=np.int8)

    nodes = [
        helper.make_node("DequantizeLinear", ["W_q", "w_scale", "w_zero"], ["W_dq"],
                         name="probe_w_dequant"),
        helper.make_node("MatMul", ["X_dq", "W_dq"], ["Y_f"], name="probe_matmul"),
        helper.make_node("QuantizeLinear", ["Y_f", "y_scale", "y_zero"], ["Y_q"],
                         name="probe_y_quant"),
        helper.make_node("DequantizeLinear", ["Y_q", "y_scale", "y_zero"], ["Y"],
                         name="probe_y_dequant"),
    ]
    initializers = [
        numpy_helper.from_array(w, "W_q"),
        numpy_helper.from_array(x_scale, "x_scale"),
        numpy_helper.from_array(x_zero, "x_zero"),
        numpy_helper.from_array(w_scale, "w_scale"),
        numpy_helper.from_array(w_zero, "w_zero"),
        numpy_helper.from_array(y_scale, "y_scale"),
        numpy_helper.from_array(y_zero, "y_zero"),
    ]
    if folded:
        initializers.append(numpy_helper.from_array(x, "X_q"))
        nodes.insert(0, helper.make_node("DequantizeLinear", ["X_q", "x_scale", "x_zero"], ["X_dq"],
                                         name="probe_x_dequant"))
        inputs = []
    else:
        inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, m, k])]
        nodes.insert(0, helper.make_node("QuantizeLinear", ["X", "x_scale", "x_zero"], ["X_q"],
                                         name="probe_x_quant"))
        nodes.insert(1, helper.make_node("DequantizeLinear", ["X_q", "x_scale", "x_zero"], ["X_dq"],
                                         name="probe_x_dequant"))
    graph = helper.make_graph(
        nodes, "ime_probe",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, m, n])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"{path}: MatMul {m}x{k}x{n} = {m * k * n / 1e9:.3f} GMAC "
          f"({2 * m * k * n / 1e9:.3f} GFLOP) per inference")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--folded"]
    build(args[0], int(args[1]), int(args[2]), int(args[3]), folded="--folded" in sys.argv)
