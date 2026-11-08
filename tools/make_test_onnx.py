#!/usr/bin/env python3
"""Build a TINY TEST-ONLY ONNX model (fixed, hand-set weights) to exercise the ONNX backend plumbing.
It is NOT a fire model: it outputs softmax logits computed from mean image brightness. Never deploy it.
    python tools/make_test_onnx.py out.onnx
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(path: str, size: int = 32):
    # input NHWC [1,size,size,3] in 0..1 -> ReduceMean over H,W,C -> [1,1] -> MatMul W[1,4] + b -> Softmax
    W = numpy_helper.from_array(np.array([[-8.0, 2.0, 6.0, 10.0]], dtype=np.float32), "W")
    B = numpy_helper.from_array(np.array([4.0, 0.0, -2.5, -5.0], dtype=np.float32), "B")
    axes = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "axes")
    nodes = [helper.make_node("ReduceMean", ["x", "axes"], ["m"], keepdims=0),
             helper.make_node("Reshape", ["m", "shape"], ["m2"]),
             helper.make_node("MatMul", ["m2", "W"], ["z"]),
             helper.make_node("Add", ["z", "B"], ["logits"]),
             helper.make_node("Softmax", ["logits"], ["probs"], axis=1)]
    shape = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "shape")
    g = helper.make_graph(nodes, "TEST_ONLY_brightness_softmax",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, size, size, 3])],
                          [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 4])],
                          initializer=[W, B, axes, shape])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "TEST_ONLY_model.onnx")
