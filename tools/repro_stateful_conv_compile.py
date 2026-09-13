#!/usr/bin/env python3
"""Localise the depth-1 control's compile refusal (window-050 §4.6, P5 died).

The stateful depth-1 graph refused to compile on BOTH cards with

    Error has occured for: convolution:GroupConvolution_118
    Weights feature maps number(=1) is not equal to: input feature maps
    number(=10240) | Weights/ifm mismatch

That op is `q4e.serving_shape.stateful_short_conv`'s depthwise
GroupConvolution -- rank-4 [conv_dim, 1, 1, K] weights, groups = conv_dim,
the shape `PagedCausalConv1DFusion` matches. On the SERVED path the fusion
replaces it (the depth-4 legs show `PagedCausalConv1D` x3 after the pass), so
the raw op only reaches the plugin on the unfused control. The default
emitter (`q4e.gdn._causal_conv_silu`) is a K-term slice/multiply unroll and
carries no GroupConvolution at all, which is why `RUN@be57428`'s control
compiled and this one does not.

Three variants, one op each, at the real conv geometry (conv_dim 10240, K 4):

    stateful          the construct as emitted: ReadValue [?,conv_dim,K] ->
                      Concat(axis=-1) -> GroupConvolution -> Slice
    groupconv_static  the same GroupConvolution over a STATIC [1,conv_dim,K+T]
                      input -- no state, no Concat, no dynamic length
    default           q4e.gdn._causal_conv_silu, the control that compiled

Whichever of `stateful` / `groupconv_static` refuses names the root: the op
itself, or its dynamic-length input. Nothing here is a fix.

    <venv>/bin/python tools/repro_stateful_conv_compile.py <variant> <device>
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

CONV_DIM, K, T = 10240, 4, 5


def main(argv):
    variant, dev = argv[1], argv[2]
    import openvino as ov
    from openvino import Model, opset13 as op
    from q4e import gdn as qgdn
    from q4e import serving_shape as ss

    rng = np.random.default_rng(0)
    conv_w = rng.standard_normal((CONV_DIM, 1, K)).astype(np.float32)
    sinks, params = [], []
    if variant == "stateful":
        x = op.parameter([1, CONV_DIM, T], ov.Type.f32)
        beam = op.parameter([-1], ov.Type.i32)
        beam.set_friendly_name("beam_idx")
        params = [x, beam]
        y = ss.stateful_short_conv(0, beam, sinks)(x, conv_w, T, CONV_DIM, K)
    elif variant == "groupconv_static":
        x = op.parameter([1, CONV_DIM, K + T], ov.Type.f32)
        params = [x]
        w = op.constant(np.ascontiguousarray(conv_w).reshape(CONV_DIM, 1, 1, K))
        y = op.group_convolution(x, w, strides=[1], pads_begin=[0], pads_end=[0],
                                 dilations=[1])
    elif variant == "default":
        x = op.parameter([1, CONV_DIM, T], ov.Type.f32)
        params = [x]
        y = qgdn._causal_conv_silu(x, conv_w, T, CONV_DIM, K)
    else:
        raise SystemExit(f"unknown variant {variant}")
    params[0].set_friendly_name("x")
    params[0].output(0).set_names({"x"})
    if len(params) > 1:
        params[1].output(0).set_names({"beam_idx"})
    model = Model([op.result(y)], sinks, params, f"conv_{variant}")
    n_ops = len(model.get_ordered_ops())
    core = ov.Core()
    props = {"INFERENCE_PRECISION_HINT": "f32"} if dev.startswith("GPU") else {}
    t0 = time.time()
    try:
        compiled = core.compile_model(model, dev, props)
    except Exception as exc:                                      # noqa: BLE001
        text = " | ".join(s.strip() for s in str(exc).splitlines() if s.strip())
        print(f"REPRO {variant:16s} {dev:6s} ops={n_ops} COMPILE FAIL "
              f"{time.time() - t0:.2f}s {type(exc).__name__}: {text}")
        return 0
    req = compiled.create_infer_request()
    x_in = rng.standard_normal(
        list(params[0].get_partial_shape().to_shape())).astype(np.float32)
    req.set_tensor("x", ov.Tensor(x_in))
    if len(params) > 1:
        req.set_tensor("beam_idx", ov.Tensor(np.zeros((1,), np.int32)))
    req.infer()
    val = req.get_output_tensor(0).data
    print(f"REPRO {variant:16s} {dev:6s} ops={n_ops} COMPILE OK "
          f"{time.time() - t0:.2f}s out{tuple(val.shape)} "
          f"finite={bool(np.isfinite(val).all())}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
