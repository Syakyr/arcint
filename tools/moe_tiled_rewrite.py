#!/usr/bin/env python3
"""Rewrite a serving-shape IR emitted BEFORE 2026-09-17 into the shape the GPU
plugin's tiled MoE matcher accepts -- in memory, before compile_model, so an
artifact that took hours to export does not have to be exported again for a
compile-time census.

The defect (campaign sub4bit-vram-kernel, status 2026-09-17): `emit_moe_tiled`
fed the down-projection MatMul [E,M,H] straight into the router-weight
Multiply, and the router side went Transpose -> Unsqueeze. The plugin's
ConvertTiledMoeBlockTo3GatherMatmuls (`build_3gemm_pattern`) anchors on
`end_reshape` = Reshape(down MatMul) and `router_reshape` =
Reshape(Transpose(ScatterElementsUpdate)) -> optional Unsqueeze. Without them
the block compiles as 230 FullyConnected over the Tile, every expert for every
token, at full residency.

This pass inserts exactly those two Reshapes, with the construction the fixed
emitter uses: target [E, 1, -1, H] for the expert outputs and [E, 1, -1] for
the router weights. B is the literal 1 this family has (hidden [1,T,H]); the
runtime -1 is what keeps the Reshape from being folded away at validate/save
(export_mtp.py:515-531). The Multiply then runs at [E,1,S,H] x [E,1,S,1] and
the ReduceSum over axis 0 yields [1,S,H] -- value-identical to the old
[M,H] with M = S, and the same shape the fused rewrite produces.

Idempotent: a block whose Multiply already reads a Reshape is left alone, so
the pass is safe on artifacts from the fixed emitter.

What a rewritten model proves: the SAME thing the fixed emitter's artifact
proves at the census (primitive types off `get_runtime_model()`), on the
measured artifact. What it does not give: an on-disk artifact the C++ serve
path can load -- for that either re-export or `ov.save_model` the rewritten
model (a full .bin write).

Usage:
    python3 moe_tiled_rewrite.py <ir.xml>              # walk, rewrite, walk
    python3 moe_tiled_rewrite.py <ir.xml> --out <dir>  # ... and save_model
"""
import argparse
import sys

import numpy as np


def _type(node):
    return node.get_type_name()


def _find_blocks(model):
    """Yield (mul3, matmul_output, unsqueeze) for every old-style tiled MoE
    block: ReduceSum(keep_dims=false) <- Multiply(MatMul, Unsqueeze(Transpose(
    ScatterElementsUpdate))). Either operand order."""
    for rs in model.get_ordered_ops():
        if _type(rs) != "ReduceSum":
            continue
        attrs = rs.get_attributes()
        if str(attrs.get("keep_dims", "false")).lower() == "true":
            continue
        mul3 = rs.input_value(0).get_node()
        if _type(mul3) != "Multiply":
            continue
        a, b = mul3.input_value(0), mul3.input_value(1)
        for outs, uns in ((a, b), (b, a)):
            if _type(outs.get_node()) != "MatMul" or _type(uns.get_node()) != "Unsqueeze":
                continue
            tr = uns.get_node().input_value(0).get_node()
            if _type(tr) != "Transpose":
                continue
            if _type(tr.input_value(0).get_node()) != "ScatterElementsUpdate":
                continue
            if outs.get_partial_shape().rank.get_length() != 3:
                continue
            yield mul3, outs, uns.get_node()
            break


def rewrite_tiled_moe(model):
    """Insert the two matcher-anchoring Reshapes into every old-style block.
    Returns the number of blocks rewritten (0 on an already-conformant
    model). Validates the model afterwards so downstream shapes follow."""
    from openvino import opset13 as op

    n = 0
    for mul3, outs, uns in list(_find_blocks(model)):
        ps = outs.get_partial_shape()
        E, H = ps[0], ps[2]
        if not (E.is_static and H.is_static):
            raise ValueError(f"{mul3.get_friendly_name()}: expert output shape {ps} "
                             f"is not [E static, M, H static]")
        E, H = E.get_length(), H.get_length()
        tag = mul3.get_friendly_name()
        outs4 = op.reshape(outs, op.constant(np.array([E, 1, -1, H], np.int32)),
                           special_zero=False)
        outs4.set_friendly_name(f"{tag}/end_reshape")
        wt = uns.input_value(0)                                     # [E,M]
        wr = op.reshape(wt, op.constant(np.array([E, 1, -1], np.int32)),
                        special_zero=False)
        wr.set_friendly_name(f"{tag}/router_reshape")
        wu = op.unsqueeze(wr, op.constant(np.array([-1], np.int32)))
        # the Multiply keeps its operand order (pattern order: end_reshape
        # first when the emitter wrote it that way)
        for i in range(2):
            src = mul3.input_value(i).get_node()
            if src is outs.get_node():
                mul3.input(i).replace_source_output(outs4.output(0))
            elif src is uns:
                mul3.input(i).replace_source_output(wu.output(0))
        n += 1
    if n:
        model.validate_nodes_and_infer_types()
    return n


def walk(model):
    """(matched, failed-by-constraint) from tools/check_tiled_pattern.py."""
    import check_tiled_pattern as ctp
    ok, fail = [], {}
    for rs in model.get_ordered_ops():
        if _type(rs) != "ReduceSum":
            continue
        try:
            ctp.check_3gemm_from_reduce_sum(rs, lambda s: None)
            ok.append(rs.get_friendly_name())
        except ctp.Fail as f:
            fail[rs.get_friendly_name()] = f.constraint
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir")
    ap.add_argument("--out", default=None, help="directory to save_model the rewritten IR into")
    a = ap.parse_args()
    import os
    import resource
    import openvino as ov
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    core = ov.Core()
    m = core.read_model(a.ir)
    before, _ = walk(m)
    n = rewrite_tiled_moe(m)
    after, fail = walk(m)
    print(f"blocks rewritten {n}; walker matched before {len(before)} after {len(after)}; "
          f"failing constraints after: {sorted(set(fail.values()))}; "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        xml = os.path.join(a.out, os.path.basename(a.ir))
        ov.save_model(m, xml, compress_to_fp16=False)
        print(f"saved {xml}")
    return 0 if (n == 0 or len(after) >= len(before) + n) else 1


if __name__ == "__main__":
    sys.exit(main())
