"""serving_shape._native_expert: the checkpoint's expert blocks decoded in
standard ops, exact on the CPU plugin (design-routing-aware-expert-execution
2.3b). Device-free cells fill the per-role tensors with RANDOM bytes -- a
fill that is NOT alike across rows or blocks (the blind-fill lesson of
2026-09-08: a fill that makes every page alike certifies nothing) -- and
compare a MatMul over the chain with numpy's exact decode; the shard-gated
cell does the same with two real experts through NativeExpertFiller.
"""
import os
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import Type, opset13 as op

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import native_blocks as nb  # noqa: E402
from q4e import serving_shape as ss  # noqa: E402

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "")
_skip = pytest.mark.skipif(not _SHARDS, reason="Q4E_GGUF_SHARDS unset (real GGUF shards absent)")


def _random_raw(rng, rows, inn, fmt):
    """Valid random blocks: every byte random, the f16 scale finite and non-zero."""
    if fmt == "IQ4_NL":
        block, nbytes = nb.IQ4_NL_BLOCK, nb.IQ4_NL_BYTES
    else:
        block, nbytes = nb.IQ3_XXS_BLOCK, nb.IQ3_XXS_BYTES
    nblk = inn // block
    raw = rng.integers(0, 256, size=(rows, nblk, nbytes), dtype=np.uint8)
    d = rng.uniform(0.01, 0.2, size=(rows, nblk)).astype("<f2")
    raw[:, :, 0:2] = d.view(np.uint8).reshape(rows, nblk, 2)
    return raw.reshape(rows, nblk * nbytes)


def _matmul_model(arena, e, out, inn, fmt, parts, T):
    x = op.parameter([1, T, inn], Type.f32, name="x")
    w = ss._native_expert(arena, e, out, inn, f"t/experts_{fmt}", fmt, parts)
    y = op.matmul(x, w, transpose_a=False, transpose_b=True)           # [E, T, out]
    res = op.result(y)
    return ov.Model([res], [x], "native_expert_chain")


@pytest.mark.parametrize("fmt,inn", [("IQ4_NL", 64), ("IQ3_XXS", 512)])
def test_the_chain_decodes_random_blocks_exactly(fmt, inn):
    rng = np.random.default_rng({"IQ4_NL": 11, "IQ3_XXS": 13}[fmt])
    e, out, T = 3, 5, 7
    raw = _random_raw(rng, e * out, inn, fmt)
    parts = nb.SPLIT[fmt](raw)
    w_ref = (nb.iq4_nl_decode(*parts) if fmt == "IQ4_NL" else nb.iq3_xxs_decode(*parts)).reshape(e, out, inn)
    # not alike: every row differs from every other, and no row is constant
    assert len({r.tobytes() for r in w_ref.reshape(e * out, inn)}) == e * out
    assert (w_ref.reshape(e * out, inn).std(axis=1) > 0).all()
    arena = ss.SparseArena()
    try:
        model = _matmul_model(arena, e, out, inn, fmt, parts, T)
        x = rng.standard_normal((1, T, inn)).astype(np.float32)
        compiled = ov.Core().compile_model(model, "CPU")
        y = compiled({"x": x})[compiled.output(0)]
    finally:
        arena.close()
    want = np.einsum("tk,eok->eto", x[0], w_ref)
    d = np.abs(y - want)
    print(f"\n[native-chain] {fmt} inn={inn}: max|diff| {d.max():.3e} vs max|want| {np.abs(want).max():.3e}")
    assert y.shape == (e, T, out)
    assert np.allclose(y, want, rtol=1e-5, atol=1e-5), f"{fmt}: max|diff| {d.max():.3e}"
    # the chain is standard ops only: nothing the plugin does not know
    types = {n.get_type_name() for n in model.get_ordered_ops()}
    assert types <= {"Parameter", "Constant", "Convert", "Gather", "Reshape", "Multiply", "Unsqueeze",
                     "BitwiseAnd", "Greater", "Select", "MatMul", "Result"}, types


def test_an_unknown_format_is_refused():
    arena = ss.SparseArena()
    try:
        with pytest.raises(ValueError):
            ss._native_expert(arena, 1, 1, 32, "t/x", "Q3_K", (None,))
    finally:
        arena.close()


@_skip
@pytest.mark.parametrize("kind,fmt", [("down", "IQ4_NL"), ("gate", "IQ3_XXS")])
def test_two_real_experts_through_the_native_filler_match_gguf_py(kind, fmt):
    from q4e import expert_fill as ef, gguf_feed as gf
    feed = gf.GgufFeed(_SHARDS)
    filler = ef.NativeExpertFiller(feed)
    e, T = 2, 3
    out, inn = (2560, 640) if kind == "down" else (640, 2560)
    got_fmt, parts = filler.native(0, kind, e, out, inn)
    assert got_fmt == fmt
    ref = np.asarray(feed.dequant(f"blk.0.ffn_{kind}_exps.weight", rows=e), np.float32)   # gguf-py, [e,out,inn]
    rng = np.random.default_rng(0)
    arena = ss.SparseArena()
    try:
        model = _matmul_model(arena, e, out, inn, fmt, parts, T)
        x = rng.standard_normal((1, T, inn)).astype(np.float32)
        compiled = ov.Core().compile_model(model, "CPU")
        y = compiled({"x": x})[compiled.output(0)]
    finally:
        arena.close()
    want = np.einsum("tk,eok->eto", x[0], ref)
    d = np.abs(y - want)
    print(f"\n[native-chain real] {kind} {fmt}: max|diff| {d.max():.3e} vs max|want| {np.abs(want).max():.3e}; "
          f"census {filler.census()}")
    assert np.allclose(y, want, rtol=1e-4, atol=1e-4), f"{kind}: max|diff| {d.max():.3e}"
    assert filler.census()["bodies"] == 1 and filler.census()["format"] == "native"
