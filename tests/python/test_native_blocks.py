"""q4e.native_blocks: the checkpoint's expert blocks re-laid per role and
decoded exactly (DESIGN 7.0.2bz, design-routing-aware-expert-execution 2.3a/b).

Two kinds of cell: synthetic blocks whose expected values are derived by
hand from the block layout (the same blocks as tests/test_gguf.cpp's
decoder cells, so the numpy and the C++ side are pinned to one expectation),
and the real shards against gguf-py's own `dequantize` -- an oracle outside
this repository. `Q4E_GGUF_SHARDS` gates the latter, as in test_gguf_feed.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from q4e import native_blocks as nb  # noqa: E402

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "")
_skip = pytest.mark.skipif(not _SHARDS, reason="Q4E_GGUF_SHARDS unset (real GGUF shards absent)")


def _f16_bytes(x):
    return np.array([x], dtype="<f2").view(np.uint8)


def test_iq4_nl_split_and_decode_on_a_hand_built_block():
    # d = 1.0, nibble j low / 15-j high in byte j: y[j] = T[j], y[j+16] = T[15-j]
    block = np.concatenate([_f16_bytes(1.0),
                            np.array([(j & 0xF) | ((15 - j) << 4) for j in range(16)], np.uint8)])
    # a second block at d = 0.5
    two = np.concatenate([block, block]); two[18:20] = _f16_bytes(0.5)
    codes, scales = nb.iq4_nl_split(two[None, :])
    assert codes.shape == (1, 64) and codes.dtype == np.uint8 and scales.shape == (1, 2)
    assert list(codes[0, :16]) == list(range(16)) and list(codes[0, 16:32]) == list(range(15, -1, -1))
    y = nb.iq4_nl_decode(codes, scales)
    want = np.concatenate([nb.KVALUES_IQ4NL, nb.KVALUES_IQ4NL[::-1]]).astype(np.float32)
    assert np.array_equal(y[0, :32], want)
    assert np.array_equal(y[0, 32:], want * np.float32(0.5))


def test_iq3_xxs_split_and_decode_on_a_hand_built_block():
    # d = 1.0; sub-block 0: index pair (1, 0) for the first 8 values, sign
    # index 5 (= 0b101: values 0 and 2 negative), scale 3 -> db = 1.75:
    # [-35, 7, -7, 7, 7, 7, 7, 7]; the rest of sub-block 0: grid 0 -> 7;
    # sub-blocks 1..7: scale 0 -> db 0.25, grid 0 -> 1.0
    block = np.zeros(98, np.uint8)
    block[0:2] = _f16_bytes(1.0)
    block[2] = 1
    block[66:70] = np.array([(3 << 28) | 5], "<u4").view(np.uint8)
    gridix, signix, scales = nb.iq3_xxs_split(block[None, :])
    assert gridix.shape == (1, 64) and signix.shape == (1, 32) and scales.shape == (1, 8)
    assert gridix[0, 0] == 1 and signix[0, 0] == 5 and scales[0, 0] == np.float32(1.75)
    assert np.all(scales[0, 1:] == np.float32(0.25))
    y = nb.iq3_xxs_decode(gridix, signix, scales)
    assert np.array_equal(y[0, :8], np.array([-35, 7, -7, 7, 7, 7, 7, 7], np.float32))
    assert np.all(y[0, 8:32] == 7.0) and np.all(y[0, 32:] == 1.0)


@pytest.fixture(scope="module")
def feed():
    from q4e import gguf_feed as gf
    return gf.GgufFeed(_SHARDS)


@_skip
@pytest.mark.parametrize("name,fmt", [("blk.0.ffn_down_exps.weight", "IQ4_NL"),
                                      ("blk.0.ffn_gate_exps.weight", "IQ3_XXS"),
                                      ("blk.24.ffn_up_exps.weight", "IQ3_XXS")])
def test_the_split_decodes_the_real_shard_exactly_as_gguf_py_does(feed, name, fmt):
    """The oracle outside the repository: gguf-py's dequantize of the same
    bytes. Two experts, every row. The split is a byte re-arrangement, so the
    decode must land on gguf-py's f32 to float rounding, not to a tolerance
    that could hide a wrong table or a swapped nibble."""
    assert feed.gguf_type(name) == fmt
    raw = feed.raw_rows(name, rows=2)                      # [2, out, row_bytes]
    ref = np.asarray(feed.dequant(name, rows=2), np.float32)   # [2, out, in]
    E, out, row_bytes = raw.shape
    flat = raw.reshape(E * out, row_bytes)
    parts = nb.SPLIT[fmt](flat)
    y = (nb.iq4_nl_decode(*parts) if fmt == "IQ4_NL" else nb.iq3_xxs_decode(*parts))
    y = y.reshape(E, out, -1)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    d = np.abs(y - ref)
    print(f"\n[native-blocks] {name} ({fmt}): rows {E * out}, max|diff| {d.max():.3e}, "
          f"max|ref| {np.abs(ref).max():.4f}, exact {np.array_equal(y, ref)}")
    assert np.allclose(y, ref, rtol=1e-6, atol=0.0), f"{name}: max|diff| {d.max():.3e}"
    # the byte budget per role, the numbers the Fit table is recomputed with
    if fmt == "IQ4_NL":
        codes, scales = parts
        assert codes.shape == (E * out, ref.shape[-1]) and scales.shape == (E * out, ref.shape[-1] // 32)
    else:
        gridix, signix, scales = parts
        assert gridix.shape == (E * out, ref.shape[-1] // 4) and signix.shape == (E * out, ref.shape[-1] // 8)
        assert signix.max() <= 127 and scales.shape == (E * out, ref.shape[-1] // 32)
