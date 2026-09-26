"""The checkpoint's own expert blocks, re-laid per tensor, and their exact
decode -- the numpy side of DESIGN 7.0.2bz / design-routing-aware-expert-
execution 2.3a-2.3b (2026-09-18).

The shipped UD-Q3_K_XL GGUF stores the routed experts as IQ3_XXS (gate, up)
and IQ4_NL (down). Re-quantising them into the plugin's u4 grouped-affine
codes costs 0.10-0.13 relative RMS per expert tensor and 0.73 nats of KL at
depth 48 (measured), so the experts are carried as the GGUF's own numbers.
The GGUF interleaves scale, indices and signs inside each block; the
emitter wants one Constant per role, so this module re-lays a tensor's raw
block bytes into per-role arrays (a byte re-arrangement, no arithmetic on
the codes) and decodes them exactly the way ggml does -- the oracle for
every cell here is gguf-py's own `dequantize`, and the C++ twins are
`src/core/gguf_dequant.cpp`'s `dequantize_row_iq4_nl` / `_iq3_xxs`.

Block layouts (`code`: llama.cpp `ggml-common.h`, `ggml-quants.c`, pinned
clone 56b9eb28):

  IQ4_NL   per 32 values, 18 B: f16 d | 16 B nibbles; y[j] = d * T[qs[j] & 15],
           y[j+16] = d * T[qs[j] >> 4], T = KVALUES_IQ4NL (16 signed entries)
  IQ3_XXS  per 256 values, 98 B: f16 d | 64 B grid indices | 8 x u32
           scales-and-signs; per 32-value sub-block ib: indices qs[8ib..8ib+8]
           (each selects 4 magnitudes of GRID[256]), aux = u32[ib]: for the
           l-th 8 values (l = 0..3) the sign mask is KSIGNS[(aux >> 7l) & 127]
           (bit j flips value j), the sub-block scale is (aux >> 28);
           y = d * (0.5 + s) * 0.5 * magnitude * sign

Per-role tensors (row-major over [rows, ...], the roles the plugin's nine
per-expert slots carry -- weights / scales / zero-points):

  IQ4_NL   codes   u4  [rows, in]        linear nibble order (element k)
           scales  f32 [rows, in // 32]  d per block
  IQ3_XXS  gridix  u8  [rows, in // 4]   one grid index per 4 values
           signix  u8  [rows, in // 8]   one 7-bit sign-mask index per 8 values
           scales  f32 [rows, in // 32]  d * (0.5 + s) * 0.5 per sub-block
"""
import numpy as np

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                          1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

KMASK_IQ2XS = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

KSIGNS_IQ2XS = np.array([
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
], dtype=np.uint8)

_IQ3XXS_GRID_U32 = np.array([
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
], dtype=np.uint32)

# [256, 4] magnitudes: the uint32's little-endian bytes are values 0..3
IQ3XXS_GRID = _IQ3XXS_GRID_U32.view(np.uint8).reshape(256, 4).copy()

IQ4_NL_BLOCK, IQ4_NL_BYTES = 32, 18
IQ3_XXS_BLOCK, IQ3_XXS_BYTES = 256, 98
IQ4_XS_BLOCK, IQ4_XS_BYTES = 256, 136
Q8_0_BLOCK, Q8_0_BYTES = 32, 34

# (values per block, bytes per block) per format this module splits. The
# shipped checkpoint mixes them per layer (read, not assumed, 2026-09-18):
# 43 layers IQ3_XXS gate/up over an IQ4_NL down; layer 2 IQ4_XS gate/up over
# a Q8_0 down; layers 4, 30, 46, 47 IQ3_XXS gate/up over a Q8_0 down.
BLOCK_BYTES = {"IQ4_NL": (IQ4_NL_BLOCK, IQ4_NL_BYTES), "IQ3_XXS": (IQ3_XXS_BLOCK, IQ3_XXS_BYTES),
               "IQ4_XS": (IQ4_XS_BLOCK, IQ4_XS_BYTES), "Q8_0": (Q8_0_BLOCK, Q8_0_BYTES)}


def _f16_le(bytes2):
    """[..., 2] u8 -> f32: the block's little-endian f16 scale."""
    b = np.ascontiguousarray(bytes2, dtype=np.uint8)
    return b.view("<f2").reshape(b.shape[:-1]).astype(np.float32)


# --------------------------------------------------------------------- IQ4_NL
def iq4_nl_split(raw):
    """raw: u8 [rows, in // 32 * 18] -> (codes u8 [rows, in] in 0..15 in LINEAR
    element order, scales f32 [rows, in // 32])."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    rows, nbytes = raw.shape
    assert nbytes % IQ4_NL_BYTES == 0, f"{nbytes} B is not whole IQ4_NL blocks"
    nb = nbytes // IQ4_NL_BYTES
    blk = raw.reshape(rows, nb, IQ4_NL_BYTES)
    scales = _f16_le(blk[:, :, 0:2])                               # [rows, nb]
    qs = blk[:, :, 2:]                                              # [rows, nb, 16]
    codes = np.empty((rows, nb, IQ4_NL_BLOCK), dtype=np.uint8)
    codes[:, :, :16] = qs & 0x0F                                    # y[j]    <- low nibble
    codes[:, :, 16:] = qs >> 4                                      # y[j+16] <- high nibble
    return codes.reshape(rows, nb * IQ4_NL_BLOCK), scales


def iq4_nl_decode(codes, scales):
    """The exact decode of the split form: y = d * T[code]."""
    rows, inn = codes.shape
    vals = KVALUES_IQ4NL[codes.astype(np.int64)].astype(np.float32)   # [rows, in]
    return (vals.reshape(rows, inn // IQ4_NL_BLOCK, IQ4_NL_BLOCK)
            * scales[:, :, None]).reshape(rows, inn)


# -------------------------------------------------------------------- IQ3_XXS
def iq3_xxs_split(raw):
    """raw: u8 [rows, in // 256 * 98] -> (gridix u8 [rows, in // 4], signix u8
    [rows, in // 8] (0..127), scales f32 [rows, in // 32])."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    rows, nbytes = raw.shape
    assert nbytes % IQ3_XXS_BYTES == 0, f"{nbytes} B is not whole IQ3_XXS blocks"
    nb = nbytes // IQ3_XXS_BYTES
    blk = raw.reshape(rows, nb, IQ3_XXS_BYTES)
    d = _f16_le(blk[:, :, 0:2])                                     # [rows, nb]
    qs = blk[:, :, 2:66]                                            # [rows, nb, 64]: 8 per sub-block
    aux = np.ascontiguousarray(blk[:, :, 66:98]).view("<u4").reshape(rows, nb, 8)   # [rows, nb, 8 sub-blocks]
    s4 = (aux >> 28).astype(np.float32)
    scales = (d[:, :, None] * (0.5 + s4) * 0.5).reshape(rows, nb * 8)  # per 32 values
    signix = np.stack([(aux >> (7 * l)) & 127 for l in range(4)], axis=-1)  # [rows, nb, 8, 4]
    return (qs.reshape(rows, nb * 64), signix.astype(np.uint8).reshape(rows, nb * 32), scales)


def iq3_xxs_decode(gridix, signix, scales):
    """The exact decode of the split form."""
    rows, n4 = gridix.shape
    inn = n4 * 4
    mag = IQ3XXS_GRID[gridix.astype(np.int64)].astype(np.float32).reshape(rows, inn)
    masks = KSIGNS_IQ2XS[signix.astype(np.int64)]                       # [rows, in // 8]
    bits = (masks[:, :, None] & KMASK_IQ2XS[None, None, :]) != 0       # [rows, in // 8, 8]
    sign = np.where(bits, np.float32(-1.0), np.float32(1.0)).reshape(rows, inn)
    y = mag * sign
    return (y.reshape(rows, inn // 32, 32) * scales[:, :, None].astype(np.float32)).reshape(rows, inn)


# --------------------------------------------------------------------- IQ4_XS
def iq4_xs_split(raw):
    """raw: u8 [rows, in // 256 * 136] -> (codes u8 [rows, in] in 0..15 in
    LINEAR element order, scales f32 [rows, in // 32]) -- the IQ4_NL layout.

    A 256-value block carries one f16 d and eight 6-bit sub-block scales
    (low nibble in scales_l, two high bits in scales_h): the per-32 scale is
    d * (ls - 32), the nibbles index the same 16-entry table in the same
    order as IQ4_NL (ggml-quants.c dequantize_row_iq4_xs), so the split lands
    on the IQ4_NL layout and the IQ4_NL decode reads it."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    rows, nbytes = raw.shape
    assert nbytes % IQ4_XS_BYTES == 0, f"{nbytes} B is not whole IQ4_XS blocks"
    nb = nbytes // IQ4_XS_BYTES
    blk = raw.reshape(rows, nb, IQ4_XS_BYTES)
    d = _f16_le(blk[:, :, 0:2])                                     # [rows, nb]
    scales_h = np.ascontiguousarray(blk[:, :, 2:4]).view("<u2").reshape(rows, nb).astype(np.int64)
    scales_l = blk[:, :, 4:8].astype(np.int64)                      # [rows, nb, 4]
    ib = np.arange(8)
    lo = (scales_l[:, :, ib // 2] >> (4 * (ib % 2))) & 0xF          # [rows, nb, 8]
    hi = (scales_h[:, :, None] >> (2 * ib)) & 3
    ls = lo | (hi << 4)
    scales = (d[:, :, None] * (ls - 32).astype(np.float32)).reshape(rows, nb * 8)
    qs = blk[:, :, 8:136].reshape(rows, nb, 8, 16)                  # 16 bytes per sub-block
    codes = np.empty((rows, nb, 8, 32), dtype=np.uint8)
    codes[..., :16] = qs & 0x0F
    codes[..., 16:] = qs >> 4
    return codes.reshape(rows, nb * IQ4_XS_BLOCK), scales


iq4_xs_decode = iq4_nl_decode


# ----------------------------------------------------------------------- Q8_0
def q8_0_split(raw):
    """raw: u8 [rows, in // 32 * 34] -> (codes i8 [rows, in], scales f32
    [rows, in // 32]): y = d * q (ggml-quants.c dequantize_row_q8_0)."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    rows, nbytes = raw.shape
    assert nbytes % Q8_0_BYTES == 0, f"{nbytes} B is not whole Q8_0 blocks"
    nb = nbytes // Q8_0_BYTES
    blk = raw.reshape(rows, nb, Q8_0_BYTES)
    scales = _f16_le(blk[:, :, 0:2])
    codes = np.ascontiguousarray(blk[:, :, 2:]).view(np.int8).reshape(rows, nb * Q8_0_BLOCK)
    return codes, scales


def q8_0_decode(codes, scales):
    rows, inn = codes.shape
    return (codes.astype(np.float32).reshape(rows, inn // Q8_0_BLOCK, Q8_0_BLOCK)
            * scales[:, :, None]).reshape(rows, inn)


SPLIT = {"IQ4_NL": iq4_nl_split, "IQ3_XXS": iq3_xxs_split, "IQ4_XS": iq4_xs_split, "Q8_0": q8_0_split}
DECODE = {"IQ4_NL": iq4_nl_decode, "IQ3_XXS": iq3_xxs_decode, "IQ4_XS": iq4_xs_decode, "Q8_0": q8_0_decode}
