"""THE ROW-ID PRODUCER, test side of the parity seam -- moved out of
tests/python/test_ple_block.py (feed-the-ports increment) so the boot driver
can feed `ngram_chunk_ids` / `ngram_local_ids` from the same generator the
PLE parity cells validate against the committed Link-3 vectors
(tests/ngram_row_ids_vectors.h). Serving uses src/exec/ngram_row_ids.h; this
is its Python twin, byte-for-byte on the vectors, and it is NOT reimplemented
anywhere else.

Python ints are true 64-bit and the result is np.int64 -- no float touches
the index path. The hash constants are derived with the pinned reference's
own `_build_layer_multipliers` / `_find_nth_prime_after`, imported lazily so
a caller that only needs `split_by_partition` does not need the pin.
"""
import numpy as np


def _pin():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as pin_mod
    return pin_mod


# --- the index derivation, transcribed (matches src/exec/ngram_row_ids.h) ----
_MASK64 = (1 << 64) - 1


def _s64(x):
    x &= _MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def derive(vocab_size, ngram_size, heads_per_ngram, base, ple_idx, seed=1234):
    nh = (ngram_size - 1) * heads_per_ngram
    mult = _pin()._build_layer_multipliers(vocab_size, ngram_size, ple_idx, seed).tolist()
    sizes, offs, tot = [], [], 0
    for h in range(nh):
        gh = ple_idx * nh + h
        sz = _pin()._find_nth_prime_after(base - 1, gh + 1)
        sizes.append(sz)
        offs.append(tot)
        tot += sz
    return mult, sizes, offs


def row_ids(mult, sizes, offs, ng, hpn, eos, context, tokens):
    ctx = ng - 1
    packed = list(context) + list(tokens)
    W = len(packed)
    prev, last = [-1] * W, -1
    for p in range(W):
        prev[p] = last
        if packed[p] == eos:
            last = p
    in_seg = [p - prev[p] - 1 for p in range(W)]
    shifted = [list(packed)] + [
        [packed[p - s] if (p - s >= 0 and in_seg[p] >= s) else eos for p in range(W)]
        for s in range(1, ng)
    ]
    per = [[r[ctx + i] for i in range(len(tokens))] for r in shifted]
    out = []
    for i in range(len(tokens)):
        heads = []
        for n in range(2, ng + 1):
            st = (n - 2) * hpn
            mixed = _s64(per[0][i] * mult[0])
            for pos in range(1, n):
                mixed = _s64((mixed & _MASK64) ^ (_s64(per[pos][i] * mult[pos]) & _MASK64))
            for h in range(st, st + hpn):
                heads.append(mixed % sizes[h] + offs[h])
        out.append(heads)
    return out


def gen_row_ids(config, ple_idx, tokens):
    """The TEST-SIDE row-id producer (this side of the parity seam; serving
    uses src/exec/ngram_row_ids.h). Fresh (all-eos) context, no cache. Python
    ints are true 64-bit and the result is np.int64 -- no float touches the
    index path. Returns [1, T, num_ngram_heads]."""
    mult, sizes, offs = derive(config.vocab_size, config.ngram_size,
                                config.heads_per_ngram, config.ngram_vocab_size_base, ple_idx)
    eos = int(config.eos_token_id)
    ctx = [eos] * (config.ngram_size - 1)
    rows = row_ids(mult, sizes, offs, config.ngram_size, config.heads_per_ngram,
                    eos, ctx, [int(t) for t in tokens])
    return np.array(rows, dtype=np.int64)[None]  # [1, T, Hn]


def ple_ordinal(config, layer_idx):
    """WHICH hash constants a PLE layer uses: the reference derives them from
    the layer's ORDINAL among the PLE layers -- modeling_qwen4_exp.py:1268,
    `config.ple_layer_ids.index(layer_idx + 1)` (the ids are 1-based) -- and
    the served loader does the same (`derive_hash_constants(..., k)` for the
    k-th entry of ple_layer_ids). NOT the decoder layer index: the real
    model's PLE sits at decoder layer 1, is the first (only) PLE layer, and
    hashes with ordinal 0. The parity cells use 1 on both of their sides,
    which is consistent there and wrong here -- the boot driver fed 1 in its
    first real-weight run and that run is retracted on that point
    (window-050 §4.8)."""
    ids = getattr(config, "ple_layer_ids", None)
    if ids:
        ids = list(ids)
        if layer_idx + 1 in ids:
            return ids.index(layer_idx + 1)
        raise ValueError(f"decoder layer {layer_idx} is not a PLE layer: ple_layer_ids={ids}")
    return 0


def split_by_partition(global_ids, rows_per_chunk):
    """The HOST'S half of the chunked table contract: global row ids ->
    (chunk ids i32, local ids i64) at the port partition read off the first
    `ngram_table.K` port's row count. Exact integer arithmetic here; the graph
    does none (q4e.serving_shape.ngram_chunked_gather)."""
    g = np.asarray(global_ids, dtype=np.int64)
    per = int(rows_per_chunk)
    return (np.ascontiguousarray((g // per).astype(np.int32)),
            np.ascontiguousarray((g % per).astype(np.int64)))


__all__ = ["derive", "row_ids", "gen_row_ids", "ple_ordinal", "split_by_partition"]
