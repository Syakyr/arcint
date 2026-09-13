"""THE CONTRACT TEST AS HANDSHAKE: the serving-shape IR against what the C++
serving path actually expects, tensor-for-tensor-name.

A green contract turns the 0.5.0 window from an experiment into a boot: if the
IR the exporter emits satisfies, name for name and shape for shape, what
`src/exec/backend_ov.cpp` reads at load time, then the remaining risk is the
card and not the artifact. Everything here is device-free and needs no shards:
it reads an ov::Model built over sparse pages.

TWO KINDS OF CELL, and the distinction is the point:

  * MET -- the export side satisfies the C++ side, asserted directly.
  * NOT MET AND NAMED -- a strict xfail that carries the exact file:line where
    the handshake dies. `strict=True` means the day someone closes the gap the
    cell FAILS, which is how a known gap gets retired instead of forgotten.

WHAT THIS FILE FOUND, on its first run (2026-09-12), and it is a finding about
the C++ and not about the export:

`slot_pool_from_ir` (backend_ov.cpp:577-623) identifies a MoE layer by
    std::string tname = node->get_type_name();  ... tolower ...
    if (tname.find("moe") == std::string::npos) continue;      // :585
NO ARCINT-EXPORTED IR CARRIES AN OP WHOSE TYPE NAME CONTAINS "moe". Measured
over the whole model store on the dev host, 2026-09-12, widened by
REVIEW 2a45349 and re-run independently here:

    find /models/ov -name '*.xml' -size +100k | wc -l   ->  52
    find /models/ov -name '*.xml'             | wc -l   -> 172   (no filter)
    ... of 172, carrying a <layer type="...moe...">     ->   0
    ... of 172, carrying the string "moe" at all        ->   0

The negative is stronger than this file first stated it: 0 of 172, not 0 of 52.
The population includes `qwen36-35b-a3b-int4-ov/openvino_language_model.xml`,
the 35B-A3B MoE checkpoint that is the ground truth `moe_block_tiled` was extracted from
(export_mtp.py:406-409). Its op histogram is Const/Convert/Multiply/Reshape/
Subtract/MatMul/Swish -- the tiled dequant chain -- and no fused MoE node,
because the fusion (`ConvertTiledMoeBlockToGatherMatmuls`) is a GPU-PLUGIN
COMPILE-TIME pass and `slot_pool_from_ir` runs on `read_model`, before it.

So the analytic IR route returns nullopt on every artifact this fleet serves,
and the `else` branch at backend_ov.cpp:3746+ (3 x hidden x moe_intermediate x
bytes-per-weight per expert, from config.json) is what has always run. This is
not a defect the serving-shape IR introduces and it is not one it can fix from
the export side: an exporter cannot give a node a different OpenVINO type name.
It is recorded here, with the line, because the mission's standard is that the
first failure named exactly is worth more than a success.

`slot_pool_from_tiled_ir` below is the matcher that WOULD work on the shape
arcint actually exports -- pattern, not type name -- and its figures are
cross-checked against `src/exec/flash_next_offload.h:45`
(kFlashNextSliceBytes = 2,457,600 B per expert-layer for gate+up+down).

WHAT THIS FILE FOUND SECOND (2026-09-13), and it is a correction TO THIS FILE:

the paged serving ports are not something an exporter emits. `load_paged` runs
`ov::pass::SDPAToPagedAttention` over the artifact it just read, before it
compiles (backend_ov.cpp:2574), and every one of the ports in the table below
is that pass's output -- measured both ways, on the real served artifact and on
this emitter's own, in the block above `paged_census`. The cells here used to
check the emitter's parameter list against the table and call the gap "NOT
emitted"; they now run the pass and check what it produces. The gap was real
and it was bigger than it read: at the time of that reading this IR carried
none of the three constructs the pass converts, and the pass refused by name at
`sdpa_to_paged_attention.cpp:75` before it looked at anything else.

WHERE THAT STANDS NOW is the ports table's own `status` column and nothing
else -- and the paragraph that used to sit here, describing a half-done state,
is why that sentence is not decoration: it went stale inside the same batch
that wrote it. All three constructs are emitted, every row of the table reads
`present`, and the strict xfail retired. Read the table, not this paragraph.
"""
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import openvino as ov  # noqa: E402
# THE pass `load_paged` runs before it compiles (backend_ov.cpp:2574 registers
# `ov::pass::SDPAToPagedAttention`); this is its Python entry point, and the
# cells below run it rather than describing what it would do.
from openvino._offline_transformations import (  # noqa: E402
    paged_attention_transformation)

from q4e import attention as qattn  # noqa: E402
from q4e import piecewise_export as pwe  # noqa: E402
from q4e import serving_shape as ss  # noqa: E402

# A real-geometry build of the whole 48-layer stack is expensive; the contract
# is per-layer and the layer kinds repeat with period 4, so 8 layers exercises
# 6 GDN + 2 dense-causal-attention layers AND the PLE layer (index 1). The
# full-stack build is its own cell, opt-in via Q4E_SERVING_FULL=1, because it
# is the keystone claim and must be run deliberately.
_CONTRACT_LAYERS = 8
_T = 8

# A DATED HOST CENSUS, NOT A GENERATED COUNT -- and the distinction is the
# point of writing it this way. This suite cannot regenerate these figures: a
# staged tree has no /models/ov, and the population is the dev host's model
# store rather than anything this repository owns. So they are recorded with
# their date and their exact command, and the only thing asserted below is the
# ZERO, which is the load-bearing half.
#
# Measured 2026-09-12, twice -- first by 198b736 over the size-filtered
# population, then WIDER by REVIEW 2a45349 and re-run here independently:
#
#   A  find /models/ov -name '*.xml' -size +100k        -> 52 IRs
#   B  find /models/ov -name '*.xml'   (no size filter) -> 172 IRs
#   C  of B, carrying a <layer type="...moe..."> (case-insensitive) -> 0
#   D  of B, carrying the string "moe" ANYWHERE, names included     -> 0
#
# C is the right method because the C++ test is
# `lowercase(get_type_name()).find("moe")`, and in an IR an op's type name is
# exactly the `<layer ... type="...">` attribute. D is the widest form the
# claim could take and it also holds. The negative is stronger than 198b736
# stated it: 0 of 172, not 0 of 52.
#
# RE-MEASURED A THIRD TIME, 2026-09-13, independently, when a brief named this
# gap a hard blocker: A -> 52, B -> 172, C -> 0, D -> 0. Same four figures, same
# commands. The zero is now three readings old and has never moved.
FLEET_IRS_SIZE_FILTERED = 52          # population A, 198b736's own
FLEET_IRS_ALL = 172                   # population B, the wider one
FLEET_IRS_WITH_MOE_TYPED_OP = 0       # C, and D too


@pytest.fixture(scope="module")
def built():
    arena = ss.SparseArena()
    model, report = ss.build_serving_shape_ir(
        seq_len=_T, arena=arena, n_layers=_CONTRACT_LAYERS)
    yield model, report, arena
    arena.close()


# ---------------------------------------------------------------------------
# MET -- the port contract, tensor for tensor name
# ---------------------------------------------------------------------------

def test_the_input_ports_are_the_names_and_shapes_the_serving_path_feeds(built):
    """Names, shapes and element types, exactly.

    `ngram_row_ids` is [1, T, 16] i64. 16 is not a choice: it is
    `HashParams::num_ngram_heads() = (ngram_size - 1) * heads_per_ngram`
    (src/exec/ngram_row_ids.h:59), which the same file's header states at :21
    as "16 on Qwen3.8: 8 x 2-gram + 8 x 3-gram", and each head gathers one
    160-wide row (:22). `position_ids` is the name backend_ov.cpp:98 declares
    (`kPositionIds`). `conv_mask` is the port q4e.backbone already declares
    (backbone.py:105-106).
    """
    model, report, _ = built
    cfg = pwe.real_config()
    Hn = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    assert Hn == 16, f"num_ngram_heads moved to {Hn}; ngram_row_ids' 16 is derived"

    got = {name: (tuple(shape), etype)
           for name, shape, etype in report["inputs"]}
    want = {
        "input_ids":     ((1, _T), "int64_t"),
        "position_ids":  ((1, _T), "int64_t"),
        "ngram_row_ids": ((1, _T, Hn), "int64_t"),
        "conv_mask":     ((1, _T), "float32"),
        # Declared for the transformation, which looks them up by name and
        # removes them; -1 is a dynamic dimension (`_dims`). `attention_mask`
        # spans past + current, so its length is not the query block's.
        "attention_mask": ((1, -1), "int64_t"),
        "beam_idx":       ((-1,), "int32_t"),
    }
    print("\n[contract-ports] inputs:")
    for k in sorted(got):
        print(f"  {k:16s} {got[k][0]}  {got[k][1]}")
    for name, (shape, et) in want.items():
        assert name in got, f"input port {name!r} missing; got {sorted(got)}"
        assert got[name][0] == shape, f"{name}: shape {got[name][0]} != {shape}"
        assert et in got[name][1], f"{name}: type {got[name][1]} != {et}"
    assert set(got) == set(want), (
        f"unexpected input ports: {sorted(set(got) - set(want))}")


def test_the_output_is_logits_at_the_real_vocabulary(built):
    model, report, _ = built
    cfg = pwe.real_config()
    outs = {n: (tuple(s), t) for n, s, t in report["outputs"]}
    print(f"[contract-ports] outputs: {outs}")
    assert list(outs) == ["logits"], outs
    assert outs["logits"][0] == (1, _T, cfg.vocab_size), outs["logits"]
    assert "float32" in outs["logits"][1]


def test_the_layer_kinds_follow_the_checkpoint_not_a_convenience(built):
    """12 of 48 blocks are full-attention at layer_idx % 4 == 3
    (piecewise_export.REAL_GEOMETRY:154-156). At 8 layers that is 6 GDN + 2."""
    _, report, _ = built
    print(f"[contract-layers] gdn={report['gdn_layers']} "
          f"attn={report['attn_layers']} of {report['n_layers']}")
    assert report["attn_layers"] == _CONTRACT_LAYERS // 4
    assert report["gdn_layers"] == _CONTRACT_LAYERS - report["attn_layers"]


# ---------------------------------------------------------------------------
# MET -- the expert bodies are declared, slot-referenced, never materialised
# ---------------------------------------------------------------------------

def _expert_constants(model, num_expert):
    """Every Constant whose leading dimension is the expert count -- the same
    selector backend_ov.cpp:600 uses, applied without the type-name gate."""
    out = []
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Constant":
            continue
        sh = list(node.get_output_shape(0))
        if sh and sh[0] == num_expert:
            out.append((node.get_friendly_name(), tuple(sh),
                        node.get_output_element_type(0)))
    return out


def test_expert_bodies_are_rank4_u4_and_one_gate_up_down_triple_per_layer(built):
    """The tiled lowering's weight shape, at real geometry.

    verify_moe_lowering.py:29-31: "expert weights as rank-4
    [E,out,groups,group_size] Constants in a compressed integer type
    (u4/i4/u8/i8)". Three per MoE layer -- gate, up, down -- plus their
    zero-points, which carry the same leading dimension by construction.
    """
    model, report, _ = built
    cfg = pwe.real_config()
    E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    gs = ss.EXPERT_GROUP_SIZE
    consts = _expert_constants(model, E)
    weights = [c for c in consts if c[0].endswith("/weight_u4")]
    zps = [c for c in consts if c[0].endswith("/zero_point")]

    print(f"\n[contract-experts] E={E} I={I} H={H} group_size={gs}")
    print(f"  expert weight constants {len(weights)}, zero-points {len(zps)}, "
          f"layers {report['n_layers']}")
    for name, sh, et in weights[:3]:
        print(f"  {name:34s} {sh}  {et}")

    assert len(weights) == 3 * report["n_layers"], (
        f"{len(weights)} expert weight constants for {report['n_layers']} "
        f"layers; expected gate/up/down per layer")
    assert len(zps) == len(weights), "every expert weight needs its zero-point"

    want = {(E, I, H // gs, gs), (E, H, I // gs, gs)}
    for name, sh, et in weights:
        assert len(sh) == 4, f"{name}: rank {len(sh)}, the tiled pass wants 4"
        assert sh in want, f"{name}: {sh} not in {want}"
        assert et == ss.EXPERT_DECLARED_TYPE, f"{name}: {et}"


def test_the_dequant_chain_is_convert_subtract_multiply_reshape(built):
    """The chain the fusing pass matches, INCLUDING the trailing Reshape --
    verify_moe_lowering.py:33-42 records a real GPU compile crashing inside the
    pass's own rewrite when that Reshape was absent."""
    model, _, _ = built
    cfg = pwe.real_config()
    E = cfg.num_experts
    reshapes = [n for n in model.get_ordered_ops()
                if n.get_type_name() == "Reshape"
                and n.get_friendly_name().endswith("/dequant_reshape")]
    assert reshapes, "no dequant Reshape found; the matcher anchors on it"
    walked = 0
    for r in reshapes:
        mul = r.input_value(0).get_node()
        assert mul.get_type_name() == "Multiply", \
            f"{r.get_friendly_name()}: parent is {mul.get_type_name()}, want Multiply"
        sub = mul.input_value(0).get_node()
        assert sub.get_type_name() == "Subtract", \
            f"{r.get_friendly_name()}: want Subtract, got {sub.get_type_name()}"
        conv = sub.input_value(0).get_node()
        assert conv.get_type_name() == "Convert", \
            f"{r.get_friendly_name()}: want Convert, got {conv.get_type_name()}"
        const = conv.input_value(0).get_node()
        assert const.get_type_name() == "Constant"
        assert list(const.get_output_shape(0))[0] == E
        assert const.get_output_element_type(0) == ss.EXPERT_DECLARED_TYPE
        # rank 4 in, rank 3 out -- the collapse the pass looks for
        assert len(list(const.get_output_shape(0))) == 4
        assert len(list(r.get_output_shape(0))) == 3
        walked += 1
    print(f"[contract-dequant] walked {walked} Convert->Subtract->Multiply->"
          f"Reshape(4->3) chains, all anchored on a u4 [E,...] Constant")


def test_no_expert_constant_is_materialised(built):
    """The arena declares tens of GiB of constants and occupies zero blocks on
    disk, because no page is ever written.

    WHAT THIS CELL DOES AND DOES NOT PROVE -- corrected 2026-09-12, while
    accepting the fill. `disk_kib()` is NOT a discriminator on the dev host's
    filesystem: ZFS allocates on transaction-group commit rather than on
    msync, and stores an all-zero record as a hole. Measured, three rows, one
    4 GiB sparse file each: nothing written -> 512 B; 512 MiB of zeros -> 512
    B; 512 MiB of RANDOM bytes -> 512 B after msync and only 439,174,656 B
    after a system `sync` and twelve seconds. So `disk_kib <= 64` is true of
    this build, and would be equally true of one that had materialised every
    constant as zeros, and of one that had just written real weights.

    It is kept because it is cheap and it is one more sign, and because on a
    filesystem that accounts blocks eagerly it does discriminate. The GUARD is
    `test_the_full_48_layer_stack_emits_at_real_geometry`'s peak RSS
    (CF-RESIDENT) -- which is exactly the finding that cell exists for, one
    level further down than the reviewer found it.
    """
    _, report, arena = built
    declared = report["arena_declared_bytes"]
    disk_kib = arena.disk_kib()
    print(f"\n[contract-residency] declared {declared / 2**30:.2f} GiB of "
          f"constant storage, {disk_kib} KiB actually on disk")
    assert declared > 8 * 2**30, (
        f"only {declared / 2**30:.2f} GiB declared at {_CONTRACT_LAYERS} real "
        f"layers -- the geometry is not real")
    assert disk_kib <= 64, (
        f"{disk_kib} KiB on disk: something WROTE to the arena, so the "
        f"constants are materialising after all. NB the converse does not "
        f"follow -- see this cell's docstring; 0 KiB is not proof of an "
        f"unwritten arena on a copy-on-write filesystem.")


# ---------------------------------------------------------------------------
# THE HANDSHAKE with the C++ slot-pool arithmetic
# ---------------------------------------------------------------------------

def slot_pool_from_tiled_ir(model, num_expert, ratio_pct):
    """What `slot_pool_from_ir` would find if it matched the PATTERN arcint
    exports instead of a type name: the Constants with leading dim
    `num_expert` that feed a dequant chain, grouped per MoE layer.

    Same per-expert arithmetic as backend_ov.cpp:600-604 (product of dims[1:]
    times the CEILED element size) and the same slot ceiling as fit.h:95.
    """
    per_layer = {}
    for node in model.get_ordered_ops():
        if node.get_type_name() != "Reshape":
            continue
        fname = node.get_friendly_name()
        if not fname.endswith("/dequant_reshape"):
            continue
        layer = fname.split("/")[0]
        const = node.input_value(0).get_node()          # Multiply
        const = const.input_value(0).get_node()         # Subtract
        const = const.input_value(0).get_node()         # Convert
        const = const.input_value(0).get_node()         # Constant
        sh = list(const.get_output_shape(0))
        if not sh or sh[0] != num_expert:
            continue
        elems = 1
        for d in sh[1:]:
            elems *= d
        et = const.get_output_element_type(0)
        per_layer[layer] = per_layer.get(layer, 0) + elems * ((et.bitwidth + 7) // 8)
    if not per_layer:
        return None
    first = per_layer[sorted(per_layer)[0]]
    total = sum(ss._expert_slot_bytes(num_expert, ratio_pct, b, 1)
                for b in per_layer.values())
    return {"total_bytes": total, "per_expert_bytes": first,
            "slots": ss._expert_slot_bytes(num_expert, ratio_pct, 1, 1),
            "moe_layers": len(per_layer)}


def test_the_cpp_type_name_matcher_finds_nothing_and_the_line_is_named(built):
    """THE HANDSHAKE FAILURE, named exactly.

    `slot_pool_from_ir`'s gate is backend_ov.cpp:585

        if (tname.find("moe") == std::string::npos) continue;

    and no arcint-exported IR carries a moe-typed op -- not this one, and not
    any of the 52 IRs in the dev host's model store, including the 35B-A3B MoE
    language model that `moe_block_tiled` was extracted from. The fusion that
    creates such a node is a GPU-plugin COMPILE-time pass; this function runs
    on `read_model`.

    WHY NO EMITTER CAN CLOSE THIS, measured in the plugin source 2026-09-13
    rather than inferred, because a brief proposed closing it from the export
    side as a hard blocker:

      * the op types whose names contain "moe" are `MoeOp` and
        `MoeOpWithRouting`, and they are PRODUCED, not read. The chain is named
        in one line of the plugin's own pipeline --
        `transformations_pipeline.cpp:644`: "MOE: TiledMoeBlock ->
        GatherMatmuls(compressed) -> MoeOp(compressed) ->
        MoeOpWithRouting(compressed)" -- registered at :654 as
        `ov::pass::ConvertTiledMoeBlockToGatherMatmuls`, inside
        `compile_model`.
      * `TiledMoeBlock` IS NOT AN OP. `grep -rn 'OPENVINO_OP("TiledMoeBlock'`
        over the whole plugin tree returns nothing: it is the name of a
        PATTERN (the tiled dequant chain this file already asserts --
        Convert/Subtract/Multiply/Reshape/MatMul), matched by a `MatcherPass`.
        There is no class in any opset for an exporter to instantiate.

    So the demand at :585 is not a contract an exporter can satisfy in
    principle, and emitting something to satisfy it would mean inventing a type
    name for a log line. The route that CAN work is the pattern matcher
    (`slot_pool_from_tiled_ir`, below, and the C++ change it implies), and it
    is worth exactly what the nullopt path costs -- which is nothing:

    Asserted, not lamented: the analytic route is nullopt here, so
    backend_ov.cpp:3746+ config.json fallback is what prices the host ledger.
    THAT LEDGER LINE IS INFORMATIONAL. backend_ov.cpp:3729-3734 says so in the
    source -- "Host-side ledger (GTT): informational, never charged against the
    device budget" -- and the device term is priced by the plateau probe
    (:3631+), with the analytic figure used only if the probe throws. Nothing
    in the load path refuses an IR for lacking a moe-typed op: `grep -n
    'throw\\|log::error' src/exec/backend_ov.cpp | grep -iE 'moe|expert'` is
    empty. A missing moe-typed op is a `source: config` log line, not a gate.
    """
    model, _, _ = built
    cfg = pwe.real_config()
    got = ss.slot_pool_from_ir(model, cfg.num_experts, 0)
    typed = sorted({n.get_type_name() for n in model.get_ordered_ops()
                    if "moe" in n.get_type_name().lower()})
    print(f"\n[contract-otd] moe-typed ops in the serving-shape IR: {typed}")
    print(f"[contract-otd] slot_pool_from_ir(backend_ov.cpp:577) -> {got}")
    print(f"[contract-otd] dev-host model store, 2026-09-12: "
          f"{FLEET_IRS_WITH_MOE_TYPED_OP} of {FLEET_IRS_ALL} IRs carry one "
          f"({FLEET_IRS_SIZE_FILTERED} of them over 100k)")
    assert typed == [], (
        "an op type now contains 'moe'; slot_pool_from_ir may match -- "
        "re-derive this cell instead of editing it")
    assert got is None, (
        "slot_pool_from_ir matched. That is GOOD NEWS and this cell must be "
        "rewritten to assert the figures rather than the gap.")
    assert FLEET_IRS_WITH_MOE_TYPED_OP == 0


def test_the_pattern_matcher_prices_the_expert_pool_and_lands_on_the_cpp_constant(built):
    """The arithmetic the contract WOULD produce, cross-checked against a
    number the C++ already holds.

    `src/exec/flash_next_offload.h:45`:
        constexpr uint64_t kFlashNextSliceBytes = 2'457'600;
        // one expert-layer int4 slice (gate/up/down)

    gate+up+down at real geometry = 2*(640*2560) + 2560*640 = 4,915,200 int4
    values = 2,457,600 bytes. The IR walk cannot reproduce that figure, and the
    reason is structural rather than a bug in either side:
    backend_ov.cpp:604 uses `element_type().size()`, which CEILS a 4-bit width
    to one whole byte -- so it reads 4,915,200 B per expert, EXACTLY 2x. The
    C++ comment at :610-615 anticipates over-reservation ("this over-reserves
    rather than under-reserves, pending an on-card audit"); this cell measures
    the factor and pins it at exactly 2.
    """
    model, report, _ = built
    cfg = pwe.real_config()
    E = cfg.num_experts
    got = slot_pool_from_tiled_ir(model, E, 0)
    assert got is not None, "the pattern matcher found no MoE layer either"

    slice_bytes_cpp = 2_457_600            # flash_next_offload.h:45
    true_bytes = (2 * cfg.moe_intermediate_size * cfg.hidden_size
                  + cfg.hidden_size * cfg.moe_intermediate_size) // 2
    print(f"\n[contract-slots] moe_layers {got['moe_layers']} "
          f"slots/layer {got['slots']} (ratio 0)")
    print(f"[contract-slots] per-expert bytes: IR walk {got['per_expert_bytes']:,} "
          f"| true int4 {true_bytes:,} | flash_next_offload.h:45 "
          f"{slice_bytes_cpp:,}")
    print(f"[contract-slots] total {got['total_bytes'] / 2**30:.2f} GiB over "
          f"{got['moe_layers']} layers")

    assert got["moe_layers"] == report["n_layers"], (
        f"{got['moe_layers']} MoE layers found, {report['n_layers']} emitted")
    assert true_bytes == slice_bytes_cpp, (
        f"the geometry no longer gives the C++ constant: {true_bytes} != "
        f"{slice_bytes_cpp} (flash_next_offload.h:45)")
    assert got["per_expert_bytes"] == 2 * slice_bytes_cpp, (
        f"the u4 ceiling factor moved: {got['per_expert_bytes']} is "
        f"{got['per_expert_bytes'] / slice_bytes_cpp:.3f}x the C++ slice, "
        f"expected exactly 2 (element_type().size() ceils 4 bits to 1 byte)")
    assert got["slots"] == E, "ratio 0 must keep every expert resident"


def test_the_slot_arithmetic_transcription_matches_the_cpp_ceiling():
    """fit.h:95, `ceil(num_expert * (100 - ratio) / 100)` slots per layer.
    Checked at the boundaries a ceiling gets wrong."""
    cases = [(512, 0, 512), (512, 50, 256), (512, 100, 0),
             (512, 1, 507), (512, 99, 6), (10, 33, 7), (3, 50, 2)]
    for E, ratio, want in cases:
        got = ss._expert_slot_bytes(E, ratio, 1, 1)
        assert got == want, f"E={E} ratio={ratio}: {got} != {want}"


# ---------------------------------------------------------------------------
# LOADABLE BY ov.Core -- at a geometry whose .bin fits, and stated as such
# ---------------------------------------------------------------------------

def test_the_serving_shape_survives_save_and_read_back(tmp_path):
    """`ov.save_model` -> `ov.Core().read_model` round-trip, asserting the u4
    expert constants and the dequant chain come back intact.

    THE GEOMETRY IS REDUCED ON PURPOSE and the reason is arithmetic, not
    convenience: `ov.save_model(model, path, compress_to_fp16)` is the only
    serialisation this OpenVINO build's Python API offers, and it writes the
    whole .bin. At real geometry that is 183 GiB against 27 GiB free on the dev
    host, so the full-geometry model is validated as a live ov::Model (which is
    OpenVINO's own shape inference accepting every op as it is built) and NOT
    round-tripped. A weightless / `ov::weights_path` write -- the form
    backend_ov.cpp:552-556 says the load path can consume -- has no Python
    entry point here; that is recorded as blocked, with the reason, rather than
    claimed.

    What this cell does prove is that the SHAPE serialises and re-reads: same
    ports, same u4 element types, same rank-4 -> rank-3 dequant collapse.
    """
    cfg = pwe.real_config()
    # one MoE layer's worth of structure at a width whose .bin is a few MiB
    small = type(cfg)(
        hidden_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        hc_count=cfg.hc_count, hc_lowrank=32,
        ple_embed_dim=256, ple_conv_kernel_size=cfg.ple_conv_kernel_size,
        ngram_size=cfg.ngram_size, heads_per_ngram=cfg.heads_per_ngram,
        vocab_size=512, rms_norm_eps=cfg.rms_norm_eps,
        linear_key_head_dim=32, linear_num_key_heads=2,
        linear_value_head_dim=32, linear_num_value_heads=4,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        hidden_act="silu", layer_types=["linear_attention"],
    )
    small.ngram_total_vocab = 4096
    arena = ss.SparseArena(capacity_bytes=1 << 32)
    try:
        model, report = ss.build_serving_shape_ir(
            config=small, seq_len=4, arena=arena, n_layers=1)
        xml = tmp_path / "serving_shape.xml"
        ov.save_model(model, str(xml), compress_to_fp16=False)
        size_mib = (xml.stat().st_size
                    + xml.with_suffix(".bin").stat().st_size) / 2**20
        back = ov.Core().read_model(str(xml))
        names_before = {n for n, _, _ in report["inputs"]}
        names_after = {p.get_node().get_friendly_name() for p in back.inputs}
        u4_before = sum(1 for n in model.get_ordered_ops()
                        if n.get_type_name() == "Constant"
                        and n.get_output_element_type(0) == ov.Type.u4)
        u4_after = sum(1 for n in back.get_ordered_ops()
                       if n.get_type_name() == "Constant"
                       and n.get_output_element_type(0) == ov.Type.u4)
        print(f"\n[contract-roundtrip] reduced geometry E={small.num_experts} "
              f"H={small.hidden_size} I={small.moe_intermediate_size}, "
              f"xml+bin {size_mib:.2f} MiB")
        print(f"  ports  before {sorted(names_before)}")
        print(f"  ports  after  {sorted(names_after)}")
        print(f"  u4 constants  before {u4_before}  after {u4_after}")
        assert names_after == names_before, (names_before, names_after)
        assert u4_before > 0 and u4_after == u4_before, (u4_before, u4_after)
        assert list(back.outputs[0].get_shape()) == [1, 4, small.vocab_size]
    finally:
        arena.close()


# ---------------------------------------------------------------------------
# NOT MET AND NAMED -- the paged serving ports
# ---------------------------------------------------------------------------

_BACKEND_OV = REPO_ROOT / "src" / "exec" / "backend_ov.cpp"


def cite(anchor, path=None):
    """`<file>:<line>` for the ONE line of `path` that contains `anchor`.

    CF-COUNTS (REVIEW 2a45349 F4). The table below used to write its line
    numbers out by hand, and three of its rows had drifted by one line:
    `gated_delta_state_table.` was written :3196 where the prefix test is at
    :3195, and `key_cache.` / `value_cache.` were both written :3200 where it
    is at :3199 (:3200 is the `is_value` line underneath). Nothing was wrong
    with the ports; the citations rotted because a line was inserted above
    them, in a repository whose standard is grep-verified citations.

    Hand-correcting them would be right until the next edit above them. This
    resolves an ANCHOR -- a substring of the cited code, which is the thing
    actually meant -- and derives the number, so the citation cannot drift at
    all. The PROSE-CLAIM LAW's clause 2 ("counts are generated by their
    defining gate, never recited") applied to line numbers.

    Uniqueness is asserted, not hoped for: an anchor that matches two lines
    names neither, and one that matches none has been edited away.
    """
    p = path or _BACKEND_OV
    hits = [i for i, line in enumerate(p.read_text().splitlines(), 1)
            if anchor in line]
    assert len(hits) == 1, (
        f"citation anchor {anchor!r} matches {len(hits)} lines of {p.name}"
        + (f" ({hits[:6]})" if hits else " -- it has been edited away")
        + ". An anchor must identify exactly one line or it cites nothing.")
    return f"{p.name}:{hits[0]}"


# THE PORTS TABLE -- ONE RECORDED TABLE, AND BOTH CELLS BELOW READ IT.
#
# Every fact about the paged gap lives here and nowhere else in this suite: the
# port, whether the emitter produces it yet, which of the two C++ sites names
# it, and WHICH CONSTRUCT of the stateful graph the transformation turns into
# it. A port landing is ONE edit to the `status` column plus that port's live
# cells; the inventory cell and the umbrella xfail follow by construction and
# neither can rot against the other. The count of rows is never written down --
# every count in this section is `len()` of some slice of this table -- so the
# count-drift class that cost REVIEW 2a45349 F4 three wrong line numbers cannot
# recur in the other direction either.
ABSENT, PRESENT = "absent", "present"

# The `produced by` column, named once each so the rows stay readable. These
# are not descriptions: they are what the pass was MEASURED to do, below.
_BY_SDPA = ("ScaledDotProductAttention over a rank-4 KV Variable"
            " -> PagedAttentionExtension")
_BY_CONV = "a rank-3 Variable (GDN short-conv state) -> PagedCausalConv1D"
# READ OUT OF THE PASS (fuse_gated_delta_net.cpp + paged_gated_delta_net_
# fusion.cpp, pinned commit). Unlike the other two this is a TWO-STAGE chain
# and the first stage is not a transcription: PagedGatedDeltaNetFusion matches
# an `ov::op::internal::GatedDeltaNet` NODE, which only exists because
# FuseGDNLoop has already fused a v5::Loop -- one whose body is the
# TOKEN-SEQUENTIAL delta rule, one timestep per iteration -- into it. The
# emitter writes the CHUNKED delta rule, which is a different computation and
# not a near-miss of this pattern.
_BY_GDN = ("a rank-4 Variable behind a token-sequential v5::Loop"
           " -> FuseGDNLoop -> GatedDeltaNet -> PagedGatedDeltaNet")
_BY_PA_INDEX = "PagedAttentionExtension's own index ports"
# READ OUT OF THE PASS, not inferred from which ports appeared together. Both
# linear-attention fusions call `pa_params.add` for all four of these and for
# `subsequence_begins`: paged_causal_conv1d_fusion.cpp and
# paged_gated_delta_net_fusion.cpp, at the pinned OpenVINO commit. So EITHER
# construct alone declares the whole `la.*` group -- this file first recorded
# the attribution as unseparated, and it is separated now: the conv construct
# on its own was measured producing all four.
_BY_LA_INDEX = ("declared by EITHER linear-attention fusion"
                " (PagedCausalConv1D / PagedGatedDeltaNet), both add all four")

# port prefix | status | site | anchor: the CODE that classifies or feeds it |
# produced by | how many of it the served IR carries (census below).
# The file:line in every printed inventory is derived from the anchor by
# `cite`, never typed.
_PAGED_PORT_TABLE = (
    # classified by name prefix at load time
    ("conv_state_table.", PRESENT, "classify",
     'name.rfind("conv_state_table.", 0) == 0', _BY_CONV, 48),
    ("gated_delta_state_table.", PRESENT, "classify",
     'name.rfind("gated_delta_state_table.", 0) == 0', _BY_GDN, 48),
    ("key_cache.", PRESENT, "classify",
     'name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0',
     _BY_SDPA, 16),
    ("value_cache.", PRESENT, "classify",
     'name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0',
     _BY_SDPA, 16),
    # fed every forward
    ("past_lens", PRESENT, "feed",
     'set_i32("past_lens"', _BY_PA_INDEX, 1),
    ("subsequence_begins", PRESENT, "feed",
     'set_i32("subsequence_begins"', _BY_PA_INDEX, 1),
    ("block_indices", PRESENT, "feed",
     'set_i32("block_indices"', _BY_PA_INDEX, 1),
    ("block_indices_begins", PRESENT, "feed",
     'set_i32("block_indices_begins"', _BY_PA_INDEX, 1),
    ("max_context_len", PRESENT, "feed",
     'set_i32("max_context_len"', _BY_PA_INDEX, 1),
    ("la.block_indices", PRESENT, "feed",
     'set_i32("la.block_indices"', _BY_LA_INDEX, 1),
    ("la.block_indices_begins", PRESENT, "feed",
     'set_i32("la.block_indices_begins"', _BY_LA_INDEX, 1),
    ("la.past_lens", PRESENT, "feed",
     'set_i32("la.past_lens"', _BY_LA_INDEX, 1),
    ("la.cache_interval", PRESENT, "feed",
     'set_i32("la.cache_interval"', _BY_LA_INDEX, 1),
)

# The same rows with the citation resolved. `cite` asserts at import that each
# anchor still identifies exactly one line of the C++, so a table row that has
# lost its code cannot survive collection.
_PAGED_PORTS = [
    {"port": port, "status": status, "site": site, "cite": cite(anchor),
     "produced_by": by, "served_count": served}
    for port, status, site, anchor, by, served in _PAGED_PORT_TABLE
]
_PAGED_ABSENT = [r for r in _PAGED_PORTS if r["status"] == ABSENT]
_PAGED_PRESENT = [r for r in _PAGED_PORTS if r["status"] == PRESENT]


def _matches(port, name):
    """Does one parameter name satisfy one table row?

    A row whose port ends in `.` is a FAMILY -- `key_cache.` stands for
    `key_cache.0`, `key_cache.1`, one per layer -- and the C++ classifies those
    by prefix (`name.rfind("key_cache.", 0) == 0`). Every other row is a single
    whole name, fed by `set_tensor` with that exact string, and prefix-matching
    one of those is a bug this cell already paid for: `block_indices` is a
    prefix of `block_indices_begins`, so a prefix rule counts the latter twice
    and would let a graph declaring only `block_indices_begins` claim both.
    """
    return name.startswith(port) if port.endswith(".") else name == port


def _declares(port, names):
    return any(_matches(port, n) for n in names)


def test_the_paged_port_citations_resolve_to_the_code_they_name():
    """The anchors in the table must each still identify exactly one line --
    `cite` asserts that as it builds `_PAGED_PORTS`, so this cell documents the
    result and prints it. It also pins the fact the inventory depends on: the
    table's two sites are the C++'s two sites, and each row sits in the one it
    claims."""
    print("\n[contract-cite] paged-port citations, resolved from anchors:")
    for r in _PAGED_PORTS:
        print(f"  {r['port']:28s} {r['cite']:22s} {r['status']:8s} {r['site']}")
    # The split is checked by READING the C++ line each row resolved to rather
    # than by counting the table's own `site` column against itself: the column
    # is the claim, the line number is the evidence, and a row that drifts from
    # one site to the other has to be noticed here.
    by_site = {}
    for r in _PAGED_PORTS:
        by_site.setdefault(r["site"], []).append(int(r["cite"].split(":")[1]))
    classify, feed = sorted(by_site["classify"]), sorted(by_site["feed"])
    print(f"[contract-cite] {len(classify)} classified at "
          f"{min(classify)}-{max(classify)}, {len(feed)} fed at "
          f"{min(feed)}-{max(feed)}")
    assert max(classify) < min(feed), (
        f"a 'classify' row resolved below a 'feed' row (classify "
        f"{classify}, feed {feed}); the inventory's two sites moved and the "
        f"table's site column is no longer the document's")


# WHY THESE PORTS ARE THE BOOT, read out of the C++ 2026-09-13, because the
# question "can a shallow boot run before the ports exist" is worth a definite
# answer: there is no non-paged forward to boot into.
#
#   * the backend compiles the served IR as `paged_model_`
#     (backend_ov.cpp:2965) and every lane's request comes from it (:3019);
#   * the forward at :6141-6151 sets the table's nine "feed" rows
#     UNCONDITIONALLY -- `set_tensor` calls with no branch, which is what those
#     rows' citations resolve to;
#   * `ov::InferRequest::set_tensor` on a name the compiled model does not
#     declare throws. A static full-sequence IR therefore cannot be served by
#     this path at any depth, shallow included.
#
# So the ports are not a refinement of a working boot; they ARE the boot. That
# is a fact about the C++, and it is the reason the MoE-handshake gap
# (`test_the_cpp_type_name_matcher_finds_nothing_and_the_line_is_named`) was
# withdrawn as a blocker: it changes a log line, these change whether a forward
# can be issued at all.
#
# ---------------------------------------------------------------------------
# HOW A PORT IS MADE -- measured 2026-09-13, and it corrects this file
# ---------------------------------------------------------------------------
#
# This section used to test `report["inputs"]` -- the parameters of the model
# the emitter returns -- against the table, and the xfail's own reason said the
# ports are "NOT emitted", as if emitting them were the export side's job. THAT
# WAS WRONG ABOUT WHO MAKES THEM, and the error was load-bearing: it pointed a
# whole item's work at hand-declaring a list of parameters.
#
# `load_paged` reads the artifact and then runs a pass over it before compiling
# (backend_ov.cpp:2574, `ov::pass::SDPAToPagedAttention`). EVERY PORT IN THE
# TABLE IS THAT PASS'S OUTPUT. Measured on the real served artifact, dev host,
# 2026-09-13, OV 2026.4.0-22849 -- `read_model`, then the same transformation,
# device-free, no compile and no card:
#
#     openvino_language_model.xml of the served hybrid agent model
#     BEFORE: 4 parameters -- attention_mask [?,?] i64, inputs_embeds
#             [?,?,5120] f32, position_ids [4,?,?] i64, beam_idx [?] i32
#             128 Variables: 48 rank-3 [?,10240,4]   (GDN short conv)
#                            48 rank-4 [?,48,128,128] (GDN recurrent state)
#                            32 rank-4 [?,4,?,256]    (attention K and V)
#             ops: ReadValue 128, Assign 128, ScaledDotProductAttention 16
#     AFTER : 139 parameters -- conv_state_table.N x48,
#             gated_delta_state_table.N x48, key_cache.N x16, value_cache.N x16,
#             and past_lens / subsequence_begins / block_indices /
#             block_indices_begins / max_context_len / la.block_indices /
#             la.block_indices_begins / la.past_lens / la.cache_interval
#             ops: PagedAttentionExtension 16, PagedCausalConv1D 48,
#                  PagedGatedDeltaNet 48
#
# That is the `produced by` column of the table above, and the `served_count`
# column is the multiplicity in that same reading. The rank-3/rank-4 split is
# the one `load_paged` itself reads off the STATEFUL graph at :2557-2569, which
# is why it must be read before the pass runs -- the transformed ports leave
# those dims dynamic.
#
# The converse, on this emitter's own output, same day and same method -- THE
# READING THAT OPENED THIS ITEM, kept as it was taken:
#
#     serving-shape IR, 8 layers, T=8
#     parameters: input_ids, position_ids, ngram_row_ids, conv_mask
#     Variables : 0        convertible ops: none
#     the pass REFUSES, by name:
#       RuntimeError: Check '!model->get_variables().empty()' failed at
#       src/core/src/pass/sdpa_to_paged_attention.cpp:75
#
# So the gap was not a set of missing parameters. It was that the IR carried
# NONE OF THE THREE CONSTRUCTS the pass converts, and the pass said so by name
# before looking at anything else. `test_the_paged_gap_is_inventoried_precisely`
# below asserts THE TABLE against what the pass actually produces, and prints
# that refusal verbatim on any tree where it still stands.
#
# THE SAME READING, after the full-attention layers became stateful (the first
# of the three constructs, `q4e.serving_shape.emit_stateful_attention`):
#
#     serving-shape IR, 8 layers, T=8
#     parameters: input_ids, position_ids, ngram_row_ids, conv_mask,
#                 attention_mask, beam_idx
#     Variables : 4        ReadValue 4, Assign 4, ScaledDotProductAttention 2
#     AFTER     : + max_context_len, past_lens, subsequence_begins,
#                 block_indices, block_indices_begins, key_cache.0/1,
#                 value_cache.0/1;  PagedAttentionExtension 2
#                 attention_mask and beam_idx are GONE -- consumed by the pass
#
# Two of the eight layers are full-attention (index % 4 == 3), which is why the
# caches number two. THAT READING IS DATED: at the time it was taken the GDN
# layers were untouched and the two state tables and four la.* ports were
# absent. Both GDN constructs landed later the same day and the table's status
# column -- not this comment -- is where the current state lives.
#
# The xfail over all the rows was all-or-nothing, so it retired on the commit
# that landed the LAST port rather than the first; see the block above
# `test_the_paged_port_contract_is_satisfied`. The inventory cell reds on any
# port whose real state stops matching its `status` column, in either
# direction, and a port landing is one edit to that column in the same commit
# as the port.


@pytest.fixture(scope="module")
def paged_census():
    """The ports that exist after THE PASS ARCINT ITSELF RUNS.

    Its own build and its own arena: `paged_attention_transformation` mutates
    the model in place, and the `built` fixture is module-scoped and shared, so
    transforming that one would hand every later cell a different graph than it
    asked for.
    """
    arena = ss.SparseArena()
    try:
        model, _ = ss.build_serving_shape_ir(
            seq_len=_T, arena=arena, n_layers=_CONTRACT_LAYERS)
        before = {p.get_node().get_friendly_name() for p in model.inputs}
        # The variables AS THE LOAD PATH SEES THEM: read off the stateful graph
        # before the pass runs, which is what backend_ov.cpp:2557-2569 does and
        # for the same reason -- the transformed ports leave these dims dynamic.
        variables = []
        for var in model.get_variables():
            info = var.get_info()
            ps = info.data_shape
            variables.append({
                "id": info.variable_id,
                "dims": [d.get_length() if d.is_static else -1 for d in ps]
                        if ps.rank.is_static else None,
                "type": str(info.data_type)})
        sinks = len(model.get_sinks())
        refusal, after = None, before
        try:
            paged_attention_transformation(model)
        except Exception as exc:                                  # noqa: BLE001
            refusal = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}"
        else:
            after = {p.get_node().get_friendly_name() for p in model.inputs}
        hist = {}
        for node in model.get_ordered_ops():
            hist[node.get_type_name()] = hist.get(node.get_type_name(), 0) + 1
        paged_ops = {k: v for k, v in sorted(hist.items())
                     if k.startswith("Paged")}
        counts = {r["port"]: sum(1 for n in after if _matches(r["port"], n))
                  for r in _PAGED_PORTS}
        yield {"before": before, "after": after, "variables": variables,
               "sinks": sinks, "refusal": refusal, "paged_ops": paged_ops,
               "counts": counts}
    finally:
        arena.close()


# RETIRED 2026-09-13, which is the whole point of having written it strict.
# This cell carried `@pytest.mark.xfail(strict=True)` from the day the gap was
# named until the day the table's last row read `present`, at which point
# strict turned the pass into a FAILURE and the cell had to be looked at
# instead of forgotten. It was NOT relaxed and it did not move to
# `xfail(strict=False)`: the decorator is gone and the assertion it always
# carried is now a live one. The suite's xfail count drops by one with this
# commit; the other one is test_moe_block.py's and is untouched.
def test_the_paged_port_contract_is_satisfied(paged_census):
    """Every port in the table, produced by the pass the load path runs."""
    missing = [r for r in _PAGED_PORTS
               if not _declares(r["port"], paged_census["after"])]
    assert not missing, (
        "paged ports the transformation does not produce from this IR:\n"
        + "\n".join(f"  {r['port']:28s} fed at {r['cite']:22s} would come from "
                    f"{r['produced_by']}" for r in missing)
        + (f"\nthe pass refused: {paged_census['refusal']}"
           if paged_census["refusal"] else ""))
    assert not _PAGED_ABSENT, (
        f"the table still lists {len(_PAGED_ABSENT)} port(s) absent while the "
        f"pass produces all of them: {[r['port'] for r in _PAGED_ABSENT]}")


def test_the_paged_gap_is_inventoried_precisely(paged_census):
    """THE TABLE IS THE ASSERTION. Every row marked `present` must really be
    produced, every row marked `absent` must really be missing -- so the first
    port that lands reds this cell until its row is flipped, and a row flipped
    ahead of its port reds it too. The xfail above proves the whole gap; this
    cell is what keeps the table honest one row at a time."""
    produced = {r["port"]: _declares(r["port"], paged_census["after"])
                for r in _PAGED_PORTS}
    print(f"\n[contract-paged] the pass produced "
          f"{len(paged_census['after'] - paged_census['before'])} new "
          f"parameter(s) from {len(paged_census['variables'])} Variable(s); "
          f"paged ops: {paged_census['paged_ops'] or 'none'}")
    if paged_census["refusal"]:
        print(f"[contract-paged] the pass REFUSED: {paged_census['refusal']}")
    # `served_count` is the multiplicity the SERVED artifact carries, and this
    # is the one place it earns its column: printed beside what this 8-layer
    # build produces, so a reader sees the two populations side by side rather
    # than mistaking one for the other. It is deliberately NOT asserted -- the
    # served model is a different checkpoint with a different layer count.
    for r in _PAGED_PORTS:
        mark = "yes" if produced[r["port"]] else "no "
        print(f"  {r['port']:28s} table={r['status']:8s} produced={mark} "
              f"x{paged_census['counts'][r['port']]:<3d} "
              f"(served x{r['served_count']:<3d}) "
              f"{r['cite']:22s} {r['produced_by']}")
    print(f"[contract-paged] {len(_PAGED_PRESENT)} row(s) present, "
          f"{len(_PAGED_ABSENT)} absent")
    wrong = [(r, produced[r["port"]]) for r in _PAGED_PORTS
             if produced[r["port"]] != (r["status"] == PRESENT)]
    assert not wrong, (
        "the ports table disagrees with what the transformation produces:\n"
        + "\n".join(
            f"  {r['port']:28s} table says {r['status']}, pass "
            f"{'produces' if got else 'does not produce'} it"
            for r, got in wrong)
        + "\nedit the `status` column of _PAGED_PORT_TABLE in the same commit "
          "as the port, rather than either cell.")


# ---------------------------------------------------------------------------
# THE PORTS THAT HAVE LANDED -- one live cell group per construct
# ---------------------------------------------------------------------------
# The table says WHICH ports exist. These say the construct behind them is the
# one the serving path reads, at the geometry it reads it at. A port whose
# status flips to `present` without this much is a port that satisfies a name
# check and nothing else.

def test_the_kv_variables_are_the_shape_the_load_path_reads_prototypes_from(
        paged_census):
    """`load_paged` reads its KV/state prototypes off the STATEFUL graph
    (backend_ov.cpp:2557-2569): rank 4, leading dim replaced by 1, the tail
    static, the sequence dim NOT static -- that is the test it applies
    (`tail_static` over dims 1.. , and the attention KV is the case it excludes
    with "attention KV: dynamic seq dim"). Two variables per full-attention
    layer, key and value, and nothing else carries a variable yet."""
    cfg = pwe.real_config()
    kv, d = cfg.num_key_value_heads, cfg.head_dim
    attn_layers = _CONTRACT_LAYERS // 4
    print("\n[contract-kv] variables on the stateful graph:")
    for v in paged_census["variables"]:
        print(f"  {v['id']:34s} {v['dims']}  {v['type']}")
    # Rank alone no longer separates them: the GDN recurrent state is rank 4
    # too. `load_paged` separates them by the SEQUENCE DIM -- a rank-4 variable
    # whose tail is static is a state prototype, and the attention KV is the
    # case it excludes with "attention KV: dynamic seq dim" (:2557-2569). So
    # that is the split used here, which is reading the C++ rather than the ids.
    rank4 = [v for v in paged_census["variables"]
             if v["dims"] is not None and len(v["dims"]) == 4
             and any(d == -1 for d in v["dims"][1:])]
    assert len(rank4) == 2 * attn_layers, (
        f"{len(rank4)} rank-4 variable(s) for {attn_layers} full-attention "
        f"layer(s); expected one key and one value each")
    for v in rank4:
        assert v["dims"] == [-1, kv, -1, d], (
            f"{v['id']}: {v['dims']} is not [batch?, {kv}, seq?, {d}] -- "
            f"rank 4 with a dynamic sequence dim is what :2557-2569 reads")
        assert "f32" in v["type"] or "float32" in v["type"], v["type"]
    # read and written: a state that is never assigned is not state
    assert paged_census["sinks"] == len(paged_census["variables"]), (
        f"{paged_census['sinks']} Assign(s) for "
        f"{len(paged_census['variables'])} Variable(s)")


def test_one_paged_attention_op_and_one_cache_pair_per_full_attention_layer(
        paged_census):
    """The pass turns each stateful attention layer into exactly one
    PagedAttentionExtension with its own `key_cache.N` / `value_cache.N`. The
    count is the layer count, not a number written down here: a build that
    silently shared one cache across layers, or emitted a spare, fails."""
    attn_layers = _CONTRACT_LAYERS // 4
    ops = paged_census["paged_ops"]
    counts = paged_census["counts"]
    print(f"[contract-kv] paged ops {ops}; key_cache x{counts['key_cache.']}, "
          f"value_cache x{counts['value_cache.']}, "
          f"{attn_layers} full-attention layer(s)")
    assert ops.get("PagedAttentionExtension") == attn_layers, ops
    assert counts["key_cache."] == attn_layers, counts
    assert counts["value_cache."] == attn_layers, counts
    # the index ports are shared by every PagedAttention op: one each, always
    for r in _PAGED_PORTS:
        if r["produced_by"] is _BY_PA_INDEX:
            assert counts[r["port"]] == 1, (r["port"], counts[r["port"]])


def test_the_conv_state_is_one_table_per_gdn_layer_at_the_conv_geometry(
        paged_census):
    """The rank-3 short-conv Variable, one per GDN layer, at
    `[batch?, conv_dim, kernel]`.

    `conv_dim` is not a number written here: it is the checkpoint's own
    `2 * key_dim + value_dim` over the linear-attention geometry, and the
    fusion refuses the match unless the state's dim 1 equals the conv weights'
    dim 0 and its dim 2 the kernel width. So this cell asserts the shape the
    pass itself checks, at the geometry `piecewise_export.real_config` reads
    from the checkpoint.
    """
    cfg = pwe.real_config()
    conv_dim = (cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
                + cfg.linear_value_head_dim * cfg.linear_num_value_heads)
    K = cfg.linear_conv_kernel_dim
    gdn_layers = _CONTRACT_LAYERS - _CONTRACT_LAYERS // 4
    rank3 = [v for v in paged_census["variables"]
             if v["dims"] is not None and len(v["dims"]) == 3]
    print(f"\n[contract-conv] {len(rank3)} rank-3 variable(s) for "
          f"{gdn_layers} GDN layer(s); conv_dim {conv_dim}, kernel {K}")
    assert len(rank3) == gdn_layers, [v["id"] for v in rank3]
    for v in rank3:
        assert v["dims"] == [-1, conv_dim, K], (v["id"], v["dims"])
    assert paged_census["paged_ops"].get("PagedCausalConv1D") == gdn_layers, \
        paged_census["paged_ops"]
    assert paged_census["counts"]["conv_state_table."] == gdn_layers, \
        paged_census["counts"]


def test_the_recurrent_state_is_one_table_per_gdn_layer_at_the_head_geometry(
        paged_census):
    """The rank-4 recurrent Variable, one per GDN layer, at
    `[batch, v_heads, key_head_dim, value_head_dim]`.

    This is the shape `matches_linear_attention_loop` binds as
    `[?, head_num, k_head_size, v_head_size]` inside the Loop body, and the
    shape `load_paged` takes as a rank-4 prototype with a static tail
    (backend_ov.cpp:2557-2569). The attention KV Variables are rank 4 too and
    are NOT prototypes there -- their sequence dim is dynamic, which is the
    exclusion that same code writes as "attention KV: dynamic seq dim" -- so
    this cell checks that separation holds on the emitted graph rather than
    trusting the ids.
    """
    cfg = pwe.real_config()
    HV = cfg.linear_num_value_heads
    Dk, Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    gdn_layers = _CONTRACT_LAYERS - _CONTRACT_LAYERS // 4
    rank4 = [v for v in paged_census["variables"]
             if v["dims"] is not None and len(v["dims"]) == 4]
    # the load path's own split: a static tail is a state prototype, a dynamic
    # sequence dim is attention KV
    proto = [v for v in rank4 if all(d != -1 for d in v["dims"][1:])]
    kvish = [v for v in rank4 if any(d == -1 for d in v["dims"][1:])]
    print(f"\n[contract-gdn] {len(proto)} rank-4 state prototype(s), "
          f"{len(kvish)} rank-4 attention KV; want [1, {HV}, {Dk}, {Dv}]")
    assert len(proto) == gdn_layers, [v["id"] for v in proto]
    for v in proto:
        assert v["dims"] == [1, HV, Dk, Dv], (v["id"], v["dims"])
    assert len(kvish) == 2 * (_CONTRACT_LAYERS // 4), [v["id"] for v in kvish]
    assert paged_census["paged_ops"].get("PagedGatedDeltaNet") == gdn_layers, \
        paged_census["paged_ops"]
    assert paged_census["counts"]["gated_delta_state_table."] == gdn_layers, \
        paged_census["counts"]


def test_the_la_index_ports_come_with_the_linear_attention_conversion(
        paged_census):
    """All four `la.*` ports, exactly one of each, and the conversion that
    declares them is present. The pass adds them from EITHER linear-attention
    fusion, so they arrive with the first one that matches -- which is why they
    are here beside the conv state and not waiting on the recurrent one."""
    la = {r["port"]: paged_census["counts"][r["port"]]
          for r in _PAGED_PORTS if r["port"].startswith("la.")}
    print(f"[contract-conv] {la}")
    assert all(n == 1 for n in la.values()), la
    assert paged_census["paged_ops"].get("PagedCausalConv1D"), \
        "the la.* ports are here with no linear-attention conversion to " \
        "declare them; something else produced them and the table's " \
        "`produced_by` column is wrong"


# The two tensors the forward feeds that are NOT paged ports and so are not in
# the table above: they are not the transformation's output, they are the
# parameter surface it preserves. They are fed by the same unconditional
# `set_tensor` calls, immediately above the nine, so `set_tensor` on a name the
# compiled model does not declare throws for these exactly as it does for those.
_FORWARD_ALSO_FEEDS = tuple(
    (name, cite(anchor)) for name, anchor in (
        ("inputs_embeds", "lane.req.set_tensor(kInputsEmbeds, embeds);"),
        ("position_ids", "lane.req.set_tensor(kPositionIds, pos);"),
    ))


def test_the_converted_surface_against_every_tensor_the_forward_feeds(
        paged_census):
    """WHAT STILL STANDS BETWEEN THE CONVERTED IR AND A FORWARD, counted from
    both sides and printed, because the ports table answers only half of it.

    Feeding is unconditional in both directions' worth of trouble:

      * a name the forward feeds that the model does NOT declare throws at
        `set_tensor` -- that is the table's business, and the table says which
        are still missing.
      * a port the model declares that the forward NEVER feeds is the other
        half, and nothing was watching it. It does not throw; it is left at
        whatever the runtime allocates, which for `conv_mask` is a zero mask
        that annihilates the GDN input rather than passing it.

    This cell names the second set. It is expected to be NON-EMPTY today and
    the assertion is that it is EXACTLY the recorded set, so the day one is
    closed -- or a new one appears -- the cell fails and the list gets read
    again instead of drifting.
    """
    fed = {n for n, _ in _FORWARD_ALSO_FEEDS} | {
        r["port"] for r in _PAGED_PORTS if not r["port"].endswith(".")}
    families = tuple(r["port"] for r in _PAGED_PORTS if r["port"].endswith("."))
    declared = paged_census["after"]
    unfed = sorted(n for n in declared
                   if n not in fed and not n.startswith(families))

    # The recorded set, with WHY each one is here. Every entry is a gap, not a
    # decision: none of them is something the serving path knows how to feed.
    expected = {
        # the forward feeds `inputs_embeds` (the embedding is its own compiled
        # model); this emitter gathers the embedding inline off `input_ids`,
        # so the name the forward reaches for is not the name declared.
        "input_ids",
        # this emitter's own ports, which no serving forward has ever fed:
        # the hashed n-gram row ids and the GDN/PLE padding mask.
        "ngram_row_ids",
        "conv_mask",
    }
    print("\n[contract-feed] the forward feeds, unconditionally:")
    for name, c in _FORWARD_ALSO_FEEDS:
        print(f"  {name:26s} {c:22s} "
              f"{'declared' if name in declared else 'NOT DECLARED'}")
    print(f"[contract-feed] declared but never fed: {unfed}")
    assert set(unfed) == expected, (
        f"the never-fed set moved: {sorted(set(unfed) ^ expected)}. Every "
        f"name here is a port the compiled model carries that no forward "
        f"writes to -- read the list, do not widen it.")


def test_the_rope_table_only_spans_the_query_block(built):
    """A KNOWN LIMITATION, PINNED so it cannot be forgotten rather than
    asserted because it is desirable.

    `emit_stateful_attention` bakes the rope cos/sin for positions 0..T-1 and
    gathers them by `position_ids`. Every other part of that layer is built for
    `past > 0` -- the causal mask derives `past` from the KV Variable's dynamic
    length -- so the rope is the one place the static query block still leaks
    into correctness: a position at or past T indexes off the end of the table
    and OpenVINO's Gather does not throw for that.

    This cell asserts the table's extent IS the query block, which is the
    current state and not the desired one. The day it is sized from
    `max_position_embeddings` (~1.6 GiB over the 12 full-attention layers at
    real geometry, which is why it was not done in the increment that found
    it), this cell fails and the limitation gets promoted instead of forgotten
    -- the same mechanic the ports table's strict xfail used.
    """
    cfg = pwe.real_config()
    cos, sin = qattn._freqs_tables(cfg, _T)
    print(f"\n[contract-rope] cos/sin tables {cos.shape} for a query block of "
          f"{_T}; max_position_embeddings is {cfg.max_position_embeddings}")
    assert cos.shape[0] == _T and sin.shape[0] == _T, (
        f"the rope tables now span {cos.shape[0]} positions, not the query "
        f"block's {_T}. If they were sized from max_position_embeddings "
        f"({cfg.max_position_embeddings}) the limitation this cell pins is "
        f"GONE -- delete the cell, delete the paragraph in "
        f"emit_stateful_attention's docstring, and re-derive the residency "
        f"keystone, which this change moves.")
    assert cos.shape[0] < cfg.max_position_embeddings, (
        "the query block reaches max_position_embeddings, so there is nothing "
        "left for a position to run off the end of -- the limitation is gone "
        "for this configuration and the cell no longer means anything.")


def test_the_transformation_consumes_attention_mask_and_beam_idx(paged_census):
    """Both are declared so the pass can find them by name, and NEITHER
    survives it -- the served artifact's transformed input list has neither.
    A build that left them behind would hand the serving path two ports it
    never feeds, and `load_paged` would compile a model with dangling inputs.
    """
    consumed = {"attention_mask", "beam_idx"}
    print(f"[contract-kv] before: {sorted(paged_census['before'])}")
    print(f"[contract-kv] after : {sorted(paged_census['after'])}")
    assert consumed <= paged_census["before"], (
        f"the pass looks these up by name: {sorted(consumed)}")
    assert not (consumed & paged_census["after"]), (
        f"survived the transformation: "
        f"{sorted(consumed & paged_census['after'])}")


# ---------------------------------------------------------------------------
# The keystone: the whole 48-layer stack. Opt-in, because it is expensive.
# ---------------------------------------------------------------------------

# CF-RESIDENT (REVIEW 2a45349 F2). The peak-RSS ceiling for the 48-layer build,
# DERIVED from measurement on both sides rather than chosen. Dev host,
# 2026-09-12, OV 2026.4.0-22849, one variable per run (`rssprobe`, one module
# dropped from `_C_MODULES`, nothing else).
#
# The derivation used to be a table in this comment, and REVIEW 23938c1 F2 is
# what a table in a comment costs: `backbone` entered `_C_MODULES` in the same
# commit, no row was added for it, and a sentence below counted "four of the
# six" over a population that had become seven. So the table is DATA now.
# Every module in `_C_MODULES` must appear here -- the cell below fails if one
# does not -- and every count in prose about it is generated from this dict.
#
# The ceiling must sit above the authored peak and below the CHEAPEST defect.
# It is the GEOMETRIC MEAN of that pair, because that is the value with the
# same RELATIVE margin on each side, and peak RSS moves multiplicatively with
# how much of the model a defect copies, not additively.
#
# NO MARGIN IS QUOTED HERE, and that is the fix rather than an omission.
# REVIEW 23938c1 F3: this comment used to say "17.2% of headroom above the
# authored peak, 17.5% below the cheapest defect". Neither figure matched any
# consistent definition of its side, and with the rounded constant they are
# ordered the other way round -- in the one comment whose job is to stop the
# next session raising the ceiling without re-deriving it.
# `test_the_peak_rss_ceiling_is_the_geometric_mean_of_its_bracket` PRINTS both
# margins from the three constants and asserts the derivation that justifies
# them, so there is nothing here to drift.
#
# The ceiling stays a TYPED constant, re-derived by that cell rather than
# computed at import: a ceiling computed from the pair would absorb a raise
# silently, and the act this guard exists to catch is someone raising it. Typed
# + re-derived, that act is a red.
#
# RAISING THIS TO MAKE A RUN PASS RE-OPENS THE DEFECT. If the authored peak
# genuinely moves (a different OpenVINO, a different allocator), re-run both
# sides and re-derive; the two figures are what the constant means.
#
# RE-DERIVED 2026-09-13, because the authored peak genuinely moved and this
# comment's own instruction for that case is "re-run both sides and re-derive".
# What moved it: the GDN core became a token-sequential v5::Loop, so 48 layers'
# worth of unrolled chunked delta rule collapsed into 36 Loop bodies -- 84,158
# nodes to 16,766, and 4.52 GiB to 4.24. Every row below was re-measured the
# same day, one module per run, through the CF-KEYSTONERSS measurement path
# (parent-side os.wait4 over an un-reaped child), not carried over.
# The ceiling goes DOWN, 5.31 -> 5.12. Nothing here was raised to make a run
# pass; the run was already inside the old ceiling.
PEAK_RSS_AUTHORED_GIB = 4.24
PEAK_RSS_CHEAPEST_DEFECT_GIB = 6.19
PEAK_RSS_CEILING_GIB = 5.12        # == round(sqrt(4.24 * 6.19), 2), asserted

# Value = peak RSS in GiB of the 48-layer build with exactly that module
# dropped. `None` = NOT PROBED, with the reason in the row; no row borrows a
# figure it did not measure. Keys are module names as `_C_MODULES` reports
# them (`ss._C_MODULES` holds them under the aliases in brackets).
#
# The rows at the authored figure are why the keystone cell is not the whole
# closure: `test_every_module_binding_the_constant_factory_is_swapped` is.
PEAK_RSS_GIB_WHEN_DROPPED = {
    "moe":               6.19,   # [qmoe]  <- the CHEAPEST defect  (was 6.23)
    "attention":         8.67,   # [qattn] the reviewer's probe    (was 8.98)
    "hc":                9.49,   # [qhc]                           (was 9.51)
    "gdn":              29.62,   # [qgdn]                          (was 30.10)
    "ple":               4.24,   # [qple]  == authored: invisible to the RSS leg
    "piecewise_export":  4.24,   # [pwe]   == authored: invisible to the RSS leg
    #
    # `attention` NEARLY STOPPED BEING A WITNESS, and it is worth the four
    # lines. The first re-derivation after the reshape measured it at 4.24 --
    # invisible, the same reading the reviewer's own probe had produced 8.98
    # for. Cause: `emit_stateful_attention` was reaching for `qgdn._c` to build
    # the attention projections, as the rest of this module does, so dropping
    # `attention` from the swap list no longer copied anything. Both factories
    # are swapped and the graph is identical either way, so nothing would have
    # gone red -- the CF-RESIDENT leg would simply have had one fewer module it
    # could see, silently. The call sites now use `qattn._c`, which is where
    # those weights belong, and the row is a measured 8.67 again.
    # [qbb] NOT PROBED, and deliberately not given 4.52 by analogy.
    # `build_serving_shape_ir` imports `backbone` so `shared_constants()` can
    # swap it, and then never calls it: it reaches for `qgdn._c` directly for
    # `embed_w` and `head_w`. So dropping it cannot move peak RSS in THIS
    # build, for a structural reason rather than a measured one -- and that
    # reason is asserted, not asserted-by-comment, in
    # `test_the_rss_derivation_accounts_for_every_swapped_module`. The day
    # serving_shape reaches into `qbb`, that assertion goes red and this row
    # needs a probe.
    "backbone":          None,
}


def _rss_ceiling_margins():
    """`(above, below)`, the ceiling's two RELATIVE margins, from the constants.

    `above` = how far the ceiling sits over the authored peak; `below` = how
    far the cheapest visible defect sits over the ceiling. Ratios rather than
    differences, because that is what "the same relative margin on each side"
    means and what makes the geometric mean the right midpoint.
    """
    return (PEAK_RSS_CEILING_GIB / PEAK_RSS_AUTHORED_GIB - 1.0,
            PEAK_RSS_CHEAPEST_DEFECT_GIB / PEAK_RSS_CEILING_GIB - 1.0)


def _build_48_in_a_child():
    """Run the keystone build in a FRESH interpreter and return its report plus
    its own peak RSS.

    `ru_maxrss` is a high-water mark for the whole process, so measuring it
    inside pytest would measure whatever ran before this cell -- torch, the
    other suites, the module-scope 8-layer fixture. A child process is the only
    way the number means "this build", and it is also what makes the ceiling
    above reproducible from a bare shell.

    CF-KEYSTONERSS, landed 2026-09-13 with the reshape it was deferred to ride.
    The child used to report its OWN `getrusage(RUSAGE_SELF)`, read at one
    instant in the middle of its own run -- before `json.dumps` of the report,
    before `arena.close()`, and with no way for the parent to check it. Two
    holes, and the second is the one that matters:

      * anything the child allocates AFTER that line is invisible, so the
        number is a high-water mark of a prefix of the run, not of the run;
      * `subprocess.run` reaps the child itself, so the kernel's own accounting
        of that child is gone before the parent can look at it. The gated
        figure had exactly one witness, and it was the thing being measured.

    The parent now waits with `os.wait4` and reads `ru_maxrss` out of the
    kernel's accounting for that child, which is a high-water mark of the WHOLE
    child, taken after it exits. Both numbers are kept and the child's is
    asserted not to EXCEED the parent's -- it cannot, being a prefix of the
    same walk -- so the two witnesses disagree loudly rather than silently.
    `Popen` with the streams on temp files rather than pipes, because the child
    must be un-reaped when `wait4` runs and `communicate()` would both reap it
    and be the only safe way to drain a pipe.
    """
    import json
    import subprocess
    import tempfile
    src = r"""
import json, resource, sys, time
sys.path.insert(0, %r)
from q4e import serving_shape as ss
arena = ss.SparseArena()
try:
    t0 = time.time()
    model, report = ss.build_serving_shape_ir(seq_len=64, arena=arena)
    report["build_seconds"] = time.time() - t0
    report["disk_kib"] = arena.disk_kib()
    report["child_self_rss_gib"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 2**20
    sys.stdout.write("REPORT " + json.dumps(report) + "\n")
finally:
    arena.close()
""" % str(REPO_ROOT / "tools")
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen([sys.executable, "-c", src],
                                stdout=out, stderr=err)
        # `subprocess.run(..., timeout=1800)` carried a watchdog and `os.wait4`
        # does not -- it blocks forever. Without this, a hung 48-layer build
        # hangs the suite instead of failing it, which is a worse outcome than
        # the measurement gap the wait4 conversion closed. A timer thread that
        # kills the child is the smallest thing that keeps both.
        killer = threading.Timer(1800.0, proc.kill)
        killer.daemon = True
        killer.start()
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        finally:
            killer.cancel()
        proc.returncode = status                 # Popen must not wait() again
        out.seek(0), err.seek(0)
        stdout, stderr = out.read(), err.read()
    line = [l for l in stdout.splitlines() if l.startswith("REPORT ")]
    assert line, (
        "the keystone child produced no report.\n"
        # `os.wait4` hands back a raw wait status, not an exit code: a child
        # that exited 1 reports 256 here. A reader debugging this message
        # should see the number they would have seen from `subprocess.run`.
        f"exit={os.waitstatus_to_exitcode(status)} (raw wait status "
        f"{status})\nstdout tail:\n{stdout[-2000:]}\n"
        f"stderr tail:\n{stderr[-2000:]}")
    report = json.loads(line[-1][len("REPORT "):])
    # The kernel's accounting for the child, after it exited: the whole run.
    report["peak_rss_gib"] = usage.ru_maxrss / 2**20
    assert report["child_self_rss_gib"] <= report["peak_rss_gib"] + 1e-9, (
        f"the child reported {report['child_self_rss_gib']:.2f} GiB for itself "
        f"but the kernel accounted {report['peak_rss_gib']:.2f} GiB for the "
        f"same child. The child's read is a prefix of the parent's and cannot "
        f"exceed it; one of the two is not measuring this process.")
    return report


@pytest.mark.skipif(not os.environ.get("Q4E_SERVING_FULL"),
                    reason="Q4E_SERVING_FULL unset: the full 48-layer "
                           "real-geometry build is the keystone cell and is "
                           "run deliberately, not on every suite pass")
def test_the_full_48_layer_stack_emits_at_real_geometry():
    """FULL GEOMETRY STRUCTURE EMISSION -- the thing the refusal said was
    blocked. 48 layers, 36 GDN + 12 dense-causal, real widths, real vocabulary,
    experts slot-referenced, PLE table declared and never materialised.

    AND ITS RESIDENCY, which until CF-RESIDENT nothing measured. The headline
    is "183 GiB declared, built on a 48 GiB host". This paragraph used to name
    "the two assertions that carried it (`declared > 8 GiB`, `disk_kib <= 64`)"
    -- there is only ONE: `arena_declared_bytes` is PRINTED and never asserted,
    so the declared-bytes half of the headline has no gate behind it at all and
    the docstring was crediting it with one. What is true of `disk_kib <= 64`
    is what the paragraph was reaching for: it is INVARIANT to a module
    dropping out of `_C_MODULES`, because a copied constant lives in anonymous
    memory and never touches the arena file. The reviewer dropped
    `qattn` and every quantity this cell observed stayed bit-identical while
    peak RSS doubled. `peak_rss_gib` is the quantity that moves.
    """
    report = _build_48_in_a_child()
    cfg = pwe.real_config()
    rss = report["peak_rss_gib"]
    print(f"\n[serving-shape FULL] {report['n_layers']} layers "
          f"({report['gdn_layers']} GDN + {report['attn_layers']} attn), "
          f"T={report['seq_len']}")
    print(f"  nodes                 {report['nodes']:,}")
    print(f"  declared constants    "
          f"{report['arena_declared_bytes'] / 2**30:.2f} GiB")
    print(f"  graph const bytes     "
          f"{report['graph_const_bytes'] / 2**30:.2f} GiB")
    print(f"  arena blocks on disk  {report['disk_kib']} KiB")
    print(f"  build                 {report['build_seconds']:.1f} s")
    above, below = _rss_ceiling_margins()
    print(f"  PEAK RSS              {rss:.2f} GiB   "
          f"(authored {PEAK_RSS_AUTHORED_GIB}, ceiling "
          f"{PEAK_RSS_CEILING_GIB}, cheapest defect "
          f"{PEAK_RSS_CHEAPEST_DEFECT_GIB})")
    print(f"  ceiling margins       +{above * 100:.1f}% over the authored peak, "
          f"+{below * 100:.1f}% under the cheapest defect "
          f"(printed, never written down -- REVIEW 23938c1 F3)")
    assert report["n_layers"] == cfg.num_hidden_layers == 48
    assert report["gdn_layers"] == 36 and report["attn_layers"] == 12

    # THE NODE FLOOR IS RETIRED AND REPLACED, not lowered. It read
    # `nodes > 50_000` and the build now emits 16,766 -- because a v5::Loop is
    # a compact encoding of what used to be an unrolled chunked delta rule per
    # layer, and `get_ordered_ops` does not descend into a Loop body. Lowering
    # the number would have kept a weak proxy weak. What the floor was reaching
    # for is that 48 REAL layers are present, and the op histogram says that
    # directly, per layer kind, with every count derived from the layer counts
    # this cell has already asserted:
    gdn, attn = report["gdn_layers"], report["attn_layers"]
    hist = report["op_histogram"]
    structural = {
        "Loop": gdn,                        # one sequential delta rule per GDN
        "GroupConvolution": gdn,            # one short conv per GDN
        "ScaledDotProductAttention": attn,  # one per full-attention layer
        "ReadValue": 2 * gdn + 2 * attn,    # conv + ssm; key + value
        "Assign": 2 * gdn + 2 * attn,
    }
    print(f"  structural ops        "
          f"{ {k: hist.get(k, 0) for k in structural} }")
    for name, want in structural.items():
        assert hist.get(name, 0) == want, (
            f"{name}: {hist.get(name, 0)} in a {report['n_layers']}-layer "
            f"build ({gdn} GDN + {attn} attention); expected {want}. This is "
            f"the check that replaced a bare node-count floor -- it says which "
            f"layer kind is short, which a total never could.")
    # one more sign, not the guard -- see
    # test_no_expert_constant_is_materialised for why `disk_kib` cannot
    # distinguish an unwritten arena from a written one on this filesystem
    assert report["disk_kib"] <= 64
    assert report["outputs"][0][1] == [1, 64, cfg.vocab_size]
    assert rss <= PEAK_RSS_CEILING_GIB, (
        f"peak RSS {rss:.2f} GiB exceeds the derived ceiling "
        f"{PEAK_RSS_CEILING_GIB} GiB. The build declared the same "
        f"{report['arena_declared_bytes'] / 2**30:.2f} GiB over the same "
        f"{report['nodes']:,} nodes and still wrote {report['disk_kib']} KiB "
        f"to disk, so every other assertion in this cell is satisfied -- that "
        f"is the CF-RESIDENT signature. Check `_C_MODULES` in "
        f"tools/q4e/serving_shape.py against "
        f"test_every_module_binding_the_constant_factory_is_swapped before "
        f"touching this number; the cheapest module to drop costs "
        f"{PEAK_RSS_CHEAPEST_DEFECT_GIB} GiB.")


# ---------------------------------------------------------------------------
# CF-RESIDENT leg 2 -- the structural one, which runs on EVERY leg
# ---------------------------------------------------------------------------

def _modules_binding_the_constant_factory():
    """Every module under tools/q4e that binds the name `_c` at module level,
    by ast rather than by import: `from .gdn import _c`, `import _c as ...`,
    or its own `def _c` / `_c = ...`.

    Read from source on purpose. Importing to ask would run each module, and a
    module that failed to import would silently drop out of the answer -- the
    same shape of invisibility this whole finding is about.
    """
    import ast
    src_dir = REPO_ROOT / "tools" / "q4e"
    found = {}
    for p in sorted(src_dir.glob("*.py")):
        if p.name == "__init__.py":
            continue
        tree = ast.parse(p.read_text(), filename=str(p))
        where = None
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if (a.asname or a.name.split(".")[0]) == "_c":
                        where = node.lineno
            elif isinstance(node, ast.FunctionDef) and node.name == "_c":
                where = node.lineno
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "_c":
                        where = node.lineno
        if where is not None:
            found[p.stem] = where
    return found


def test_every_module_binding_the_constant_factory_is_swapped():
    """CF-RESIDENT, the leg the peak-RSS assertion cannot cover.

    `shared_constants()` swaps `_c` per module, and each importer holds its OWN
    binding, so a module missing from `_C_MODULES` keeps the copying factory
    and materialises its family's weights. The 48-layer cell catches that for
    SOME of the swapped modules and not others: dropping `qple` or `pwe` leaves
    peak RSS at the authored figure EXACTLY, because this particular build
    never reaches their `_c` with a large arena array. Measured, not inferred
    -- which is why a behavioural leg alone would be a guard with holes in it.

    HOW MANY of each is not written here (REVIEW 23938c1 F2: the count that
    stood in this docstring said "four of the six" on the day `_C_MODULES`
    became seven). The split is generated from `PEAK_RSS_GIB_WHEN_DROPPED` and
    printed by `test_the_rss_derivation_accounts_for_every_swapped_module`,
    which also fails if a module joins the swap list without a row.

    So the invariant is asserted structurally instead: a module that BINDS `_c`
    must be in `_C_MODULES`, whether or not today's build happens to call it.
    On its first run this found `q4e.backbone`, which binds `_c`
    (backbone.py:70) and spends it on `embed_w` and `head_w` -- 2.37 GiB each
    at real geometry, the largest pair in the model.

    The technique is `test_suite_guards.py`'s: read the source with `ast`,
    device-free, no shards, runs on every leg.
    """
    binders = _modules_binding_the_constant_factory()
    listed = {m.__name__.rsplit(".", 1)[-1] for m in ss._C_MODULES}
    print(f"\n[contract-cmodules] modules binding `_c`: "
          f"{', '.join(f'{k} (:{v})' for k, v in sorted(binders.items()))}")
    print(f"[contract-cmodules] _C_MODULES: {sorted(listed)}")

    assert binders, (
        "no module under tools/q4e binds `_c` -- the scanner is broken, not "
        "the tree (q4e.gdn defines it and at least five modules import it)")
    missing = sorted(set(binders) - listed)
    assert not missing, (
        "module(s) bind the `_c` constant factory but are absent from "
        "`_C_MODULES` in tools/q4e/serving_shape.py: "
        + ", ".join(f"{m} (q4e/{m}.py:{binders[m]})" for m in missing)
        + ". `shared_constants()` will not swap them, so every constant they "
          "build during a serving-shape build is COPIED into anonymous memory "
          "-- invisible to the node count, to the declared bytes and to "
          "`disk_kib`, and visible only as peak RSS or as an OOM.")
    # and the reverse: a name in _C_MODULES that no longer binds `_c` is a
    # stale entry whose swap is a no-op, which would make this gate weaker
    # than it reads.
    stale = sorted(listed - set(binders))
    assert not stale, (
        f"`_C_MODULES` lists {stale}, which bind no `_c`; the swap is a no-op "
        f"for them. Remove them, or this gate over-reports its own coverage.")


_SERVING_SHAPE_PY = REPO_ROOT / "tools" / "q4e" / "serving_shape.py"


def _serving_shape_attribute_bases():
    """The module aliases `serving_shape.py` actually REACHES INTO (`alias.x`),
    by ast. An alias that is imported and never dereferenced is in
    `_C_MODULES` for the swap alone -- which is the whole of `backbone`'s
    position in the RSS derivation.
    """
    import ast
    tree = ast.parse(_SERVING_SHAPE_PY.read_text())
    return {n.value.id for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}


def _serving_shape_import_alias(module):
    """The local name `serving_shape.py` binds for `q4e.<module>`, by ast.

    CF-GUARDALIAS (REVIEW 7c26cce F8). The deref guard below used to spell its
    subject as the literal string `"qbb"`. That string is not the module; it is
    one spelling of the alias the module happened to be imported under, and an
    assertion keyed on a spelling passes VACUOUSLY the moment the spelling
    changes. Measured on a mutated tree before this function existed: rename
    every `qbb` to `qback` and add one real `qback._c` dereference, and the
    cell reported `1 passed` while `backbone` WAS being reached into -- the
    `None` row in `PEAK_RSS_GIB_WHEN_DROPPED` kept a licence it had just lost.

    So the alias is DERIVED from the `ImportFrom` node instead of typed -- the
    same move `cite()` makes for line numbers one level down, and for the same
    reason (clause 2: the defining source generates it, nobody recites it).
    Uniqueness and existence are both asserted: a `backbone` import that has
    been removed, or bound twice, names nothing, and that is a red rather than
    a silent pass.
    """
    import ast
    tree = ast.parse(_SERVING_SHAPE_PY.read_text())
    bound = [a.asname or a.name.rsplit(".", 1)[-1]
             for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             for a in n.names if a.name.rsplit(".", 1)[-1] == module]
    assert len(bound) == 1, (
        f"`serving_shape.py` binds {module!r} {len(bound)} times ({bound}); "
        f"this guard needs exactly one alias to track. If the import was "
        f"removed, {module!r} no longer belongs in `_C_MODULES` either; if it "
        f"is bound twice, the deref guard below would only watch one of them.")
    return bound[0]


def test_the_rss_derivation_accounts_for_every_swapped_module():
    """REVIEW 23938c1 F2. The ceiling is derived over a POPULATION, and this
    asserts the population is the one `shared_constants()` actually swaps.

    The defect being closed is not arithmetic: the derivation table lived in a
    comment, `backbone` joined `_C_MODULES` in the commit that wrote the table,
    no row was added, and a docstring counted "four of the six" over seven.
    Nothing was mismeasured and nothing went red. That is the shape.

    So: every swapped module has a row here, every row is a swapped module, the
    cheapest visible defect is MINIMISED OUT OF THE ROWS rather than copied
    into a constant, and the one unprobed row states its structural reason --
    `serving_shape` imports `backbone` for the swap and never dereferences it.
    That last one is asserted from source, so the day the build reaches into
    `backbone` this cell goes red and asks for a probe instead of letting the
    ceiling be derived over a population with a hole in it.

    CF-GUARDALIAS (REVIEW 7c26cce F8): that deref assertion names the module
    and resolves its alias through `_serving_shape_import_alias`, because the
    hardcoded spelling it used to carry passed vacuously under a rename -- see
    that function's docstring for the measured red.
    """
    listed = {m.__name__.rsplit(".", 1)[-1] for m in ss._C_MODULES}
    rows = set(PEAK_RSS_GIB_WHEN_DROPPED)
    probed = {m: v for m, v in PEAK_RSS_GIB_WHEN_DROPPED.items() if v is not None}
    visible = sorted(m for m, v in probed.items() if v > PEAK_RSS_AUTHORED_GIB)
    invisible = sorted(listed - set(visible))

    print(f"\n[rss-derivation] swapped modules: {len(listed)}  "
          f"probed: {len(probed)}  unprobed: {len(rows) - len(probed)}")
    print(f"[rss-derivation] the 48-layer cell SEES {len(visible)} of "
          f"{len(listed)}: {', '.join(visible)}")
    print(f"[rss-derivation] INVISIBLE to it ({len(invisible)}): "
          f"{', '.join(invisible)} -- covered by the structural leg only")
    for m in sorted(PEAK_RSS_GIB_WHEN_DROPPED):
        v = PEAK_RSS_GIB_WHEN_DROPPED[m]
        print(f"    {m:20s} {'not probed' if v is None else f'{v:6.2f} GiB'}")

    assert rows == listed, (
        f"the peak-RSS derivation and `_C_MODULES` describe different "
        f"populations. Rows with no swapped module: "
        f"{sorted(rows - listed) or 'none'}; swapped modules with no row: "
        f"{sorted(listed - rows) or 'none'}. A module that joins the swap list "
        f"without a row makes every count about the derivation wrong and the "
        f"ceiling's bracket unattributed -- add the row (probe it, or state "
        f"why it cannot move peak RSS) rather than this assertion.")
    assert visible, "no row exceeds the authored peak: the bracket has no upper end"
    cheapest = min(probed[m] for m in visible)
    assert PEAK_RSS_CHEAPEST_DEFECT_GIB == cheapest, (
        f"PEAK_RSS_CHEAPEST_DEFECT_GIB is {PEAK_RSS_CHEAPEST_DEFECT_GIB}, but "
        f"the cheapest visible defect in the rows is {cheapest} "
        f"({min(visible, key=lambda m: probed[m])}). The constant is the "
        f"ceiling's upper bracket; it must be the minimum of the measured "
        f"defects, not a figure that was true when it was typed.")
    alias = _serving_shape_import_alias("backbone")
    print(f"[rss-derivation] `backbone` is imported as `{alias}` (derived from "
          f"the ImportFrom node, not typed) and is dereferenced: "
          f"{alias in _serving_shape_attribute_bases()}")
    assert alias not in _serving_shape_attribute_bases(), (
        f"`serving_shape.py` now dereferences `{alias}` (its alias for "
        f"`backbone`), so dropping `backbone` from `_C_MODULES` may move peak "
        f"RSS. Its row in PEAK_RSS_GIB_WHEN_DROPPED is `None` on the grounds "
        f"that it cannot. Probe it (rssprobe, one module dropped) and record "
        f"the figure.")


def test_the_peak_rss_ceiling_is_the_geometric_mean_of_its_bracket():
    """REVIEW 23938c1 F3. The ceiling's justification, checked against the
    ceiling -- and the two margins PRINTED instead of quoted.

    What was wrong: the comment quoted "17.2% above / 17.5% below". The exact
    geometric mean gives the same figure on both sides by construction, and
    rounding it to 0.01 GiB splits them slightly the OTHER way (the margin
    above the authored peak becomes the larger one). So the sentence disagreed
    with its own argument, with the constant, and with itself, and nothing in
    the suite could notice.

    What is asserted now is the argument itself: the ceiling brackets its pair,
    and it IS the geometric mean rounded to 0.01 GiB. The constant stays typed
    on purpose -- a ceiling computed at import would quietly absorb a raise,
    and a raise is the act this whole guard exists to catch. Typed and
    re-derived here, raising `PEAK_RSS_CEILING_GIB` without re-recording a
    measured side is a red.

    CF-TOLGEN (REVIEW 7c26cce F9). The margin tolerance was the hand-typed
    `0.005`, and it was unfailable: with `c == round(gm, 2)` (the round-trip
    assertion below, which is the real gate) the error `|c - gm| <= 0.005`
    propagates to the gap with derivative `1/a + d/c**2`, so at 4.52 / 6.23 the
    widest gap that arithmetic ALLOWS is 0.44219 * 0.005 = 0.002211 -- inside
    0.005 by 2.3x, always, while its sibling passes. `0.005` also tracked
    nothing: the bound scales with the bracket, and at a = 1, d = 100 the same
    constant is a SPURIOUS red at 0.01.

    The tolerance is now GENERATED from the bracket it is a bound for, which
    fixes both halves. What that leaves is a check that is not dead weight, and
    the distinction is worth stating because F9 read it as dead: the round-trip
    assertion constrains the CEILING against the two measured sides, and
    nothing in it constrains `_rss_ceiling_margins()`, which computes the two
    margins by its own two formulas. Break either formula and the gap leaves
    the bound while the ceiling still round-trips. So it can fail, for a defect
    its sibling cannot see.

    RETRACTION, H1 (REVIEW f8229d8), and it is the defect class this cell is
    named after. This paragraph used to read "goes red at 30.8 pp against a
    0.2211 pp bound". BOTH FIGURES WERE WRONG AND NEITHER CAME FROM A RUN.
    30.8 was typed from intent before the mutation was executed; the run that
    produced 20.3540 happened in the same session, corrected the commit
    message, and never came back to this docstring. 0.2211 is the worse half:
    it is the derivative at `c`, which the paragraph four lines below this one
    DISOWNS as the wrong bound in the same breath -- so the docstring quoted as
    "the bound" the exact number it names as the tolerance that can red without
    a defect, leaving a live invitation to "reconcile" the two by setting the
    tolerance to it. Clause 2 does not exempt red figures: a number written
    from what a mutation OUGHT to produce is a recital whichever side of the
    assertion it sits on. Regenerated below by running each mutation.

    THE ORDERING INVARIANT WAS RIGHT ABOUT THE RULE AND WRONG ABOUT THE
    DIRECTION, found 2026-09-13 when the bracket was re-derived. It read
    `0 < below < above < 1`, and its message explained that "`below` must be
    the SMALLER, because rounding the geometric mean UP to 0.01 GiB is what
    makes the upper margin the larger one". True of the 4.52 / 6.23 bracket,
    whose mean 5.3149 rounds UP -- and generalised from that one instance. The
    4.24 / 6.19 bracket's mean is 5.123046, which rounds DOWN, so the larger
    margin is the one BELOW and a cell with nothing wrong in it went red. The
    rule is unchanged and is now stated over the direction rather than assuming
    it: the larger margin is on the side the rounding moved the ceiling toward.
    It kills every mutation it killed before, re-run rather than assumed.

    WHAT THE MUTATIONS ACTUALLY PRODUCE, re-run 2026-09-13 on the re-derived
    bracket (staged tree, dev host CPU, this cell alone, `-k geometric`). The
    control is the live tree: `1 passed`, gap 0.1437 pp <= 0.2362 pp.

        ENG  below vs the AUTHORED PEAK    gap 25.2358 pp   1 failed
        M1   the two returns SWAPPED       gap  0.1437 pp   1 failed
        M2   the -1.0 dropped              gap  0.1437 pp   1 failed
        M3   bracket ends swapped          gap  0.0984 pp   1 failed
        M4   both margins the SAME expr    gap  0.0000 pp   1 failed

    ONLY `ENG` IS CAUGHT BY THE GAP BOUND. M1-M4 all sit INSIDE 0.2362 pp and
    passed everything this cell asserted until H2 added the ordering invariant
    -- REVIEW f8229d8 §14 found them by attacking the cell rather than reading
    it. M1 is the one that matters: it orders the two margins backwards, which
    is F3 itself, the defect this cell was written to prevent walking straight
    through its own guard. M4 makes "two independently computed margins"
    vacuous at gap exactly 0. The bound tests that the margins AGREE; it never
    tested that either is the right formula, and the invariant is what closes
    that. The figures above are from the six runs, not from what they ought to
    produce -- the H1 retraction below is what that costs when it is skipped.

    The bound is the exact one, not the derivative at `c`: `gap(gm) == 0` and
    `gap(c) = |integral from gm to c of (1/a + d/t**2) dt|`, so with
    `|c - gm| <= 0.005` and the integrand falling in `t`, the supremum is
    `0.005 * (1/a + d/(c - 0.005)**2)` -- 0.2213 pp here against the observed
    0.1521 pp. Taking the derivative at `c` instead gives 0.2211 pp, which is
    below the true supremum and would be a tolerance that can be exceeded
    without a defect; the 1.01 slack factor REVIEW 7c26cce F9 suggested is
    that same gap covered by a fudge instead of by the integral. (The 0.2213
    figure is the OLD bracket's; at 4.24 / 6.19 the same formula gives 0.2362,
    which the cell generates and prints rather than carrying here.)
    """
    above, below = _rss_ceiling_margins()
    exact = math.sqrt(
        PEAK_RSS_CHEAPEST_DEFECT_GIB / PEAK_RSS_AUTHORED_GIB) - 1.0
    unrounded = math.sqrt(PEAK_RSS_AUTHORED_GIB * PEAK_RSS_CHEAPEST_DEFECT_GIB)

    print(f"\n[rss-ceiling] bracket {PEAK_RSS_AUTHORED_GIB} < "
          f"{PEAK_RSS_CEILING_GIB} < {PEAK_RSS_CHEAPEST_DEFECT_GIB} GiB "
          f"(authored < ceiling < cheapest defect)")
    print(f"[rss-ceiling] sqrt({PEAK_RSS_AUTHORED_GIB} * "
          f"{PEAK_RSS_CHEAPEST_DEFECT_GIB}) = {unrounded:.6f} -> "
          f"{PEAK_RSS_CEILING_GIB} after rounding to 0.01 GiB")
    print(f"[rss-ceiling] margins: +{above * 100:.3f}% above the authored "
          f"peak, +{below * 100:.3f}% below the cheapest defect "
          f"(exact mean: {exact * 100:.3f}% each side)")
    # CF-TOLGEN: the tolerance is GENERATED from the bracket, never typed. The
    # supremum of the gap over `|c - gm| <= 0.005`, integrand taken at the
    # worst end of the interval so this is an upper bound and not an estimate.
    _HALF_GRID = 0.005                     # half of the 0.01 GiB rounding grid
    gap_bound = _HALF_GRID * (1.0 / PEAK_RSS_AUTHORED_GIB
                              + PEAK_RSS_CHEAPEST_DEFECT_GIB
                              / (PEAK_RSS_CEILING_GIB - _HALF_GRID) ** 2)
    print(f"[rss-ceiling] margin gap {abs(above - below) * 100:.4f} pp <= "
          f"{gap_bound * 100:.4f} pp, the most the 0.01 GiB rounding can "
          f"explain at this bracket (generated: 0.005 * (1/{PEAK_RSS_AUTHORED_GIB} "
          f"+ {PEAK_RSS_CHEAPEST_DEFECT_GIB}/"
          f"{PEAK_RSS_CEILING_GIB - _HALF_GRID:.3f}^2))")

    assert (PEAK_RSS_AUTHORED_GIB < PEAK_RSS_CEILING_GIB
            < PEAK_RSS_CHEAPEST_DEFECT_GIB), (
        f"the ceiling {PEAK_RSS_CEILING_GIB} does not bracket: it must sit "
        f"above the authored peak {PEAK_RSS_AUTHORED_GIB} and below the "
        f"cheapest defect {PEAK_RSS_CHEAPEST_DEFECT_GIB}. One side of the "
        f"derivation was re-measured and the other was not.")
    assert PEAK_RSS_CEILING_GIB == round(unrounded, 2), (
        f"the ceiling {PEAK_RSS_CEILING_GIB} is not sqrt("
        f"{PEAK_RSS_AUTHORED_GIB} * {PEAK_RSS_CHEAPEST_DEFECT_GIB}) = "
        f"{unrounded:.6f} rounded to 0.01 GiB ({round(unrounded, 2)}). Either "
        f"the ceiling was raised without re-deriving it -- which re-opens the "
        f"defect it guards -- or a measured side moved and the ceiling was "
        f"not recomputed.")
    # H2 (REVIEW f8229d8). The ORDER and RANGE of the two margins, which the
    # gap bound does not constrain: it tests only that they AGREE, so every
    # formula defect preserving the near-symmetry walks through it -- including
    # the two margins ordered backwards, which is F3, the defect this cell
    # exists to prevent. Not a restatement of the formulas: `below < above` is
    # forced by the ROUNDING (c > gm makes the upper margin the larger one) and
    # `0 < .. < 1` by the bracket, both independently of how either is
    # computed. Four mutations that passed the bound alone are red on this line
    # and are listed in the docstring.
    # WHICH MARGIN IS LARGER IS DECIDED BY THE ROUNDING, AND THE ROUNDING GOES
    # BOTH WAYS. This assertion read `0 < below < above < 1` and its message
    # said "`below` must be the SMALLER, because rounding the geometric mean UP
    # to 0.01 GiB is what makes the upper margin the larger one". That is true
    # of a mean whose third decimal rounds up, which is what the 4.52 / 6.23
    # bracket did (5.3149 -> 5.31), and it was generalised from that one
    # instance. Re-deriving the bracket at 4.24 / 6.19 gives 5.123046, which
    # rounds DOWN, and the invariant went red on a cell where nothing was
    # wrong. The rule is the same rule, stated over the direction instead of
    # assuming it: the larger margin is on the side the rounding moved the
    # ceiling TOWARD.
    larger = "above" if PEAK_RSS_CEILING_GIB > unrounded else "below"
    if PEAK_RSS_CEILING_GIB == unrounded:                    # pragma: no cover
        larger = "neither"
    print(f"[rss-ceiling] the ceiling rounded "
          f"{'UP' if larger == 'above' else 'DOWN'} from {unrounded:.6f}, so "
          f"the larger margin must be the one {larger} it")
    assert 0 < above < 1 and 0 < below < 1, (
        f"the ceiling's margins are +{above * 100:.3f}% above / "
        f"+{below * 100:.3f}% below. Both must be positive (the ceiling sits "
        f"strictly inside its bracket) and both under 1 (neither side "
        f"doubles). A violation here is a margin computed by the wrong "
        f"formula, not a bracket that moved -- the bracket has its own "
        f"assertion above.")
    ordered = below < above if larger == "above" else above < below
    assert ordered, (
        f"the ceiling's margins are +{above * 100:.3f}% above / "
        f"+{below * 100:.3f}% below, but the ceiling rounded "
        f"{'UP' if larger == 'above' else 'DOWN'} from {unrounded:.6f} to "
        f"{PEAK_RSS_CEILING_GIB}, so the margin {larger} it must be the "
        f"LARGER. Ordered the other way, the two margins are not the two "
        f"formulas they claim to be -- that is F3, the defect this cell "
        f"exists to prevent, and it is what the ordering catches that the "
        f"gap bound below cannot.")
    assert abs(above - below) <= gap_bound, (
        f"the ceiling's two relative margins differ by "
        f"{abs(above - below) * 100:.4f} percentage points "
        f"(+{above * 100:.3f}% / +{below * 100:.3f}%), which EXCEEDS the "
        f"{gap_bound * 100:.4f} pp the 0.01 GiB rounding can explain at this "
        f"bracket. The assertion above already forces the ceiling to be the "
        f"rounded geometric mean, so the gap cannot get here through the "
        f"ceiling: what is wrong is how a margin is COMPUTED "
        f"(`_rss_ceiling_margins`), not how wide this tolerance is. Do not "
        f"widen it -- it is derived from the bracket, and widening it would "
        f"only hide the formula that changed.")


def test_the_constant_factory_scanner_detects_a_missing_module():
    """The scanner's own red, in tree and permanent.

    A gate that can only pass is not a gate. `_C_MODULES` minus one entry must
    be rejected by the same comparison the cell above runs.
    """
    binders = _modules_binding_the_constant_factory()
    listed = {m.__name__.rsplit(".", 1)[-1] for m in ss._C_MODULES}
    assert not (set(binders) - listed), "precondition: the tree is clean"
    for drop in sorted(listed):
        mutated = listed - {drop}
        missing = set(binders) - mutated
        assert missing == {drop}, (
            f"dropping {drop!r} from _C_MODULES was not detected as missing: "
            f"{missing}")
    print(f"[contract-cmodules] scanner rejects each of {len(listed)} "
          f"single-module deletions")
