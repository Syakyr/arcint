"""The plugin's native lowering (marfrit-openvino patch 0043), on the card.

A tiled MoE block at a small REAL-format geometry (hidden 512, inter 256,
4 experts, top-2 -- IQ3_XXS needs 256-wide rows) with RANDOM native blocks
(not alike across rows or pages), saved as IR so the offload path has
file-backed Constants, compiled on the GPU with the served properties
(OFFLOAD_RATIO, MOE_CPU_TIER, WEIGHTS_PATH) and compared against the CPU
plugin's exact run of the same IR. The census asserts the fused primitive
exists (the lowering fired) -- a silent fall-through to generic ops would
still compute the right numbers here and would be the wrong mechanism.

Runs only with the suite's recorded GPU gate set (`Q4E_GPU=GPU.0`, see
tests/python/q4e_device.py; a new gate would be a new axis in the count
space, test_suite_guards) and the patched runtime on PYTHONPATH; the SOP
card window applies (docs/sop-card-window.md).
"""
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import Type, opset13 as op

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q4e import native_blocks as nb  # noqa: E402
from q4e import serving_shape as ss  # noqa: E402
from q4e_device import device_params  # noqa: E402

_GPUS = [d for d in device_params() if d != "CPU"]
_skip = pytest.mark.skipif(not _GPUS, reason="Q4E_GPU unset (a card window)")


class _RandomNativeFiller:
    """A native filler serving random valid blocks in the checkpoint's two
    per-layer combinations: IQ3_XXS gate/up over an IQ4_NL down (43 layers)
    and IQ4_XS gate/up over a Q8_0 down (layer 2)."""

    def __init__(self, gate_up_fmt, down_fmt, seed=0):
        self.fmts = {"gate": gate_up_fmt, "up": gate_up_fmt, "down": down_fmt}
        self.rng = np.random.default_rng(seed)

    def native(self, layer, kind, e, out, inn):
        fmt = self.fmts[kind]
        block, nbytes = nb.BLOCK_BYTES[fmt]
        rows, nblk = e * out, inn // block
        raw = self.rng.integers(0, 256, size=(rows, nblk, nbytes), dtype=np.uint8)
        d = self.rng.uniform(0.01, 0.2, size=(rows, nblk)).astype(np.float32) / (31.0 if fmt == "IQ4_XS" else 1.0)
        raw[:, :, 0:2] = d.astype("<f2").view(np.uint8).reshape(rows, nblk, 2)
        return fmt, nb.SPLIT[fmt](raw.reshape(rows, nblk * nbytes))


class _RandomAffineFiller:
    """The CONTROL: the stock u4 grouped-affine bodies (serving_shape's
    fusing chain, group 128) at the same geometry, through the same harness
    and properties. A failure here is the harness or the environment, not
    the native lowering."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def body(self, layer, kind, e, out, inn):
        from q4e.expert_fill import pack_u4
        gs = ss.EXPERT_GROUP_SIZE
        groups = inn // gs
        codes = self.rng.integers(0, 16, size=(e, out, groups, gs), dtype=np.uint8)
        zp = self.rng.integers(0, 16, size=(e, out, groups, 1), dtype=np.uint8)
        sc = self.rng.uniform(0.001, 0.02, size=(e, out, groups, 1)).astype(np.float32)
        return pack_u4(codes), pack_u4(zp), sc


def _config():
    from transformers.models.qwen4_exp import configuration_qwen4_exp as pin_cfg
    return pin_cfg.Qwen4ExpTextConfig(
        hidden_size=512, num_hidden_layers=1, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True, moe_intermediate_size=256, shared_expert_intermediate_size=64,
        hidden_act="silu", hc_count=4, hc_lowrank=8, rms_norm_eps=1e-6,
        layer_types=["linear_attention"], vocab_size=257, eos_token_id=0, pad_token_id=0,
    )


def _build(tmp_path, T, gate_up_fmt, down_fmt):
    cfg = _config()
    H, I, E, Is = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, cfg.shared_expert_intermediate_size
    rng = np.random.default_rng(1)
    arena = ss.SparseArena(path=str(tmp_path / "arena.bin"))
    st = {
        "mlp.gate.weight": (rng.standard_normal((E, H)) * 0.05).astype(np.float32),
        "mlp.shared_expert.gate_proj.weight": (rng.standard_normal((Is, H)) * 0.05).astype(np.float32),
        "mlp.shared_expert.up_proj.weight": (rng.standard_normal((Is, H)) * 0.05).astype(np.float32),
        "mlp.shared_expert.down_proj.weight": (rng.standard_normal((H, Is)) * 0.05).astype(np.float32),
        "mlp.shared_expert_gate.weight": (rng.standard_normal((1, H)) * 0.05).astype(np.float32),
    }
    # T dynamic as in every served artifact: the emitter's Reshape targets keep
    # M as the runtime -1, and the plugin's router/MoE lowering is only ever
    # exercised with a dynamic token dim (the first form of this cell declared
    # T static and the stock control failed at compile on a missing router
    # primitive, 2026-09-18 GPU.1)
    x = op.parameter([1, -1, H], Type.f32, name="x")
    filler = _RandomAffineFiller() if gate_up_fmt == "affine" else _RandomNativeFiller(gate_up_fmt, down_fmt)
    y = ss.emit_moe_tiled(x, cfg, st, arena, T, "layer0/moe", filler=filler, layer=0)
    model = ov.Model([op.result(y)], [x], "native_moe_block")
    ov.save_model(model, str(tmp_path / "moe.xml"), compress_to_fp16=False)
    return arena, cfg


@_skip
@pytest.mark.parametrize("dev", _GPUS)
@pytest.mark.parametrize("gate_up_fmt,down_fmt", [("affine", "affine"), ("IQ3_XXS", "IQ4_NL"), ("IQ4_XS", "Q8_0")])
def test_the_native_block_lowers_to_the_fused_primitive_and_matches_the_cpu_plugin(tmp_path, dev, gate_up_fmt, down_fmt):
    _DEV = dev
    T = 6
    arena, cfg = _build(tmp_path, T, gate_up_fmt, down_fmt)
    try:
        core = ov.Core()
        xml = str(tmp_path / "moe.xml")
        x = np.random.default_rng(2).standard_normal((1, T, cfg.hidden_size)).astype(np.float32)
        ref = core.compile_model(core.read_model(xml), "CPU")
        want = ref({"x": x})[ref.output(0)]
        props = {"OFFLOAD_RATIO": "50", "MOE_CPU_TIER": "YES",
                 "WEIGHTS_PATH": str(tmp_path / "moe.bin"), "INFERENCE_PRECISION_HINT": "f16"}
        gpu = core.compile_model(core.read_model(xml), _DEV, props)
        got = gpu({"x": x})[gpu.output(0)]
        types = {}
        native_nodes = []
        for n in gpu.get_runtime_model().get_ordered_ops():
            t = n.get_rt_info()["layerType"].astype(str) if "layerType" in n.get_rt_info() else n.get_type_name()
            types[t] = types.get(t, 0) + 1
            if "MOECompressedNative" in n.get_friendly_name():
                native_nodes.append(n.get_friendly_name())
    finally:
        arena.close()
    moe_typed = {k: v for k, v in types.items() if "moe" in k.lower()}
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    # per token: the block's output is a top-2 sum of expert rows, so a wrong
    # expert (or a wrong sign table in one) moves elements by the order of
    # the row's RMS; the f16 hidden/output path moves them by ~1e-3 of it.
    # The band is 1% of the element plus 0.5% of its row's RMS -- not a
    # fraction of the tensor's peak applied everywhere (review, 2026-09-18).
    rms_row = np.sqrt((want.astype(np.float64) ** 2).mean(axis=-1, keepdims=True))
    band = 1e-2 * np.abs(want) + 5e-3 * rms_row
    print(f"\n[native-lowering] {_DEV} {gate_up_fmt}/{down_fmt}: moe-typed primitives {moe_typed}; native nodes {native_nodes}; "
          f"max|diff| {d.max():.4e} vs max|want| {np.abs(want).max():.4e}; max diff/band {(d / band).max():.3f}; "
          f"corr {np.corrcoef(got.ravel(), want.ravel())[0, 1]:.6f}")
    assert moe_typed, f"no MoE-typed primitive in the runtime graph: the lowering did not fire ({sorted(types)[:12]})"
    if gate_up_fmt != "affine":
        # the native pass names its op; the stock fusion never produces this name
        assert native_nodes, "a MoE primitive exists but none carries the native pass's name: the stock fusion took it"
    assert (d <= band).all(), f"max diff/band {(d / band).max():.3f}, max|diff| {d.max():.4e}"
