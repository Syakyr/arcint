# Routing-aware expert execution — design note

Campaign: `docs/campaigns/sub4bit-vram-kernel.md`.
Gate: a model with ≥ 512 experts serves on one card with routing-aware expert
execution — only the routed experts computed per token, a GPU LRU cache
holding the hot set, misses fed from the host pool. Prüfstand 10/10;
greedy output byte-identical across two cold starts (§3.4).

---

## §1 — what this replaces and why

The plugin's MoE fusion (`moe_3gemm_swiglu_opt.cpp`, matcher
`keep_moe_3gemm_const_precision.cpp`) stacks every expert's gate/up/down
weights into one `[num_slots, …]` constant tensor per layer per kind and
computes them all in one fused kernel. The fusion's own matcher requires u4
Constants on all twelve weight and zero-point inputs (DESIGN §7.0.2ah,
`measured-here`). A host-resident expert cannot be a Constant (the plugin
stages every Constant in host memory at compile — 66 GiB for 48 layers,
DESIGN §7.0.2 CORRECTION III). Leaving Constant-land for Parameters loses
the fusion and with it the routing: every expert computes for every token
(window-051 B.3: 7.06× device residency per layer, 623× warm forward,
`measured-here`). At `num_experts_per_tok: 10` of 512, that is 51.2× the
expert work the architecture requires.

The per-expert kernel this design describes bypasses the fusion entirely.
It takes one expert's packed u4 weights, dequantises in-kernel (in
registers, never materialised), and computes the SwiGLU MLP
`down(SiLU(gate(x)) · up(x))` for the routed tokens only. Combined with
the GPU LRU cache (§3) and the host miss tier (§4), this is the mechanism
FreeToken implements (`code`: `~/src/FreeToken-ref`,
`docs/research-freetoken-code.md`) — ported to the OpenVINO GPU plugin's
kernel infrastructure.

The fusion stays for the fully-resident case (the 35B coder at ratio ≤ 75,
where the slot pool fits in Constants). The per-expert kernel activates
when the offload tier is live AND the expert count exceeds the resident slot
capacity — the Flash-Next regime.

---

## §2 — the per-expert kernel contract

### §2.1 — operation

The MoE expert MLP is a SwiGLU block:

    hidden_out = down_proj(SiLU(gate_proj(x)) * up_proj(x))

where `x` is the hidden state of one token (or a batch of tokens) after the
router has selected this expert, and `hidden_out` is accumulated into the
layer output weighted by the router's gating coefficient.

The shared expert (`shared_expert_intermediate_size: 640`, 0.31 GiB,
always computed for every token) is NOT part of this kernel — it runs
through the backbone's non-MoE path as a device-resident constant.

Three weight matrices per routed expert per layer:
- `gate_proj`: `[moe_intermediate_size, hidden_size]` — 640 × 2560 for
  Flash-Next
- `up_proj`: `[moe_intermediate_size, hidden_size]` — 640 × 2560
- `down_proj`: `[hidden_size, moe_intermediate_size]` — 2560 × 640

At int4 (0.5 B/element): 2,457,600 bytes per expert per layer
(`measured-here`, `docs/design-qwen-flash-next.md` Fit table).

### §2.2 — inputs and outputs

**Inputs** to the kernel, per expert invocation:
1. Packed u4 weight bytes for gate, up, down — from the GPU LRU slot or
   the host pool (§4). Shape: the three matrices concatenated or addressed
   separately. Group scales and zero-points in their existing format
   (f16 scales, u4 zero-points, per the artifact's group-quantisation
   layout).
2. Hidden state `x` — the tokens routed to this expert. f16 on the plugin's
   served path. For decode: one token, shape `[1, hidden_size]` (GEMV). For
   prefill: a batch, shape `[n_tokens, hidden_size]` (GEMM).
3. Router gating weight — f16 scalar per token, from the top-k softmax.

**Output**: the weighted expert contribution, shape `[n_tokens, hidden_size]`,
accumulated (index_add) into the layer's MoE output tensor. The accumulation
is the same `scatter_reduce` / `index_add` path the existing grouped-GEMM
and per-expert-onednn paths use (patch 0037's `host_only` path is the
model).

### §2.3 — in-kernel dequant

The u4 weight bytes reach the kernel in their packed form (two elements per
byte, low nibble = element 2j, high nibble = element 2j+1 — the same
packing as `moe_3gemm_swiglu_mlp.cl`, derived in patch 0011's header from
the generated kernel's lane pattern). Dequantisation happens inside the
GEMV/GEMM K-loop:

    w_f16 = (nibble - zero_point) * scale

where `zero_point` and `scale` are per-group (the group size from the
artifact, typically 128). The f16 multiplication with the hidden-state
element and the partial-sum accumulation happen in the same register pass.
No intermediate f16 or f32 weight tensor is ever written to memory.

This is the same mechanism FreeToken uses for all three of its supported
formats (`code`: `_BANK_SCHEMAS` comments in `offload_cache.py` — "dequantizes
in the K-loop (no bf16 materialization)" for fp8_block, "dequantized inside the
borrowed ggml MoE kernels" for q4_0, "Triton inline-dequant kernels" for nvfp4).

### §2.3a — the resident format is the checkpoint's own blocks (amended 2026-09-18)

§2.3's u4 affine form is what the artifact holds today, and its price is
now measured (`measured-here`, campaign `serving-shape-logits`, DESIGN
§7.0.2bz): re-quantising the checkpoint's experts into u4 grouped-affine
codes costs 0.10–0.13 relative RMS per expert tensor at group 128 and
still 0.07–0.08 at group 16, and shows as 0.73 nats of KL at depth 48 on
a served model that is otherwise the model's (every other block agrees
with llama.cpp within quantisation noise). A finer group is mechanically
available (the fused MoE takes `{experts, ofm, num_groups, group_size}`)
and buys a third, not the order of magnitude the KLD gate needs. The
kernel therefore decodes the GGUF's own blocks, and the fill becomes a
byte copy of them:

* **IQ4_NL** (the `down` experts; 4.5 bpw): per 32 values, one f16 scale
  `d` and 16 bytes of nibbles; value = `d * kvalues_iq4nl[nibble]` where
  `kvalues_iq4nl` is a fixed 16-entry signed-int8 table (`code`:
  `ggml-quants.c dequantize_row_iq4_nl`, `ggml-common.h block_iq4_nl`).
  In §2.3's inner loop this is the u4 path with the affine `(nibble -
  zero_point) * scale` replaced by a 16-entry LUT and a per-32 scale — a
  sub-group-constant table lookup, no zero point.
* **IQ3_XXS** (the `gate` / `up` experts; 3.06 bpw): per 256 values, one
  f16 scale `d`, 64 bytes of 8-bit grid indices (each selecting 8 values
  of a fixed 256-entry grid, `iq3xxs_grid`), and 8 × uint32 of
  scales-and-signs — per 32-value sub-block four 7-bit sign-mask indices
  (`ksigns_iq2xs`, 8 signs each) and a 4-bit scale; value = `d * (0.5 +
  scale4) * 0.5 * grid[idx][j] * sign` (`code`: `dequantize_row_iq3_xxs`).
  In the kernel: two constant tables (the 256 × 8 grid, the 128 sign
  masks) in `__constant` memory, one grid read per 8 values, the sign as a
  select.

Both keep §2.3's contract that no dequantised tensor is written to memory,
and both remove the fill's re-quantisation entirely: the expert bodies in
the artifact become the GGUF's block bytes, expert-major, which is also
the slot format of the GPU cache and the host tier (§3, §4 — the host
kernel decodes the same bytes; patch 0011's `dequant_weight` grows the two
decoders). The byte budget per expert-layer moves from 2,457,600 B (u4 +
f16 scales at 128) to the checkpoint's own 3.06 / 4.5 bpw — smaller for
gate/up, larger for down; the Fit table (§5.1) is recomputed at that point.

Red first, on the record before the kernel: (1) a host decoder for each
format pinned bit for bit against llama.cpp's `dequantize_row_*` on real
blocks of the shipped shards (the analogue of `test_expert_fill`'s
executed-piece cells, with `gguf_feed`'s exact dequant as the oracle);
(2) the per-expert kernel's harness cell fed with block bytes whose
decode is NOT alike across pages (the blind-fill lesson of 2026-09-08);
(3) the depth-4 cut ladder against llama.cpp's whole tensors reading
layer 3 at ≥ 0.9999 where the u4 repack reads 0.9987; (4) the KLD gate.

### §2.3b — how the native blocks reach the op (decided 2026-09-18)

Read from the plugin source (`code`, the pinned tree on the dev host):

* `ov::op::internal::MOECompressed::Config` (`ov_ops/moe_compressed.hpp`:
  hidden_size, inter_size, num_expert, top_k, group_size, has_zp,
  out_type) is where a `weight_format` field lives (gate/up and down
  separately: the checkpoint mixes IQ3_XXS and IQ4_NL).
* The offload copy path is format-agnostic already:
  `expert_tensor_span` sizes one expert as the tensor's bytes divided by
  the expert count and `fill_weights_memory` copies nine per-expert
  tensors by name — gate/up/down weights, scales, zero-points. Nine
  tensors carry the native formats without new plumbing: for IQ4_NL the
  nibble tensor and the per-32 scales (zero-points unused); for IQ3_XXS the
  grid-index tensor, the per-32 scales (d·(0.5 + s)·0.5, precomputed at
  fill in f32 — the checkpoint's own numbers, no re-quantisation) and the
  sign-mask index tensor in the zero-point slot.
* The CPU tier (`moe_cpu_expert.cpp`) decodes per element inside a
  per-group loop; a native row decodes into a per-row f32 scratch of `ic`
  values first (the tier is bandwidth-bound, the extra pass is free), with
  the two decoders of `gguf_dequant.cpp` ported.
* The fused kernels select by weight type (u4/i4/u8) and a group size; a
  native format is refused there until its OpenCL decode exists, so the
  first served form runs every expert on the tier.

The GRAPH SIDE decides the order of work. The plugin's tiled matcher
(`convert_tiled_moe_block_to_gather_matmuls.cpp`) recognises a
`CompressedWeightsBlock` — a u4/i4/u8 Constant through Convert, an optional
Subtract of a zero-point and a Multiply by a scale — and nothing else; a
raw block tensor cannot be matched, and the emitter (python) cannot
construct the internal op. So the emitter expresses the native decode in
STANDARD ops over the checkpoint's own bytes re-laid per tensor:

* IQ4_NL: `Gather(table[16] as f16, Convert(nibbles u4 → i32))` × per-32
  scale — the affine chain with the Subtract replaced by a table Gather;
* IQ3_XXS: `Gather(grid[256×8] → magnitudes)` reshaped to the row,
  `Gather(signs[128], sign_index)` → `BitwiseAnd` with the eight masks →
  a sign of ±1, × per-32 scale.

Both are shape-valid, exact in f32, and RUN ON THE CPU PLUGIN as they are —
the device-free oracle of the suite — and on the card as generic ops in the
unfused form (every expert computed, the pre-fusion residency), which is
enough for the depth-4 ladder against llama.cpp's whole tensors without a
plugin patch. The plugin patch then adds the two `CompressedWeightsBlock`
variants to the matcher, `weight_format` to the config, the tier's row
decoders and, last, the OpenCL decode in the fused and per-expert kernels.
The fill becomes a re-layout of the GGUF's block bytes into these tensors
(nibbles, grid indices, sign indices, per-block scales) — exact, and
pinned against `gguf_dequant.cpp`'s decoders on the real shards.

A depth-4 llama.cpp reference by `--override-kv qwen4exp.block_count` is
NOT available (`measured-here`: the loader refuses, `compress_ratios` is an
array of 48); the depth-4 acceptance stays the whole-tensor comparison at
`layer3/out` (≥ 0.9999, the u4 repack reads 0.9987).

### §2.3c — the native formats in the fused op's own layout (2026-09-18)

`MOECompressed` (`ov_ops/moe_compressed.hpp`, `code`) takes each expert
weight as rank-4 `[E, out, groups, group_size]` with a scale
`[E, out, groups, 1]` and an optional zero-point `[E, out, groups, 1]`;
`validate_and_infer_types` checks K = groups × group_size against the
scale's group count. Both native formats fit that layout at group 32
without any re-layout of the per-role arrays:

| format | weight slot | scale slot | zero-point slot | decode |
|---|---|---|---|---|
| IQ4_NL (down) | u4 codes `[E, out, K/32, 32]`, linear element order | f32 `d` `[E, out, K/32, 1]` | none (`has_zp = false`) | `d · T[code]`, T the 16-entry table |
| IQ3_XXS (gate, up) | u8 grid indices `[E, out, K/32, 8]` (4 values each) | f32 `d·(0.5+s)·0.5` `[E, out, K/32, 1]` | u8 sign indices `[E, out, K/32, 4]` (8 signs each) | `scale · grid[idx][j] · sign` |

The emitter's Constants (`_native_expert`, tree after 8ae3d15) are these
shapes exactly, so the plugin patch's matcher lowers them as they are, the
offload path copies one expert's bytes as it does today (tensor bytes ÷
experts, nine slots), and a `weight_format` per projection in
`MOECompressed::Config` (visited as an attribute, so it serialises) tells
the kernels how to read the three slots. The validation relaxes at one
place under a native format: the weight's last dimension (8 for IQ3_XXS,
not the group size); the zero-point check (`check_zp`) only pins the
element type against `has_zp` and never looked at the shape, so the
`[E, out, K/32, 4]` sign indices pass it as it is (`code`, corrected after
review 2026-09-18: the first draft of this note promised two). Per expert-layer the
bytes are 2 × (640 × 80 × 8 + 640 × 80 × 4 + 640 × 80 × 4) + 2560 × 20 ×
16 + 2560 × 20 × 4 = 1,638,400 + 1,024,000 = 2,662,400 B — 8% more than
the u4 repack's 2,457,600 B (the f32 per-32 scales; f16 scales would put
it at 2,355,200 B), the Fit table row to recompute.

Order of the plugin work, each a patch on the series: (1) the graph side
— a `NativeExpertBlock` pattern for the two chains beside
`CompressedWeightsBlock`, the chain-to-compressed-GatherMatmul conversion
carrying the format, `weight_format` in the config with the two relaxed
checks; (2) the tier — every routed expert through the CPU tier under a
native format (patch 0038's mechanism, at any offload ratio; a ratio of
100 is "disabled" in `prepare_moe_otd_params`, so the resident set is
never empty and the fused kernels must refuse the format until they can
read it) with the two row decoders of `gguf_dequant.cpp` in
`moe_cpu_expert.cpp`; (3) the OpenCL decode in the fused and per-expert
kernels. After (1)+(2) the depth-4 native artifact runs on the card
through the boot driver's cut ladder and the served binary serves the
model from the tier alone — the first native serve, and the KLD gate's
first native reading.

### §2.3d — what the checkpoint actually ships, and the scale's precision (2026-09-18)

The two-format premise of §2.3a–c was read off `blk.0` and `blk.24`. The
first native depth-4 export refused layer 2, and the census of all 48
layers (`measured-here`, every `ffn_*_exps` tensor's GGML type read from
the shards) is:

| layers | gate / up | down |
|---|---|---|
| 0, 1, 3, 5–29, 31–45 (43 layers) | IQ3_XXS | IQ4_NL |
| 2 | IQ4_XS | Q8_0 |
| 4, 30, 46, 47 | IQ3_XXS | Q8_0 |

Two more formats, both fitting the group-32 layout (`code`, ggml-quants.c
`dequantize_row_iq4_xs` / `dequantize_row_q8_0`, pinned clone 56b9eb28):

| format | weight slot | scale slot | zero-point slot | decode |
|---|---|---|---|---|
| IQ4_XS (gate, up of layer 2) | u4 codes `[E, out, K/32, 32]` — **the IQ4_NL layout**: a 256-value block's nibbles index the same 16-entry table in the same order | f16 `d·(ls−32)`, the 6-bit sub-block scale folded in by the exporter (`native_blocks.iq4_xs_split`) | the scale again | the IQ4_NL decode, unchanged |
| Q8_0 (down of 5 layers) | i8 codes `[E, out, K/32, 32]` | f16 `d` | the scale again | `d · q` (plugin format 3, `NativeQ8WeightsBlock`, a third row decoder in the tier) |

So the plugin knows three decodes (IQ4_NL-table, IQ3_XXS-grid, Q8_0)
for four checkpoint formats; the config's `weight_format` names the
decode, and `serving_shape` keeps the provenance (`IQ4_XS` in the census,
the chain is the IQ4_NL one).

**The block scale is an f16 Constant, not f32** (`measured-here`, GPU.0,
2026-09-18, first attempt of the lowering cell): an f32 scale Constant
under the fused op is wrapped in a `Convert(f16)` by the plugin's
`ConvertPrecision` (transformations_pipeline.cpp: the native pass runs at
line ~671, the precision pass at ~777), the offload series cannot fold it
(file-backed Constants), and the op translation (`ops/moe.cpp:65`) refuses
a non-Constant input. Every stock scale is f16, so the native one is too,
behind a `Convert(f32)` in the emitter's chain that the pattern blocks
accept as optional. Exactness: IQ4_NL's and Q8_0's `d` IS an f16, so those
stay exact; IQ3_XXS's `d·(0.5+s)·0.5` and IQ4_XS's `d·(ls−32)` round once
to f16, ≤ 2⁻¹¹ relative per block (asserted per random block in
`test_native_expert_chain`, and the real layer-0 / layer-2 experts through
the chain against gguf-py within that bound plus f32 summation). Against
the 0.10–0.13 relative RMS of the u4 repack this is three orders of
magnitude below; the f16 scales also put the per expert-layer bytes at the
2,355,200 B of the §2.3c note, not 2,662,400 B.

The lowering itself fired on the card (the compile reached the plugin's
op translation with a `MOECompressed` of the native inputs, first
attempt). The second attempt, with f16 scales, never got to run the pass's
output: the B60 wedged at the first job of the process (kernel-owned
queue, "not started", GuC reset cascade, then a NULL dereference in
`xe_sched_job_set_error` inside the driver's own timeout path — the xe
DKMS build `xe-ringorder/7.0.14+p1` on kernel 7.0.14-12-pve). That is an
observation of the host, not a measurement of the native path: the
stock-affine control cell through the same harness (the `affine` parameter
of `test_native_lowering_gpu.py`) is the first thing to run when a card is
back, before any native cell — a wedge that reproduces on the control is
the harness or the driver; one that does not is the native path's, and
then the tier-only execution (`_native_tier_only`, batched-GEMV path with
every expert a sentinel) is where to look.

### §2.4 — kernel technology

The plugin's own OCL kernel infrastructure (`moe_3gemm_swiglu_mlp.cl` and
its micro-GEMM path) is the implementation target. The per-expert kernel is
a new `.cl` file (or an extension of the existing MLP kernel) that:

- Takes a single expert's weight pointers instead of the stacked
  `[num_slots, …]` tensor
- Reads packed u4 bytes and dequants in the subgroup work-items
- Uses the same subgroup size (32 on A770 Xe-HPG / B60 Xe2) and tiling
  as the existing GEMV path
- Handles both GEMV (decode, 1 token) and GEMM (prefill, batched) via a
  token-count parameter

The oneDNN GEMM path is not used for the per-expert kernel. oneDNN's type
table includes `{u4, i4, u8, i8}` (DESIGN §7.0.2ah), but it has no
packed-u4-with-group-dequant GEMV — the in-kernel dequant (per-group
scale and zero-point applied inside the K-loop) requires a custom kernel,
not an oneDNN matmul with a u4 descriptor. The kernel is a direct OCL
dispatch through the plugin's `ocl::typed_primitive_impl` infrastructure.

### §2.5 — dispatch logic

At MoE layer execution time, when the per-expert kernel is active:

1. Read the router's top-k output: `expert_ids[n_tokens, top_k]` and
   `gating_weights[n_tokens, top_k]`.
2. For each unique expert in the batch:
   a. Look up the GPU LRU cache (§3). If resident → device kernel.
   b. If not resident → host miss tier (§4).
3. Device-resident experts: batch their tokens and dispatch the per-expert
   OCL kernel. One kernel launch per expert (not per token) — the kernel
   handles the token batch internally.
4. Host-tier experts: dispatch to `moe_cpu_expert` (patch 0011), the
   existing AVX2/scalar host GEMV. FreeToken's `q*` split (§4) caps the
   per-step fetch count.
5. Accumulate all expert outputs into the layer output tensor via the
   existing `index_add` / `scatter_reduce` path.

This replaces the fusion's path entirely for the offload case. The fusion
still runs for the non-offload (fully-resident Constants) case.

**The red-first cells the campaign requires:**
- A cell proving the fused path is refused when the per-expert kernel is
  available (the fusion matcher does not fire; the per-expert dispatch does).
- A cell proving an unrouted expert is never computed (the kernel launch
  count equals the number of unique routed experts, not `num_expert`).

---

## §3 — GPU LRU expert cache

### §3.1 — what exists

Patches 0005–0007 ship a device-resident slot pool with async upload:
`expert_slot_bytes(num_expert, ratio_pct, per_expert_bytes, moe_layers)`
(fit.h) sizes it, the plateau probe measures actual device residency, and
the LRU cache (patch 0012) manages eviction. The slot pool holds expert
weights in the plugin's own constant format — the stacked tensor the fusion
consumes.

For Flash-Next the slot pool must change shape:
- 512 experts per layer, 48 layers
- Per-expert slot: 2,457,600 bytes (int4)
- Full pool: 56.25 GiB — no card holds it
- At ~24 GiB resident (~42% of pool): ~217 slots/layer, 95.0% hit rate
  (WP6b table, `measured-here` LRU replay; the per-layer replay tool
  reproduces 94.4% on the same trace — within the ~1.4-point spread the
  cache-model correction in `design-qwen-flash-next.md` records)

### §3.2 — the extended cache

The GPU LRU cache is a per-layer slot table indexed by `(layer, expert_id)`,
keyed `l·E+e` as FreeToken does (`code`: `offload_cache.py`,
`_BANK_SCHEMAS`). Each slot holds one expert's three weight matrices
(gate/up/down) plus their group scales and zero-points, in packed u4 form.
No widening: the bytes on disk are the bytes in the slot.

**Eviction policy**: LRU by last-routing time, the same policy the existing
`moe_lru_cache` (patch 0012) implements. The per-layer LRU model is
retained (each layer manages its own slots independently) — the WP6b fit
study's hit-rate table is calibrated against this model, and a global LRU
reads a materially more optimistic hit rate on the same trace
(`docs/serving-config-flash-next.md`).

**Slot allocation**: the VRAM share of the resident pool is sized by the
fit pass (the existing `expert_slot_bytes` arithmetic, unchanged). The DRAM
share is host-mapped USM memory (`usm_host` on the plugin's allocator) that
the per-expert kernel can read directly over PCIe — the same mechanism
patch 0017's readback decomposition uses for the existing host tier.

**Upload path**: on a cache miss, the expert's packed bytes are read from
the host pool (DRAM-resident mmap of the GGUF expert shards, or NVMe-backed
mmap with page faults) and uploaded to a VRAM slot via the existing async
upload ring (patches 0005–0006). The upload is one contiguous copy of
2,457,600 bytes — at the A770's measured H2D bandwidth (~1.8 GB/s, PCIe 3.0
x4, `docs/design-qwen-flash-next.md` WP2 / `project-a770-chipset-link.md`)
that is ~1.3 ms per expert. At 95% hit (24 misses per token across 48
layers), uploading ALL misses costs ~31 ms — the ceiling when every miss
is fetched to VRAM. Under the `q*` split (§4), some misses go to the
host CPU tier instead of uploading, so the actual GPU stall is less. At
the B60's PCIe 4.0 x8 (~14.3 GB/s): ~0.17 ms per expert.

### §3.3 — §3.4 invariant: deterministic cache order

DESIGN §3.4 requires history-independent greedy output: the served answer
must not depend on which experts happen to be resident. FreeToken preserves
this with a deterministic cache-fill order (`code`:
`offload_cache.py:ensure_experts` fills in the router's own id order).

arcint's mechanism: on a cold start, the cache is empty. The first
forward's router selects its top-k; the cache fills in expert-id order
within each layer (ascending id, breaking ties deterministically). Every
subsequent forward evicts and fills in the same order. Because the router
is a pure function of (hidden state, layer), and the hidden state depends
only on (seed, prompt, prior tokens) — never on cache state — the set of
experts routed at each step is identical across cold starts, and the fill
order is deterministic from the routing order.

The static partition's existing `splitmix64(seed, layer_key, expert)`
ranking (patch 0018) is a separate mechanism for the partially-offloaded
case and does not apply here — the LRU cache is the policy for the
streaming case where no expert is guaranteed resident.

---

## §4 — host miss-tier integration

### §4.1 — FreeToken's `q*` split, adapted

FreeToken's `ensure_experts_hybrid` (`code`: `offload_cache.py:855`) caps
the number of misses fetched to the GPU per decode step at
`hybrid_max_fetch`. Overflow misses get slot id `−1` and are computed on
the CPU. This bandwidth-adaptive split avoids stalling the GPU pipeline on
slow uploads.

arcint's adaptation uses the same structure:

1. The router selects `top_k` experts per token (10 for Flash-Next).
2. The GPU LRU cache is consulted. Resident experts → device kernel.
3. Of the non-resident experts, up to `max_gpu_fetch` are uploaded to
   newly-evicted slots and computed on the device after upload completes.
4. The remainder (if any) are dispatched to the existing `moe_cpu_expert`
   host kernel (patch 0011, AVX2/scalar GEMV on the host CPU).

`max_gpu_fetch` is sized from the measured upload bandwidth and the decode
latency budget. At the A770's ~1.8 GB/s and a 60 ms/token target:

    budget_bytes = 1.8 GB/s × 0.060 s = 108 MB
    max_experts  = 108 MB / 2.34 MB/expert ≈ 46

At 95% hit (24 misses per token across 48 layers, ~0.5 per layer), all
misses fit in the upload budget with room to spare. The cap matters at
lower hit rates or during cache warm-up.

### §4.2 — host kernel: the existing `moe_cpu_expert`

Patch 0011 ships the host-side per-expert kernel:
- Mmap weight accessor (`ParallelWeightReader::mapped()`)
- AVX2 GEMV with the same u4 dequant (derived nibble order, patch 0011
  header)
- Scalar fallback for non-AVX2 hosts
- Persistent worker pool (arcint's `--moe-cpu-tier-threads`, default =
  physical cores)

The host kernel already handles the "miss" case in the decode-time tier
split (patch 0012). For Flash-Next, it handles the overflow from the `q*`
split above: experts whose upload would exceed the per-step budget.

### §4.3 — prefill path

The hybrid prefill split (patch 0037, campaign: `static-partition-prefill`)
already implements the device/host dispatch for prefill:
- Resident experts → grouped-GEMM (the existing batched path)
- Non-resident → `exec_prefill_onednn` in `host_only` mode

For the per-expert kernel regime, the prefill path follows the same split
but uses the per-expert OCL kernel instead of the grouped-GEMM for the
resident subset. The host dispatch for non-resident experts is unchanged.

---

## §5 — Flash-Next specifics

### §5.1 — dimensions (from the served artifact and config)

| | value | source |
|---|---|---|
| `hidden_size` | 2560 | config |
| `moe_intermediate_size` | 640 | config |
| `num_experts` | 512 | config |
| `num_experts_per_tok` | 10 | config |
| MoE layers | 48 | config |
| per-expert-layer bytes (int4) | 2,457,600 (2.34 MiB) | `3 × hidden × moe_intermediate × 0.5`, `measured-here` |
| full expert pool | 56.25 GiB | `512 × 2,457,600 × 48` |
| PLE table | 26.82 GiB | `measured-here`, must be DRAM-resident |

### §5.2 — projected performance (from WP6b, all bandwidth-bound projections)

| resident pool | % of 56.25 GiB | hit % | t/s (NVMe 1.68 GiB/s miss) |
|---|---|---|---|
| 16 GiB | 28% | 89.5% | 11 |
| 24 GiB | 43% | 95.0% | 18 |
| 32 GiB | 57% | 97.1% | 23 |
| 40 GiB | 71% | 98.0% | 27 |

These are WP6b's bandwidth-bound projections at MTP amortization 1× (the
shipped GGUF has no MTP head). With MTP amortization (the acquired head,
WP8b): 30–40 t/s projected at the 95% hit point.

### §5.3 — per-token work

Per token, per layer: 10 expert invocations (top-k = 10). Per expert:
- gate GEMV: 640 × 2560 = 1,638,400 MACs
- up GEMV: 640 × 2560 = 1,638,400 MACs
- SiLU + elementwise multiply: 640 elements
- down GEMV: 2560 × 640 = 1,638,400 MACs
- Total: 4,915,200 MACs per expert per layer

Per token: 10 × 48 × 4,915,200 = 2.36 GMACs — the work that replaces the
fusion's 512 × 48 × 4,915,200 = 120.8 GMACs (51.2× reduction).

---

## §6 — integration checklist

1. **New OCL kernel file** — the per-expert SwiGLU GEMV/GEMM with in-kernel
   u4 dequant. Inputs: expert weight pointers (gate/up/down + scales/zp),
   hidden state, token count. Output: expert contribution tensor.
2. **Dispatch branch** in `moe_3gemm_swiglu_opt.cpp` — when
   `is_offloaded()` and expert count exceeds resident slots: read top-k,
   consult cache, dispatch per-expert kernel for residents, host tier for
   misses.
3. **Cache extension** — the slot pool (patches 0005–0007) extended for
   packed u4 expert weights addressed per-expert (not per stacked constant).
   The `OffloadExpertWeightProvider`'s `try_acquire_simultaneous` API
   already returns per-expert slots; the cache manager needs the new
   per-expert kernel as its consumer instead of the fusion.
4. **Red-first tests** — (a) fusion refused when per-expert kernel active;
   (b) unrouted expert never computed (kernel launch count assertion).
5. **Perf counters** — per-expert kernel invocations, cache hits/misses per
   layer, host-tier dispatches (extend `OtdPerfCounters`).
6. **Fit pass** — `expert_slot_bytes` unchanged; the per-expert kernel
   consumes the same byte budget as the fusion's slot pool.

---

## §7 — what is NOT designed here

- **The MoE fusion matcher** — bypassed, not patched. The fusion still
  exists and still works for the fully-resident case.
- **NVMe direct expert fetch** — campaign `nvme-direct-expert-tier`,
  blocked on an ext4 expert store. The miss tier here is DRAM (host pool)
  or NVMe-backed mmap page faults.
- **Host-side K-quant native compute** — campaign `kquant-host-storage`.
  The host tier here uses the existing int4 dequant, not K-quant blocks.
- **Sub-4-bit resident format** — the campaign measures int4 vs int3 as a
  cache-headroom lever after the kernel works at u4. The kernel's dequant
  logic is parameterised by group size and element width so int3 is a
  format addition, not a kernel rewrite.
- **Partition seeding** — campaign `partition-seeding`. The LRU cache here
  does not seed from a histogram; it warms by demand on the first forward.

---

## §8 — evidence classes

Every claim in this document carries its evidence class:

| claim | class | source |
|---|---|---|
| MoE fusion computes all experts per token | `measured-here` | window-051 B.3 |
| 7.06× device residency, 623× warm forward for ports | `measured-here` | window-051 B.3 |
| 51.2× expert work at 10 of 512 | `measured-here` | config + B.3 |
| FreeToken per-expert residency, in-kernel dequant, `q*` split | `code` | `~/src/FreeToken-ref` |
| per-expert-layer slot = 2,457,600 B (Flash-Next int4) | `measured-here` | Fit table |
| full pool = 56.25 GiB | `measured-here` | Fit table |
| LRU hit rates at various resident capacities | `measured-here` | WP6b, `expert_lru_replay.py` |
| DRAM bandwidth ~44.4 GiB/s | `measured-here` | WP2 |
| NVMe ~1.68 GiB/s in-container | `measured-here` | WP6b |
| A770 H2D ~1.8 GB/s (PCIe 3.0 x4) | `measured-here` | WP2 |
| B60 H2D ~14.3 GB/s (PCIe 4.0 x8) | `measured-here` | WP2 |
| oneDNN has no 4-bit type | `measured-here` | §7.0.2ah recon |
| existing slot pool, CPU tier, static partition | `measured-here` | patches 0005–0019 |
| projected t/s figures | `projection` | WP6b (not a served measurement) |
| kernel size 800–1,500 lines | `HYPOTHESIS` | §7.0.2y |
| decode regression sign from in-kernel dequant | `HYPOTHESIS` | §7.0.3 precedent (u4 KV +63%) |
| u4 repack error 0.10–0.13 (group 128), 0.07–0.08 (group 16) per expert tensor | `measured-here` | 2026-09-18, blk.0 / blk.24 experts, `expert_fill.quantise_group_affine` |
| 0.73 nats KL at depth 48 from that repack, every other block within noise | `measured-here` | `serving-shape-logits.md`, DESIGN §7.0.2bz |
| IQ4_NL / IQ3_XXS block layouts and decode | `code` | llama.cpp `ggml-common.h`, `ggml-quants.c` (pinned clone 56b9eb28) |
