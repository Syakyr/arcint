# sub4bit-vram-kernel — a per-expert GEMM kernel with in-kernel dequant, bypassing the MoE fusion for routing-aware expert execution

## The defect, as measured

Not a defect: a milestone re-scope, recorded 2026-09-05 (DESIGN §7.0.2ah,
`docs/milestone-0.3.0.md` M10 row), and re-scoped again 2026-09-15 after
the 0.5.1 port-route measurements (window-051 B.3) and the FreeToken code
reading (`docs/research-freetoken-code.md`).

The pinned runtime's MoE fusion (`keep_moe_3gemm_const_precision.cpp`)
computes every expert for every token. Flash-Next activates 10 of 512
experts per token (`num_experts_per_tok: 10`), so the fusion does 51.2×
the work the architecture requires (`measured-here`: window-051 B.3
CORRECTION). FreeToken's reference implementation (`code`: `~/src/
FreeToken-ref`) never computes an unrouted expert — `ensure_experts
(layer_id, expert_ids)` takes the router's own ids — and never
materialises a dequantised weight (dequant inside the GEMM K-loop,
inside the ggml kernels, or by Triton inline-dequant).

The kernel this campaign builds is the mechanism that makes routing-aware
expert execution possible on the plugin: a per-expert GEMV/GEMM that
takes packed weights and dequantises in-kernel, bypassing the MoE fusion
entirely. With it, arcint computes only the routed experts, keeps a GPU
LRU cache of hot expert weights in their packed form, and feeds misses
from the host pool — the FreeToken architecture, on Intel's kernel
library.

Sub-4-bit precision (Q3_K 3.44 bpw, IQ3_XXS 3.06, Q2_K 2.63) is one
lever for more cache headroom — int4→int3 shrinks the expert pool 25 %
(`docs/design-qwen-flash-next.md` WP6b: ~95 % hit → ~97 % at the same
resident capacity) — but the kernel works at u4 first and the gate does
not require sub-4-bit to close.

## Known against hypothesised

Known (`measured-here`): the MoE fusion computes all experts per token
(window-051 B.3; the port route's 7.06× device residency and 623× warm
forward are the cost of that fusion); the streaming fit study projects
~18 t/s on a single A770 with a 24 GiB expert LRU at 95 % hit, ~30–40
t/s with MTP amortisation (`docs/design-qwen-flash-next.md` WP6b);
FreeToken serves 35B MoE at 77–83 t/s on a 32 GB card (`paper`: arXiv
2608.16157). Known (`code`): the FreeToken reference computes only routed
experts, with in-kernel dequant, GPU LRU cache, and a bandwidth-adaptive
CPU/GPU split (`~/src/FreeToken-ref`). Known (`measured-here`): the
matcher's `u4`-only requirement and the oneDNN type table (§7.0.2ah);
K-quant byte counts; NNCF 3.3.0 has `INT3_SYM`/`INT2_SYM` in its mode
enum.

Hypothesised: the kernel's size, "800–1,500 lines" (§7.0.2y); the
decode-regression sign for a per-expert kernel with in-kernel dequant on
Xe2 — unmeasured, with a same-shaped precedent: symmetric u4 KV's
in-kernel dequant costs +63 % on the fused `PagedAttentionExtension`
(§7.0.3), a different kernel and tensor class.

Prior art, surveyed 2026-09-05 and recorded with URLs, licenses and Arc
applicability: `research-sub4bit-weights.md` (same directory). Prior art
after the survey date exists and is not yet on the record.

## Gate

A model with ≥ 512 experts serves on one card with routing-aware expert
execution: only the routed experts computed per token, a GPU LRU cache
holding the hot set, misses fed from the host pool. Prüfstand 10/10;
greedy output byte-identical across two cold starts (§3.4); decode t/s
and hit-rate reported at the reference cell. Sub-4-bit cache headroom
measured as a separate row (int4 vs int3 at the same resident capacity),
win or lose — not a pass/fail gate.

## Entry criteria

Partially met. (1) The recon is on the record (§7.0.2ah) — met. (2) The
FreeToken code reading (`docs/research-freetoken-code.md`) — met
(2026-09-15, on the `qfndev` branch). (3) The streaming fit study
(`docs/design-qwen-flash-next.md` WP6b) — met (2026-09-10). (4) A
routing histogram (patch 0013, `MOE_OTD_ROUTING_HIST`) over a corpus long
enough to characterise cache-hit distributions — unrun. (5) The resident
format (`u3` group-quant vs K-quant blocks), picked by measurement on
one expert layer against the int4 baseline — unrun.

## Scope — in / out

In: the per-expert GEMV/GEMM kernel with in-kernel dequant at u4 as the
first form; the GPU LRU expert cache (the slot pool of patches 0005–0007
extended to hold packed weights, evict by routing frequency, and feed a
per-expert kernel instead of the fused path); the host miss-tier
integration (patch 0011's CPU kernel as the miss handler, FreeToken's
`q*` split as the reference); the routing histogram for cache-hit
characterisation; the sub-4-bit resident format as a measured
cache-headroom lever; Prüfstand and equivalence measurements.

Out: the MoE fusion matcher itself (bypassed, not patched — the fusion
still exists for the dense-resident case); NVMe direct expert fetch
(`nvme-direct-expert-tier`, a separate campaign); host-side K-quant
native compute (`kquant-host-storage`, a throughput lever for the host
miss tier).

## Where it lives

DESIGN §7.0.2ah (re-scope and recon), §7.0.2y (NNCF/K-quant/runtime
findings), §7.0.3 (u4-KV precedent); `docs/milestone-0.3.0.md` M10 row;
`docs/design-qwen-flash-next.md` WP6b (the streaming fit study);
`docs/research-freetoken-code.md` (the FreeToken code reading);
window-051 B.3 (the port-route measurements that killed the fusion
path); `~/src/FreeToken-ref` (the reference implementation); the plugin's
MoE fusion matcher (`keep_moe_3gemm_const_precision.cpp`); the slot pool
(patches 0005–0007); the host CPU tier (patches 0011–0012).

## Pipeline for this campaign

Design note (`docs/design-routing-aware-expert-execution.md`): the
per-expert kernel's contract, the GPU LRU cache's eviction policy, the
host miss-tier integration, the FreeToken `q*` split adapted to this
hardware's measured bandwidths — with the fit study's numbers as input,
not re-derived → the routing histogram over a real corpus (patch 0013)
→ red-first: a cell proving the fused path is refused when the
per-expert kernel is available, and a cell proving an unrouted expert
is never computed → **the kernel work**: the per-expert GEMV/GEMM with
in-kernel dequant at u4, the cache manager, the host split → one card
window: Prüfstand 10/10, decode t/s and hit-rate at the reference cell,
byte-identical cold starts → the sub-4-bit format measurement (int4 vs
int3 on one expert layer, same source): cache-headroom gain reported,
win or lose → review before commit → DESIGN record, CHANGELOG line.

## Invariants

DESIGN §3.4 (history-independent greedy output): the GPU LRU cache
must not make the served answer depend on which experts happen to be
resident — the same invariant patch 0018's static partition enforces
for the existing tier, and the same invariant FreeToken's own
deterministic cache-fill order preserves. Ground rule 2 (a
fusion-impact profile, not a kernel micro-benchmark) applies.

## Status

- 2026-09-05 — opened from the 0.3.1 backlog; nothing started.
- 2026-09-15 — re-scoped. The "VRAM-resident sub-4-bit" framing retired:
  the port-route measurements (window-051 B.3: 7.06× device residency,
  623× warm forward, the MoE fusion computing all 512 experts for every
  token) and the FreeToken code reading showed that the mechanism is
  routing-aware expert execution with a GPU LRU cache, not smaller
  resident weights. The kernel with in-kernel dequant is the same work;
  its purpose is to bypass the fusion and compute only the routed
  experts. Sub-4-bit is one cache-headroom lever, not the gate. Prior
  art after the 2026-09-05 survey exists and is not yet on the record.
- 2026-09-16 — design note committed (169d350); per-expert dispatch framework
  committed (patch 0038, 7dbcadb). Pipeline steps 1 (design note) and 3
  (red-first cells + dispatch) done. The dispatch routes all routed experts
  through the existing CPU tier (patch 0011) with the fused GEMV bypassed
  entirely — proves the dispatch mechanism before the per-expert OCL kernel
  exists. Fable-reviewed: 3 findings fixed (clone field list, offload guard,
  entry assert). Next: the per-expert OCL kernel (pipeline step 4).
- 2026-09-16 — per-expert kernel dispatch integration committed (patch 0040).
  Wires 0039's moe_expert_swiglu.cl into the live dispatch: GPU-resident
  experts launch per-expert kernels (expert_gate_up, expert_down) with slot
  pool weight pointers; non-residents go to CPU tier (patch 0011); fused GEMV
  bypassed via sentinels for all routed experts. Removes 0038's blanket
  sentinel (the proof-of-concept all-CPU-tier redirect). Fable-reviewed:
  clean (0 findings; prior round's 6 findings C1-C4/M1-M2 all addressed).
  Pipeline step 4 (the kernel work) done. Next: one-card window measurement.
- 2026-09-16 — one-card window blocked: host OOM during compile_model,
  all three attempts. (1) --offload-ratio 100 disables partial upload
  entirely (moe_offload_constant.cpp:62, `otd_ratio < 100` boundary),
  every expert constant gets full allocate_memory + memcpy → 152 GiB
  virtual, OOM-killed. (2-3) --offload-ratio 99 enables partial upload
  (~602 MiB expert slot buffers instead of 64.6 GiB), but read_model
  (backend_ov.cpp:2570) still mmaps the full 77 GiB .bin; graph
  construction walks all 17,354 nodes, paging in mmap regions; the
  host (62 GiB RAM, 20 GiB swap) exhausts both → global OOM at 23:17
  (dmesg: pid 637454, total-vm 36 GiB, 610k swap entries, roundhouse
  killed first). The model cannot be compiled on this host at full
  depth without a plugin change to avoid mmapping expert weight regions.
  arcint CLI flag (--moe-per-expert-dispatch) in working tree, not
  tagged. Services restored.
- 2026-09-17 — **the per-expert series is inert on the Flash-Next
  artifact family: the plugin's MoE fusion never matches the
  serving-shape emitter's MoE subgraph** (`measured-here`, B60 = GPU.0,
  22.71 GiB, stock core 2026.4.0-22849 + the p17 plugin series with
  patch 0041 hand-applied, plugin sha 4f881fff…, tools tree = e50148f).
  Instrument: `compile_model` then `get_runtime_model()`, primitive
  types counted off `layerType`, residency off `GPU_MEMORY_STATISTICS`.
  - `qwen38-flash-next-d12-ov` (12 layers, 4,565 IR nodes, expert bodies
    as 36 u4 `Const`), props `OFFLOAD_RATIO=99 MOE_CPU_TIER=YES`:
    1,201 exec nodes, **0 MoE-typed primitives**, 230 `FullyConnected`,
    `usm_device` **17.73 GiB** — the 09-13 stock-plugin figure (c05) to
    the digit. Adding `MOE_PER_EXPERT_DISPATCH=YES`: identical residency
    (17.73 GiB; fdinfo vram0 18.2 GiB, GTT plateau 20.9 GiB), compile
    133 s → 9.6 s (no kernel cache on the host; the speed-up is real and
    unexplained). Partial upload, slot pool, CPU tier, per-expert
    dispatch and 0041 are all keyed on a `MOECompressed` consumer
    (`get_moe_constant_role`, `moe_offload_constant.cpp`) that this IR
    never produces; the 09-16 line "~602 MiB expert slot buffers instead
    of 64.6 GiB" was arithmetic, not a measurement, and is false for this
    family.
  - Control, `qwen36-35b-a3b-int4-ov` (HF export, 40 MoE layers), same
    props without per-expert: `moe_3gemm_fused_compressed` ×40,
    `moe_router_fused` ×40, `usm_device` **1.2 GiB** (from ~17) — the
    fusion and the offload path work where the pattern matches.
  - Control with `MOE_PER_EXPERT_DISPATCH=YES`: `compile_model` fails,
    `clBuildProgram CL_BUILD_PROGRAM_FAILURE` (program_builder.cpp:168) —
    the per-expert OpenCL kernel (0039/0040) has never built on a card;
    "pipeline step 4 done" rested on review, not on a compile.
  - Earlier the same day, two launches of the segmented port-route
    artifact (`…-seg12-ov`, dead since window-051 B.3) took the physical
    host down twice (host thrash, plug pulled); the 115 GiB compile
    footprint was on the record two days before. Host fence changed by
    the operator afterwards: ARC 16 GiB persistent, container 44 GiB.
  - Process slip on the record: the host sampler's watchdog arms only
    on a matched driver pid; the three scratch-script compiles (runtime
    graph dumps) ran without it. MemAvailable never fell below 23 GiB in
    any cell.
  Consequence: before any kernel or residency work continues, the
  artifact has to carry a MoE pattern the plugin fuses — either the
  emitter writes `ov::op::internal::MOE` (or the HF pattern) so the
  whole series applies, or the dispatch hook moves to the
  `FullyConnected` path. That is a design decision, not a window.
  Segmented port-route runtime (306 lines in backend_ov.cpp,
  `load_paged_segmented`) stays uncommitted: its route is dead.
- 2026-09-17, later — **the non-match has a cause, and it is the emitter,
  not the plugin; the (a)/(b) fork above is dissolved.** Dispositions:
  - `code` (plugin source, `convert_tiled_moe_block_to_gather_matmuls.cpp`
    `build_3gemm_pattern`, pinned build 2026.4.0-22849): the tiled matcher
    anchors on `end_reshape` = Reshape(down MatMul) and on `router_reshape`
    = Reshape(Transpose(ScatterElementsUpdate)) → optional Unsqueeze, both
    feeding the router-weight Multiply before the ReduceSum root. The pass
    is registered only under `supports_immad && use_onednn &&
    !moe_disable_fusion` (`transformations_pipeline.cpp`); both cards
    qualify, the 35B control proved it the same day.
  - `code` (this repo): `export_mtp.py:401 moe_block_tiled` carries both
    Reshapes and records (lines 515–531) that 2026.4.0 folds a rank-4
    Reshape whose target dims are all known, so B comes from ShapeOf and S
    is a runtime −1. `serving_shape.py:795 emit_moe_tiled` named that
    function as its source and emitted neither Reshape: down MatMul →
    Multiply, Transpose → Unsqueeze. "Measured to fuse" in its docstring
    was inherited, never re-measured on this emitter.
  - `measured-here` (dev host, CPU only, no card): the constraint walker
    `tools/check_tiled_pattern.py` on the depth-12 artifact's IR — 43
    ReduceSum candidates, 0 matched, all 12 MoE candidates
    `R4.router_reshape.type: observed Transpose, expected Reshape`
    (0.2 s, read_model only). The same walk on the reduced 4-layer
    geometry: 0/4 before the fix, 4/4 after, live and after
    save → read_model.
  - Correction to this document's opening line: "the MoE fusion computes
    every expert for every token" described the UNFUSED graph — the Tile
    over E makes the batched MatMul compute every expert — never the fused
    op. `GatherMatmul` takes the router's `active_indices` (`code`, the
    pass's callback), and the fused kernel's weight provider, slot pool and
    CPU tier all work on the routed set (patches 0005–0012, 0038). Every
    Flash-Next residency and forward figure to date (window-051 B.3's
    7.06×/623×, c05's 17.73 GiB) is the unfused path. The design note's
    §1 already says this; the charter line here did not.
  - Fix: `emit_moe_tiled` now emits both Reshapes exactly as
    `export_mtp.py:532–537` (B from ShapeOf, S = −1). Red-first cell
    `test_every_moe_layer_walks_the_plugins_tiled_3gemm_pattern`
    (tests/python/test_serving_shape.py). A walker PASS is not a compile;
    its docstring lists the blind spots. What proves it is one compile with
    the runtime-graph dump: `moe_3gemm_fused_compressed` ×12 on a
    re-exported depth-12 artifact, residency read off
    `GPU_MEMORY_STATISTICS` — the campaign's next card window, after the
    artifact is re-exported. Until then patch 0041's question and the
    per-expert kernel's build failure stay open behind it.
  - Re-export blocked the same day: the GGUF shards the exporter reads
    are no longer on the dev host (two of three gone with a volume
    re-purposed on 2026-09-15). Route around it for the census:
    `tools/moe_tiled_rewrite.py` inserts the two Reshapes into a pre-fix
    IR in memory before `compile_model`. `measured-here` (dev host, CPU
    only): on the depth-12 IR 12 blocks rewritten, walker 0 → 12, 0.2 s,
    0.08 GiB RSS; on an old-style block the CPU-plugin forward before and
    after is bit-identical. The census window can therefore run on the
    measured artifact; a servable on-disk artifact still needs the
    shards (or a full `save_model` of the rewritten graph).
- 2026-09-17, evening — **the Flash-Next serving-shape artifact fuses, and
  the offload series applies to it.** Three anchors were missing, each
  invisible to the previous check: (1) the two Reshapes (above);
  (2) a ONE-input Swish — the Python binding's `op.swish(x)` appends a
  beta Constant, the pattern declares `Swish({gate_matmul})` with one
  input and the C++ Matcher rejects an argument-count mismatch (`code`:
  `Matcher::match_arguments`; `measured-here`: census 2 with the
  Reshapes alone still 0 MoE primitives; the fusing 35B control carries
  `Swish/opset4 in=1`); (3) for the offload series, the dequant chain in
  f16 with a trailing Convert → f32 — the control's shape — because under
  f16 inference the plugin puts a Convert on an f32 scale Constant
  feeding the fused op and the OTD resolver demands a direct,
  FILE-BACKED Constant (`moe.cpp`: mmap source, weight-sharing buffer or
  an `otd_bin_offset`); an in-memory rewrite therefore fuses on the
  stock plugin (census 2/3) and not with `OFFLOAD_RATIO` (census 3), the
  rewritten graph saved to disk does both (census 4).
  Census ladder (`measured-here`, B60 = GPU.0, depth-12 artifact, KV u8,
  f16, primitive types off `get_runtime_model()`):
  | cell | plugin | props | MoE-typed | FullyConnected | usm_device |
  |---|---|---|---|---|---|
  | unfused (control) | p17+0041 | ratio 99, CPU tier | 0 | 230 | 17.73 GiB |
  | rewrite (Reshapes only) | both | — | 0 | 230 | 17.73 GiB |
  | rewrite (+Swish) | stock | — | 12 + 12 router | 194 | 17.62 GiB |
  | rewrite (+f16 chain), in memory | p17+0041 | ratio 99, CPU tier | compile refused (bin offset) | | |
  | rewritten artifact ON DISK | p17+0041 | ratio 99, CPU tier | **12 + 12 router** | 194 | **3.00 GiB** |
  Host peak ≤ 4.6 GiB in every cell; no watchdog. Tools: `tools/
  moe_tiled_rewrite.py` (pre-fix artifacts), `tools/check_tiled_pattern.py`
  (input counts now checked, matches the 35B control 40/40), the boot
  driver's `--rewrite-tiled-moe` / `--census`, and a device-free oracle:
  the CPU plugin runs the same tiled pass and compiles a matched block
  to three `GatherMatmul` primitives (in the suite). Commits b5946e1,
  481387b, 09daece, 763f044, 0b66c43. Open, in order: a forward on the
  fused offload path (values will differ from the unfused record: other
  kernels, f16 scales), decode at ratio 99, the ratio sweep, then 0041's
  compile-time question at 48 layers and the per-expert kernel's build
  log. A re-export from the GGUF shards replaces the rewritten artifact
  once the shards are back on the dev host.
- 2026-09-17, late — **the fused path served, on both cards, and the
  offload tier faults on the 24 GiB card.** Served binary at 0b66c43
  (registry entry for the rewritten depth-12 artifact, e384c05), the
  +p17 plugin, n-gram shard bound, `measured-here`:
  | card | artifact | offload | first forward | warm decode 64 tok |
  |---|---|---|---|---|
  | B60 | d12r fused | none (17.62 GiB) | OK, deterministic | **80.5 t/s** (unfused rung 09-13: 18.3) |
  | B60 | d12r fused | ratio 99 or 50 + CPU tier | **xe page fault** at the slot-pool probe | — |
  | B60 | d12r fused | ratio 99, no tier | probe OK; requests: "allocated output memory is necessary to set kernel arguments" | — |
  | B60 | 35B control | ratio 99 + tier, +p17 AND +p16 | the same page fault | — |
  | A770 | 35B control | ratio 99 + tier, +p17 | OK, Paris | 16.1 t/s |
  | A770 | d12r fused | ratio 99 + tier | OK, deterministic | **26.6 t/s** (7.7 cold) |
  Dispositions: (1) the fused MoE kernel runs this family and is 4.4× the
  unfused decode at full residency (`measured-here`); (2) the CPU tier's
  first forward faults on the B60 with everything else equal — plugin
  (+p16 without the per-expert series faults too, so 0038–0040 are not
  the cause), binary, artifact, flags — and serves on the A770: a
  card/driver-side fault, `xe … Faulted Address 0x1f0f5e000, Fault
  response: Unsuccessful -ENOENT`, device coredump; the tier had never
  served on the B60 on the record (the 08-30 B60 figures are full
  residency); 30-second reproducer: the 35B at ratio 99 + tier on GPU.0;
  (3) routing-aware execution with 99 % of the experts on the host runs
  the fused depth-12 rung at 26.6 t/s warm on the A770 behind its 1.8
  GB/s link — the campaign's first offload number on a card, and a
  lower bound for the B60 once its tier fault is fixed; (4) the no-tier
  request failure is a runtime binding defect on the served request
  path (the probe path allocates the output the request path does not)
  — open. Values at depth 12 are not the model's; the served France
  prefix matches the unfused record's first four tokens. Next: the B60
  tier fault (driver-side, needs the coredump and a plugin-level
  reproducer), the no-tier binding defect, then the ratio sweep and the
  gate's Prüfstand at full depth once a 48-layer fused artifact exists.
