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
