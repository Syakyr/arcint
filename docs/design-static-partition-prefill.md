# Design note: static-partition-prefill — hybrid grouped-GEMM/host prefill

Campaign: `docs/campaigns/static-partition-prefill.md`.
Gate: tier-ON prefill within 25 % of tier OFF at the reference cell, decode
still at the record, §3.4 continuation-restore and E2 passing.

## The defect (measured, §7.0.2ai)

Under a static half-partition every MoE layer's routed batch contains at
least one non-resident expert. The grouped-GEMM prefill path
(`on_before_prefill` / `build_grouped_mask_otd`) calls
`try_acquire_simultaneous()` without a `cpu_tier_misses` destination. Patch
0018's static-partition branch in `try_acquire_simultaneous` returns
`std::nullopt` on the first non-resident expert (lines 804–808 of the
patch). The nullopt triggers `needs_fallback = true` and increments
`grouped_fallbacks`. The entire layer falls back to `exec_prefill_onednn`'s
per-expert loop.

The per-expert loop already correctly routes resident experts to device
oneDNN kernels and non-resident to `moe_cpu_expert` on the host (patch
0018, bug #2 fix). But it processes experts serially: one gather, one
device-or-host forward, one index\_add per expert. Resident experts that
could run through the fast batched grouped-GEMM are instead running
through per-expert oneDNN kernel launches — 3 launches × ~46 resident
experts × 40 layers = ~5,520 kernel launches vs ~120 for the grouped-GEMM
path.

Evidence class: **code** (patch 0018 source, the dispatch in
`moe_3gemm_swiglu_opt.cpp`), **measured-here** (§7.0.2ai:
`grouped_fallbacks=40×layers`, prefill 26.6 vs 87.4 t/s).

## The fix: wire `cpu_tier_misses` into the grouped-GEMM prefill callers

`try_acquire_simultaneous` already supports `cpu_tier_misses` as an
optional parameter (patch 0018, the `static_partition_active &&
cpu_tier_wants` branch, lines 735–776): when passed, non-resident experts
are added to `cpu_tier_misses` with `kCpuTierSentinelSlot` and the lease
succeeds. The grouped-GEMM callers simply never pass it.

### Changes

**1. `on_before_prefill` (micro-GEMM prefill path):**

When the static partition is active, pass a local
`std::vector<CpuTierMiss>` as `cpu_tier_misses` to
`try_acquire_simultaneous`. The lease now succeeds. In the slot remapping
loop, entries with `kCpuTierSentinelSlot` write a sentinel slot ID
(`static_cast<uint32_t>(config.num_expert)` — one past the last valid
expert/slot) to `batch_mem_ptr` instead of a device slot. Return the
`CpuTierMiss` vector to the caller via an output parameter.

**2. `build_grouped_mask_otd` (grouped-GEMM prefill path):**

Same change: pass `cpu_tier_misses` when the static partition is active.
In the `slot_tokens` loop, skip entries with `kCpuTierSentinelSlot`.
Return the misses via an output parameter. `grouped_fallbacks` is NOT
incremented — the grouped-GEMM proceeds, covering the resident subset.

**3. `get_expert_mask_from_gpu` (CPU mask generation):**

Skip entries where `expert_no >= max_expert_num` instead of throwing.
Track `skipped` count. Change the assertion to
`count + skipped == max_topk * max_tokens`.

**4. GPU mask generation (`prefill_mask_gen` kernel):**

No change needed. The kernel is launched with `num_total_experts` as its
global work size. Each work-item scans `batch_mem_ptr` for entries
matching its ID. The sentinel (`num_expert`, which equals
`num_total_experts` for the non-OTD paths, or exceeds all slot IDs for
OTD) will not match any work-item, so non-resident entries are naturally
excluded from the mask.

**5. `moe_scatter_reduction_opt.cl` (scatter-reduce kernel):**

No change needed. The kernel initialises `start_offset_index` to
`UINT_MAX` and skips any slot whose `start_offset_index` stays at that
sentinel. A non-resident expert's sentinel slot ID won't match any entry
in `experts_ids`, so it is skipped — the kernel accumulates only resident
experts' contributions and writes zero for tokens routed exclusively to
non-resident experts.

**6. Post-grouped-GEMM host dispatch (new code in the dispatch):**

After the grouped-GEMM (micro-GEMM or oneDNN) completes, if
`cpu_tier_misses` is non-empty, run `exec_prefill_onednn` for only the
non-resident experts. That function's `index_add` accumulates on top of
the grouped-GEMM's output, producing the correct total.

Implementation: add `exec_prefill_host_misses`, a cut-down version of
`exec_prefill_onednn` that iterates only over the experts in the miss
list. Each miss carries the expert ID and the flat positions that selected
it. For each missed expert: gather its tokens from scratch.x (the same
layout `moe_cpu_expert` expects), call `moe_cpu_expert`, write back via
`index_add`. This is the same host branch the per-expert fallback loop
already uses (patch 0018, bug #2 fix), extracted into a function that
takes the miss list instead of iterating all experts.

**NOT BUILT (2026-09-16):** patch 0037 calls the existing
`exec_prefill_onednn(host_only=true)` loop for the non-resident subset
instead of the batched `exec_prefill_host_misses` described above. This
extraction is a code-quality cleanup: `exec_prefill_onednn(host_only)`
already skips resident experts (the `is_cpu_tier` check), so the miss-list
walk saves the skip logic, not the work. The serial host compute (~4
non-resident experts per layer × 40 layers × gather/matmul/index_add)
dominates prefill time and is the reason the prefill gate is not met
(§7.0.2bx). The mechanism that would close the gate — overlapping host
dispatch with the GPU grouped-GEMM, or reducing the non-resident set —
is not designed.

**7. Output initialisation:**

Zero the output before the grouped-GEMM. The existing code already does
this when `should_pre_zero_output()` returns true (which it does for OTD —
see the `should_pre_zero_output` method). Verify and assert.

### What does NOT change

- `try_acquire_simultaneous` — already has the right branch; no code change.
- The scatter-reduce kernel — sentinel handling is pre-existing.
- The per-expert fallback (`exec_prefill_onednn`) — still exists as the
  final fallback for non-static-partition OTD or when `grouped_fallbacks`
  fires for other reasons; unchanged.
- The decode path — only prefill is affected.
- §3.4: the split is a pure function of (seed, layer, expert, prompt). A
  resident expert computed by the grouped-GEMM and the same expert computed
  by the per-expert oneDNN kernel both use the same device weights in the
  same slot. A non-resident expert computed by `moe_cpu_expert` uses the
  same host weights as before. The output of each expert is independent of
  the path that computed it — what matters is `(weights, input, routing_
  weight)`, not which kernel did the matmul.

### New perf counter

`hybrid_prefill_layers`: number of layers that took the hybrid path
(grouped-GEMM for resident + host for non-resident) instead of the full
per-expert fallback. Expected: 40 per process under the static partition,
replacing `grouped_fallbacks = 40`.

### Risk

The sentinel slot ID in `batch_mem_ptr` is read by every downstream GPU
kernel. Any kernel that does not bound-check against the activated-expert
list could read garbage. The audit above covers mask generation (GPU and
CPU), scatter-reduce, and gather. The micro-GEMM kernels
(`micro_gemm_up/gate/down`) operate on the gathered buffer (which only
contains tokens from activated experts) and do not read `batch_mem_ptr`
directly. The grouped oneDNN matmuls likewise operate on the gathered
buffer using `grouped_offsets_cpu` (which only contains resident experts'
offsets). No kernel other than scatter-reduce reads the raw `batch_mem_ptr`
after mask generation.

### Patch number

0037, applied on top of 0003–0036. Ships as `marfrit-openvino +p16` (the
next number after the current `+p15`).
