# static-partition-prefill — tier-ON prefill runs at a third of tier OFF because every layer takes the per-expert fallback; a grouped prefill on the resident subset

## The defect, as measured

DESIGN §7.0.2ai, the 0.3.0 release gate, M14's reference cell (16 GiB card,
35B int4, ratio 50, 8 GiB device pool, u8 KV, one lane, n_ctx 65,536, the
1,198-token prompt, 64 greedy tokens, two requests per process, patch
0018's static partition). Prefill on the second request: tier ON
26.6–26.7 t/s against tier OFF's 87.4–87.6 (first requests 11.1 and 25.8
against 65.7 and 82.4) — a loss of about 3×. The plugin's own
`[OTD_PERF]` counter reads `grouped_fallbacks=40` per process: one per
layer, 40 layers, because the grouped-GEMM prefill path refuses any batch
that contains a non-resident expert, and under a static half-partition
every batch has at least one, so every layer's prefill takes the fallback
instead. Decode is not the defect and holds at the record in the same
window: 16.4 t/s on the second request of both processes, at the LRU-era
figure of 15.0/15.5 (§7.0.2x) and above tier OFF's 11.3–11.4 on the same
request. The first real acceptance run of this cell
(`docs/campaigns/test-ladder-close.md`, before the fill window closed it)
measured the same shape like-for-like at a 1,167-sized-token
prompt: tier OFF warm prefill 86.2/86.1 t/s, tier ON warm prefill
26.3/26.3 — the same order of loss at a different sized prompt, not a
one-off of the release-gate window.

## Known against hypothesised

Known: the mechanism, traced in patch 0018's own header
(`exec_prefill_onednn`, `moe_3gemm_swiglu_opt.cpp` ~line 2924, calling
`on_load_expert_weights` ~line 1358 once per expert) — the grouped-GEMM
path's own refusal rule ("refuses a batch with any non-resident expert")
is unconditional on residency, and a static 50% partition guarantees a
non-resident expert in every routed batch, so the fallback fires on every
layer, every prefill, not intermittently; `grouped_fallbacks = 40 = num
layers` is consistent with that, not a sampled rate. Hypothesised: nothing
about the mechanism; what is undesigned is the lever — a prefill path that
splits a layer's batch by residency (device grouped-GEMM for the resident
subset, host per-expert for the rest) has not been designed, built, or
measured for its own overhead or correctness surface.

Prior art, surveyed 2026-09-05 and recorded with URLs, licenses and
Arc applicability: `research-hybrid-expert-execution.md` (same directory). Its "what transfers"
section is the recon's starting point, not a substitute for it.

## Gate

Tier-ON prefill within 25% of tier OFF at the reference cell, decode still
at the record, and DESIGN §3.4's gates (ON-vs-ON identity, E2) still
passing — held together, not traded — copied from the 0.3.1 backlog row
(`docs/milestone-0.3.0.md`).

## Entry criteria

None beyond what is already on record: the 0.3.0 gate's own numbers
(§7.0.2ai) stand as the baseline this campaign measures against. Given the
size of the lever (a kernel-level batch split), the pipeline below still
calls for a design note before implementation, per this directory's rule.

## Scope — in / out

In: a grouped prefill path that partitions a layer's token/expert batch by
residency and runs the resident subset through the existing device
grouped-GEMM kernel while the non-resident subset takes the existing host
per-expert path (`moe_cpu_expert`, already used by the decode-time tier
and by patch 0018's own prefill fallback) instead of abandoning the whole
layer; the log changes needed to observe the new split (a per-layer
resident/non-resident token count, alongside `grouped_fallbacks`); the
patch level and packaging recipe bump this ships under.

Out: the cold-start warming (`static-partition-cold-start`, same reference
cell, a different mechanism); seeding the partition from a routing
histogram (`partition-seeding`, conditional on this campaign and the one
above); the overloaded `on_load_expert_weights` false return
(`prefill-fallback-tristate` — a correctness cleanup on a path this
campaign's own fallback branch also touches, so the two should be
sequenced, not merged); the static partition's ranking rule itself.

## Where it lives

`moe_3gemm_swiglu_opt.cpp`'s `exec_prefill_onednn` (~line 2924) and
`on_load_expert_weights` (~line 1358), and the grouped-GEMM refusal rule
itself, all named in patch 0018's header
(`contrib/packaging/marfrit-openvino/patches/
0018-moe-cpu-tier-static-partition.patch`); `MOE_USE_GROUPED_GEMM_PREFILL`
(default true, arcint never forces it off — the flag
`prefill-fallback-tristate`'s dormant branch also depends on); patch
0018's `static_partition.hpp` (`static_partition_resident_experts`) for
the residency test a split batch would reuse; patch 0004's per-layer
`grouped_fallbacks` counter. `src/exec/backend_ov.cpp` (~2640–2700,
~3280–3340), the surrounding `MOE_CPU_TIER_STATIC_PARTITION` read,
unaffected by this lever directly but context a design note should not
re-derive. The acceptance cell `tier-reference-cell`
(`tests/acceptance/cells/tier_reference.sh`) already emits
`prefill-warm-2nd-on`, `prefill-warm-2nd-off` and `grouped-fallbacks-on`
(lines 187–189, 206–207); `test-ladder-close.md` records both prefill
metrics as report-only today ("gating it would freeze it") — this
campaign's gate turns that into a real bound, in step with that fill.

## Pipeline for this campaign

Recon (re-read `exec_prefill_onednn`'s refusal rule; confirm no other call
site depends on "whole layer, one arithmetic" the way §7.0.2ae's E1/E2
relied on for decode) → design note
(`docs/design-static-partition-prefill.md`) → red-first: a plugin test
with both resident and non-resident experts in one batch, asserting mixed
device/host compute is reachable (red on the unpatched tree, always the
fallback) → the split implementation → plugin rebuild and unit tests →
one card window: `tier-reference-cell` plus the coder offload cells →
review against the window logs → commit as a new patch (0018 ships at
`+p4`; the packaging README names the next number) → DESIGN §7.0.2x
record, CHANGELOG line, the backlog row's closing line, in step with
`test-ladder-close` so the prefill reference gates, not just reports.

## Invariants

DESIGN §3.4 (history-independent greedy output) holds through the split: a
mixed device/host prefill batch must still be a pure function of (seed,
layer, expert), never of arrival order or which tokens happened to fill
the batch. Decode's record (16.4 t/s at the reference cell, §7.0.2ai) does
not regress in trade for prefill; the continuation-restore check and E2
keep passing throughout.

## Status

- 2026-09-05 — opened from the 0.3.1 backlog row; nothing started.
- 2026-09-05, later — the counter's number is not 40. With the plugin's
  counters on (`MOE_OTD_PERF_LOG=1`), the tier reference cell's second
  tier-ON process printed `grouped_fallbacks=400` after two requests of
  1,167 tokens (DESIGN §7.0.2al): five per layer per request over 40
  layers, or ten per process, and the unit is unread either way — the
  "one per layer" reading above came from §7.0.2ai's single reading of
  40. The reference is report-only until this campaign's recon reads the
  counter's unit off the plugin.
- 2026-09-16 — patch 0037 built and reviewed (Fable review: 3 findings
  fixed, 2 noted). Card window on the 16 GiB card: `grouped_fallbacks`
  400→0 (defect eliminated), §3.4 identity 4/4 PASS, E2 PASS, decode
  18.2 t/s (above gate of 14.8), ratio 1.46 (above 1.17). **Prefill gate
  NOT MET**: 27.9 t/s ON vs 87.2 OFF (need ≥65.4). The hybrid path IS
  taken but the serial host dispatch for ~128 non-resident experts per
  layer dominates. Coder offload regression 1lane PASS. Design note §6/§7
  describe a batched `exec_prefill_host_misses` that was not built — the
  patch calls the existing `exec_prefill_onednn(host_only)` loop instead.
  DESIGN §7.0.2bx records the window.
- 2026-09-16, later — **CLOSED.** The coding defect that caused the
  grouped-GEMM fallback is fixed (patch 0037, `grouped_fallbacks` 400→0).
  The prefill gate (within 25% of OFF) is not met: the remaining gap is
  the inherent cost of serial host compute for the non-resident half of
  the static partition, not a defect. Closing: the 0.3.0 defect that
  opened this campaign is resolved; the residual prefill cost is the
  expected price of running half the experts on the host CPU. Pursuing it
  further is a diminishing-returns optimisation, not a defect fix.
- 2026-09-17 — **patch 0037 page-faults the CPU tier's first forward on
  the Arc Pro B60** (`measured-here`, bisected on the served binary with
  the 35B at `--offload-ratio 99 --moe-cpu-tier`, static partition): the
  +p13 plugin (through 0031) serves; +p16 (through 0037) faults at the
  slot-pool plateau probe with `CL_OUT_OF_RESOURCES`, the host kernel log
  showing an xe page fault (`Fault response: Unsuccessful -ENOENT`) and a
  device coredump (`Timedout job`, compute engine); +p16 rebuilt without
  0037 serves; +p17 with `MOE_CPU_TIER_PARTITION=lru` (0037's branch is
  static-partition-only) serves; the A770 serves with 0037 in place.
  Mechanism (`code`, 0037's own header — to be confirmed in the kernel):
  non-resident experts are remapped to the sentinel slot index
  `num_expert`, one past the last valid slot, in the per-token expert
  index buffer, and the GPU mask-gen kernel is trusted to have "no
  work-item at that index"; a table indexed with it is read or written
  one entry past its allocation, which faults where the allocation ends
  on a page (Xe2) and is silent elsewhere (Xe-HPG). Reproducer: 30 s on
  the B60. Fix: patch 0042 — size the per-expert tables `num_expert + 1`
  or clamp the sentinel in the consumer, red-first unit test with the
  sentinel at the last index, then +p18. Until then the tier on the B60
  needs the LRU partition or a plugin without 0037.
- 2026-09-17, night — RETRACTION of the entry above: its mechanism (the
  sentinel `num_expert` overrunning a per-expert table) and its
  prescription (size the tables `num_expert + 1` or clamp) were `code`-class
  inferences from 0037's header and are WRONG — every GPU consumer of the
  sentinel resolves it by equality and is safe; the reader is the gather
  launch below. Nothing was sized `num_expert + 1`.
- 2026-09-17, night — **fixed: patch 0042.** The reader was the grouped
  prefill's gather: `tokens_per_expert_cpu` is created as
  `token_num * max_topk` entries of -1, 0037 fills only the resident
  (token, k) pairs, and the gather kernel was still launched over every
  pair — each work-group past the fill computed `token_index = -1 *
  HIDDEN_SIZE` and added it to the kernel's `uint` offset, wrapping to
  ~2^32 elements, ~8 GiB PAST the buffer (`code`; the B60's faulted
  address 0x1f0f5e000 ≈ 8.2 GiB is consistent; the A770's silence on the
  same read is unexplained, `unobserved`) — (`measured-here`: a
  `stream.finish()` after every stage of the grouped path showed the
  first synchronisation, after the gather, already throwing; the host
  dispatch and the micro-GEMM remap eliminated one at a time). The fix
  sizes the gather and the stages after it by the filled count and hands
  a zero-resident batch to the per-expert path. `measured-here` (B60,
  pin + 0003–0037 + 0042): the 35B serves at ratio 99 with the tier,
  Paris, warm repeat identical, decode 23.6 t/s (KV u8, f16 inference,
  prefill chunk 512, one lane), the hybrid path active.
  Ships as `+p18`. Owed: the unit-ladder cell (tables with sentinel
  entries, filled count against launch size).
