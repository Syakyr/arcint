# Campaigns — one defect, one document, one session's worth of work

`docs/milestone-0.3.0.md` planned 0.3.0 as fourteen milestones in one
document, and its backlog table carried what 0.3.0 did not close as four
rows. That shape has stopped fitting: a backlog row bundles several levers
with different owners, and a session that picks one up has to read the
whole record to find its edges. From 0.3.1 on, every open defect or lever
is its own **campaign**: one document in this directory, self-contained
enough that a fresh session can carry it with nothing but that document
and the DESIGN sections it cites.

## Rules

- **One campaign = one defect or one lever, with its own gate.** If two
  levers have different owners or different gates, they are two campaigns,
  even when one backlog row named both. A campaign that turns out to have
  two problems inside is split, not stretched.
- **The document is sufficient.** It names the measurement that defines
  the defect (with card, depth, precision, configuration and the DESIGN
  section that recorded it), what is known against what is hypothesised,
  the gate, the entry criteria, the files and knobs and counters involved,
  and the acceptance cells that prove it. Nothing operator-local: hosts,
  paths and unit names stay in the git-ignored `*.local.md` files, as
  `CLAUDE.md` says.
- **The gate is on the record before the work starts**, copied from the
  backlog row or the DESIGN section that found the defect, and it is a
  measurement that can fail. A campaign may close as a *verdict* — the
  defect measured and found not worth fixing, or not a defect — but only
  with the measurement that says so.
- **Pipeline, every time:** recon (read the cited sections and the code,
  write down what is actually there) → design note (a short `docs/design-
  <slug>.md` when the change is more than a fix) → red-first
  implementation → one card window at the end, not many along the way →
  review before commit → a DESIGN `§7.0.2x` record and a CHANGELOG line
  when it closes. Reviews are not skippable.
- **Invariants stay non-negotiable:** DESIGN §3.4 (history-independent
  greedy output) and §3.8, the §5 ladder, the measurement discipline in
  `CLAUDE.md`. A campaign that would trade one for a number does not
  close; it records the trade as a finding and stops.
- **Status is a dated log at the bottom of the document**, appended, never
  rewritten. The milestone document's backlog rows point here and are not
  edited further.
- **Releases collect closed campaigns.** A point release ships whatever
  closed since the last tag; no release waits for a campaign, and no
  campaign is started to fill a release.

## Template

    # <slug> — <one-line charter>
    ## The defect, as measured
    ## Known against hypothesised
    ## Gate
    ## Entry criteria
    ## Scope — in / out
    ## Where it lives
    ## Pipeline for this campaign
    ## Invariants
    ## Status

## The campaigns

Ordered by what unblocks what; the order is advice, not a queue.

| campaign | charter | origin | size |
|---|---|---|---|
| [test-ladder-close](test-ladder-close.md) | close the 0.3.1 lead item: fill the acceptance references from the runners' own windows, record the first real run | 0.3.1 lead item, `docs/design-0.3.1-test-ladder.md` | small, closed 2026-09-05 |
| [prefill-fallback-tristate](prefill-fallback-tristate.md) | patch 0018's overloaded `false` return in the per-expert prefill loop becomes a three-way answer | DESIGN §7.0.2ae, patch 0018 header | small, closed 2026-09-05 |
| [static-partition-cold-start](static-partition-cold-start.md) | a cold sequence's first processes pay minutes of warming under the static partition; find the owner, then remove it | DESIGN §7.0.2ai | medium |
| [static-partition-prefill](static-partition-prefill.md) | tier-ON prefill runs at a third of tier OFF because every layer takes the per-expert fallback; a grouped prefill on the resident subset | DESIGN §7.0.2ai | medium–large |
| [partition-seeding](partition-seeding.md) | seed the static partition from a fixed calibration routing histogram so the pinned half is the hot half — only if decode variance survives the two above | DESIGN §7.0.2ai, patch 0013 | medium, conditional |
| [u8i4-prefill-price](u8i4-prefill-price.md) | `--paged-kv u8:i4` costs +7/+25/+72 % prefill time with depth; measure the mechanism, then decide | DESIGN §7.0.2aa, M8 row | small–medium, closed 2026-09-05 (patch 0020: parity) |
| [u8i4-deep-prefill-fault](u8i4-deep-prefill-fault.md) | the out-of-resources fault at deep u8:i4 prefill on the 16 GiB card; the chunk belt mitigates, the plugin-side cause is open | DESIGN §7.0.2ab, §7.0.2ac | medium |
| [direct-submission-fault](direct-submission-fault.md) | the runtime's direct-submission semaphore evicted under VRAM pressure; diagnosed, the kernel-side fix operator-local and unmeasured on the record, not closed | DESIGN §7.0.2ad | small (measurement), external |
| [mtp-cycle-wall](mtp-cycle-wall.md) | MTP never beats plain decoding at depth on the dense agent: a 390 ms cycle against a 130 ms break-even; cut the cycle or record the verdict as final | DESIGN §7.0.2ag | medium |
| [turnstile-wall-time](turnstile-wall-time.md) | the turnstile test orders threads by wall-clock sleeps and flaked once under build load | 0.3.1 window | small, closed 2026-09-05 |
| [pruefstand-cell-remote](pruefstand-cell-remote.md) | the Prüfstand acceptance cell can only skip by name where the harness does not live; make it runnable from the card window | 0.3.1 window | small, closed 2026-09-05 |
| [sub4bit-vram-kernel](sub4bit-vram-kernel.md) | a per-expert GEMM kernel with in-kernel dequant, bypassing the MoE fusion — the mechanism that makes routing-aware expert execution (GPU LRU cache, compute only the routed experts) possible; sub-4-bit precision is one lever for cache headroom, not the gate | M10 re-scope, DESIGN §7.0.2ah, FreeToken (prior art) | large, open: the native formats serve (patch 0043, +p19), the OpenCL decode is the rate lever; the long-context f16 term is FALSIFIED at the IR level (2026-09-19/20), so the quality lever is a main-model f32-execution A/B owed on the B60, and the served-path determinism floor is a per-card defect (DESIGN §7.0.2cb; campaigns/served-prefill-determinism.md). [DATED IN PLACE 2026-09-22: the native per-expert route's rate leg ran — V1 at the ratio-99 VENICE budget, **1.81×** at ratio 75 (`ρ = 0.073`), and **V4 FIRES (RED)** on the dispatch route (the answer depends on the resident seed); see DESIGN §7.0.2ce and the campaign status.] |
| [kv-checkpoint-restore](kv-checkpoint-restore.md) | a restarted server continues a long conversation from an on-disk checkpoint of its KV and GDN state instead of prefilling it again; byte-identical to cold or refused | operator question, 2026-09-05 | medium–large, backlog |
| [nvme-direct-expert-tier](nvme-direct-expert-tier.md) | LISBON's byte path: can the host hop go? arcwell DMAs NVMe->VRAM at 2.91 GB/s but 1.125 ms/expert, so it pays as bulk residency and loses as a miss handler | 0.5.3 LISBON, github.com/marfrit/arcwell v0.0.1 | large, blocked on an ext4 expert store |
| [serving-shape-logits](serving-shape-logits.md) | the served Flash-Next artifact's logits carry no information about the model at any depth (KL 12.4 nats = the uniform floor at depth 48); find the layer, or the fill, that loses it | the full-depth KLD of 2026-09-18, `sub4bit-vram-kernel` status | large, closed 2026-09-19 (the fill fixed, the residual measured to its mechanism, the yardstick replaced by the model's own f32 forward) |
| [kquant-host-storage](kquant-host-storage.md) | the host compute tier computing K-quant blocks natively, so offloaded experts are sub-4-bit on disk and in the host pool — a throughput lever for the host miss tier, not a VRAM gate | M14 extension, DESIGN §7.0.2ah | large |
| [served-prefill-determinism](served-prefill-determinism.md) | the served Flash-Next path is not run-to-run deterministic at long context — two identical forwards differ by KL(A‖B) mean 0.136/0.151 nats and flip the argmax at 10–15 % of positions — so no KLD gate is readable on it; the decided bound sits ~44x below the floor. Warm-up, the GPU/host residency mix, chunking (D4 `--prefill-chunk 0` is WORSE: 0.0955/0.2021) and launch geometry are all exonerated, several from code. It is a **per-card defect**, but the **subgroup width** (16 on xe2/B60 vs 8 on Alchemist/A770) is a **correlate, not the mechanism**: `xe2` *requires* 16 (the one-line pin is dead) and the reduction disassembles to a fixed tree at both widths. The mechanism is **OPEN, narrowed to a WITHIN-KERNEL nondeterminism in the GDN arithmetic on Xe2** (execution level): the JIT is byte-identical, serializing all 233 enqueues changes nothing, the minimal same-shape gemm is clean, launch geometry is static, and the reduction is a fixed tree at both widths. The GDN state digest is stochastic (5 distinct hashes in one process; the first forward reproducible across cold processes) while the conv state is stable. Fingerprint: `dim0 = row 0`, heads [3,5,6,7,10,13,17,22,31,39,41,42,43,47], one f16 ulp, flip count 2423..3924. **The A770 depth-48 served floor is 0** (bit-identical r0↔r1), so BERLIN-001 clause (d) is readable on the A770 as the measurement card; the B60 stays a per-card caveat until the mechanism is found or upstream fixes it (sibling report posted to #38099 `issuecomment-5751935449`). Reproducer handoff in `docs/handoff-served-prefill-determinism.md` | the 2026-09-19 `F_served` leg, the 2026-09-20 A770/force-tier/D4/cut/state-digest arms, `docs/window-051.md` clause (d) and its cut table | medium, open (one card branch decided) |
| [ple-disk-backend](ple-disk-backend.md) | the served path pins the Flash-Next n-gram table as 26.82 GiB of USM host for the process life (`bind_ngram_ports`); the reference ships a **disk** backend as its default (`code`: `ple_disk.py`, `config.py`:32) and arcint's port contract carries the row ids host-side, so the table becomes a bounded per-forward staging buffer (`T x heads x 90 B` — 720 KiB at T=512, H=16) filled by `pread` of only the named rows. The gate is a served window through staged vs pinned, **byte-identical**, with the 26.82 GiB PLE term off the host ledger and the load-time copy gone | the 26.82 GiB pin, `docs/research-freetoken-code.md`:110–132, FIX D; LISBON 0.5.3 | medium, open |
| [expert-hot-set-lru](expert-hot-set-lru.md) | card hot-set + host LRU for expert slots, seeded from a per-token routed-expert census: patch 0013's aggregate histogram has no per-token ordering, so the census trace/histogram **format** is the first deliverable (design note `docs/design-expert-hot-set-lru.md`); the policy (frequency rank + id tie-break, static-partition seed, demand-warm LRU comparand) replaces patch 0018's routing-frequency-free `splitmix64` seed. The native artifact runs every expert on the scalar host tier (patch 0043), so the speed half is blocked on `sub4bit-vram-kernel`'s OpenCL decode; the host-bound baseline is the measured `d48n` 0.5–0.8 t/s; G was **UNPINNED** and the speed row **HELD** until that path exists (operator decision 2026-09-21). [DATED IN PLACE 2026-09-22: the path exists (native per-expert OpenCL decode, patches 0043/0045; served prefix `ov-0047`), so **G is PINNED at 1.10** in `docs/window-052.md` and the speed leg runs. DATED IN PLACE 2026-09-22 (rate leg): the speed leg **ran and returned V1** at the ratio-99 budget (A770 0.556 t/s vs a 0.579 bar; B60 0.555 vs 0.88), the speed row stays **EMPTY**, and the residency sweep puts the win at ratio 75 (**1.81×**).] | 0.5.2 VENICE, `docs/window-052.md`, DESIGN §7.0.2ah/§7.0.2ae | medium, open |
