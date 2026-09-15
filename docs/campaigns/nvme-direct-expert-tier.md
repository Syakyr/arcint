# nvme-direct-expert-tier — LISBON's byte path: NVMe controller DMA straight into VRAM, or the verdict that it cannot pay

## The defect, as measured

Not a defect: a lever for **0.5.3 LISBON**, whose feature line is "NVMe miss
tier — cold start with NOTHING prebound; bytes stream NVMe→host→card under the
same residency policy". This campaign asks whether the middle hop can go.

`arcwell` (github.com/marfrit/arcwell, v0.0.1, published 2026-09-15) is a Linux
kernel module that makes an NVMe controller DMA expert weights **straight into
an Intel Arc VRAM buffer object** — no host staging, no page cache, no CPU
touch. Seven ioctls on `/dev/arcwell`; the contract is `stub/include/aw_uapi.h`.

**Evidence class: arcwell's own measurements on its own hardware, NOT arcint's.
Per `CLAUDE.md`'s measurement discipline these are context, never our numbers,
and nothing below may be quoted as an arcint figure.**

| | arcwell | host-fed comparand |
|---|---|---|
| bandwidth, cold | 2.91 GB/s | 1.58 GB/s |
| CPU per GiB | 0.0121 s | 0.1971 s |
| latency, one expert | **1.125 ms** | — |

## Known against hypothesised

**Known, and it is the whole design constraint.** arcwell's own README states
it: "arcwell is a throughput mechanism, not a latency one." The host CPU
computes one expert in **198 µs**; arcwell fetches one in **1.125 ms**. Called
synchronously on the decode path it **loses to doing nothing at all** — 5.7×
worse than simply computing the expert on the host, which is exactly what
arcint's `--moe-cpu-tier` (patches 0011–0012) already does. arcwell's README
cites arcint's own M9 defect as the precedent: 0.4 t/s synchronous, 9.1 t/s
once overlapped, same hardware.

The consequence for this campaign is not a tuning parameter, it is the scope:
**arcwell can only pay as a BULK RESIDENCY mechanism** — cold boot, hot-set
fill, prefetch issued ahead of need through `AW_IOC_SUBMIT_BATCH` /
`AW_IOC_BATCH_WAIT` — and **never as a decode-path miss handler**. On
FreeToken's `q*` crossover (`docs/research-freetoken.md` §3.2, its Equation 4)
a 1.125 ms fetch sits on the losing side of the split for a miss; the CPU
branch wins. That is the same arithmetic, arrived at from the other end.

**Hypothesised, by arcwell itself and unresolved there:** that prefetch
actually hides the 1.125 ms against a real routing trace. Its
`M4_RESULTS.json` keeps `gain_demonstrated: false` deliberately — "a transfer
gain is measured; an inference gain is not" — and it names integration as the
thing that answers it. **arcint is the integrator.** This campaign exists
because that question is ours to close, not arcwell's.

Prior art: `research-hybrid-expert-execution.md` (same directory) for the
CPU/GPU split; `docs/research-freetoken.md` §3.2/§3.3 and
`docs/research-freetoken-code.md` §1 for what the reference does — FreeToken
keeps the CPU-resident expert pool as source of truth and moves only ROUTED
experts, per complete-expert slot.

## Gate

Cold time-to-first-token with NOTHING prebound, arcwell path against the
host-fed path, **same card, same artifact, same residency policy, both arms in
one window**, and the arcwell arm at or below the host-fed arm — with the
prefetch depth that achieves it recorded, because a win at a depth the serving
loop cannot actually reach is not a win. Byte-identity of the served answer
across both arms and across two cold boots (DESIGN §3.4; LISBON-001's own
"restart determinism" row). Decode t/s at the reference cell **must not
regress** in trade for the cold-boot number.

A verdict closes this campaign as legitimately as a landing: if the measured
prefetch depth needed to hide 1.125 ms exceeds what the router gives warning
of, the record says so with the number and LISBON keeps the host hop.

## Entry criteria

None met as of 2026-09-15.

1. **An ext4-backed expert store.** arcwell requires FIEMAP; **ZFS and btrfs do
   not implement it**, and every arcint artifact today lives on ZFS. This was
   already arcwell's own M1 finding (2026-09-13: "no expert file lies on a
   FIEMAP-capable filesystem"). Until expert files exist on ext4 —
   `fallocate`d, one file per expert, which also yields exactly one extent and
   one DMA segment each — no leg here can run. This is the first and largest
   item and it is an artifact-layout change, not a flag.
2. **A card.** Every arcwell number is the Arc Pro B60; its README excludes the
   A770 ("PCIe path goes unresponsive under load"). So this campaign is
   B60-only, which collides with the standing reservation of the other card —
   a seat question for the operator, not a technical one.
3. **The module installed and loadable** on the dev host (DKMS packaging is in
   arcwell's tree). NOT verified from arcint; no arcint session has loaded it.
4. **A consumer.** The residency policy arcwell would feed must exist and be
   the one LISBON means. arcint already ships the per-expert device slot pool
   (patches 0005–0007) and the host CPU tier (0011–0012); RED-C-02 is open on
   whether that pool converges at the admission ratio a large MoE needs
   (`docs/design-qwen-flash-next.md`). **Do not build this tier against a
   residency mechanism whose convergence is unmeasured.**

## Scope — in / out

In: file→LBA translation at load (arcwell's `stub/tools/aw_fiemap.c` is the
reference implementation, and the translation is the caller's once-at-open
job); a dma-buf exported from an xe VRAM BO meeting arcwell's mapping contract
(`NEEDS_VISIBLE_VRAM`, `CPU_CACHING_WC`, size a multiple of 64 KiB, page-aligned
offsets and lengths); batch submit/wait wired so fetches are issued ahead of
need; the cold-TTFT and byte-identity measurements the gate names; the
`AW_IOC_STATS` read that proves no host bounce happened.

Out: any synchronous read on the decode path (the Known section forbids it);
the ext4 expert-store layout itself if that turns into its own artifact-format
campaign; changes to which experts are resident (that is the residency policy's
business — `partition-seeding`, `static-partition-prefill`); the A770.

## Where it lives

`~/src/arcwell` (branch `dev`; `main` carries v0.0.1) — `stub/src/arcwell.c`,
`stub/include/aw_uapi.h`, `docs/USING_ARCWELL.md` (its §4 is the part that
decides client design), `results/` for every number above, `M4_RESULTS.json`
for `gain_demonstrated`. On the arcint side: the slot pool and host tier in
`contrib/packaging/marfrit-openvino/patches/0005-0007` and `0011-0012`,
`src/exec/fit.h` (`expert_slot_bytes`), `src/exec/flash_next_offload.h`.
Roadmap: 0.5.3 LISBON, whose acceptance commit LISBON-001 owns the cold-TTFT,
bounded-RSS and restart-determinism rows this campaign would fill.

## Pipeline for this campaign

Recon (read `docs/USING_ARCWELL.md` §4 and `aw_uapi.h` in full; confirm the
module loads and `AW_IOC_STATS` reports on the dev host) → **entry criterion 1
first**: decide and build the ext4 expert store, as its own step with its own
record → design note (`docs/design-nvme-direct-expert-tier.md`): the prefetch
schedule, where the routing warning comes from, and the fallback when a fetch
has not landed → red-first: a cell proving a synchronous read on the decode
path is REFUSED by arcint's own code, so the losing configuration cannot be
reached by accident → integration behind the existing residency policy → one
card window (B60): cold TTFT both arms, byte-identity, decode non-regression,
`AW_IOC_STATS` pasted → review → DESIGN §7.0.2x record and CHANGELOG line, or
the verdict.

## Invariants

DESIGN §3.4: the served answer is a pure function of the request, never of
which fetches happened to land first — a prefetch that changes output is a
defect, not a race to tune. Byte-identity across two cold boots is LISBON-001's
own row. Decode at the reference cell does not regress in trade. And arcwell's
numbers stay labelled as arcwell's: no arcint claim may cite 2.91 GB/s or
1.125 ms as an arcint measurement.

## Status

- 2026-09-15 — opened. Nothing started. Entry criterion 1 (an ext4 expert
  store) is the blocker and is unowned; criterion 4 (RED-C-02) is open in
  `docs/design-qwen-flash-next.md`. arcwell v0.0.1 published the same day with
  `gain_demonstrated: false`.
