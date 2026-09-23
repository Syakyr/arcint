# window-053 — 0.5.3 LISBON acceptance (LISBON-001: every measured row EMPTY)

Recorded 2026-09-23, before the LISBON card window exists and before the
expert store can be filled correctly at all. This file is the acceptance
commit of 0.5.3 in the form the roadmap's law demands
(`ROADMAP-0.5.x.local.md`:2–6): *the first commit is the acceptance criteria,
with the measured rows EMPTY; prediction commits precede measurement; gates
red-first; counts generated; comparisons to FreeToken pinned ONLY by our own
measured runs.* The commit that fills a row is a measurement commit and pastes
the raw output.

The markers are `docs/window-050.md`'s (`RUN@<sha>`, `RUN@wt+<sha>`,
`RUN@unrecorded`, `DRY`, `UNTESTED`); a row with no marker is EMPTY, and EMPTY
is the honest state of every measured column in this commit. Every disposition
below carries an evidence class (`paper` / `code` / `measured-here`) where it is
a disposition of fact; the acceptance-cell bullets are configuration, not
dispositions. The pinned threshold is arithmetic (`code`), **not**
`measured-here`.

## Feature (roadmap 0.5.3)

NVMe miss tier — cold start with **NOTHING prebound**; bytes stream
NVMe→host→card under the same residency policy. The milestone's second half is
the PLE pin removal: the Flash-Next n-gram table is now a bounded per-forward
staging buffer instead of 26.82 GiB of USM host for the life of the process
(`docs/campaigns/ple-disk-backend.md`, gate PASSED depth-4 on 2026-09-23).

## What LISBON-001 owns

Three acceptance rows, all EMPTY:

1. **cold time-to-first-token, nothing prebound** — the arcwell path
   (NVMe→VRAM direct) against the host-fed path (NVMe→host→VRAM), same card,
   same artifact, same residency policy, **both arms in one window**, arcwell
   at or below host-fed, and an absolute threshold `X` pinned here;
2. **RSS bounded through boot** — the boot child's peak RSS under a stated
   bound, read with the `wait4` child-rusage discipline;
3. **restart determinism** — two cold boots, byte-identical greedy answers in
   digest form.

The three are the campaign gate of `docs/campaigns/nvme-direct-expert-tier.md`
(Gate section) and §7.4 of `docs/design-nvme-direct-expert-tier.md`. This
document does not fill them.

## Scope depth — pinned at depth 4, with full depth as an operator decision

**Pinned: depth 4.** The mechanism under test is the byte path
(NVMe → {host | VRAM} → VRAM) under the static-partition residency policy. That
path is **depth-independent**: one expert slice is 2,457,600 B
(`code`: `src/exec/flash_next_offload.h:45`) at every layer, the partition's
membership is a pure function of configuration fixed at `bind()`
(`code`: patch 0018; patch 0046), and the plugin's slot layout is per layer, not
per depth. The accepted precedent for exactly this shape is the PLE gate, which
ran its staged-vs-pinned served window at **depth 4** because "the n-gram table
is a property of the model, not of the depth"
(`docs/campaigns/ple-disk-backend.md`, 2026-09-23 close). Depth 4 keeps both
arms in ONE window on ONE card, which is the campaign's own requirement.

**Flagged, not assumed: a full-depth (48-layer) variant is an OPERATOR
DECISION.** The document does **not** assume full depth. Two facts make that a
real decision rather than a formality: (a) the depth-4 export's pinned expert
set is a truncated subset, so the *byte volume* the fill moves at depth 4 is
small and the bandwidth claim is not exercised at milestone scale; (b) the
native full-depth artifact is where the 8.38 GB (ratio 86) / 15.10 GB
(ratio 75) pinned fills and the ~35 s cold-forward reads live, but it also
carries the B60 determinism caveat below. The operator decides whether the
gate closes at depth 4 (mechanism) or is repeated at depth 48 (scale); the
arithmetic for the full-depth variant is printed separately in §1 and is
**not** the gate.

## The acceptance cell (fixed before measuring)

- **Card: B60** (`GPU.0`, PCI `8086:e211`). The gate's measurement card is the
  B60 because arcwell's own README excludes the A770; the A770 (`GPU.1`, PCI
  `8086:56a0`) is the determinism-confirmation card only. DRM numbering is
  inverted vs OpenVINO numbering — cards are identified by PCI id, never by
  number (`docs/sop-card-window.md` §2).
- **Artifact: the depth-4 staging-PLE twin** (`qwen38-flash-next-d4s-ov`), the
  correction-matched artifact whose `config.json` is identical to the pinned
  twin and whose only difference is the n-gram port partition
  (`docs/campaigns/ple-disk-backend.md`, 2026-09-23). Using it makes LISBON-001
  one window for both halves and lets row 2 read the freed 26.82 GiB.
- **Residency policy: the static partition, `--offload-ratio 86`** → the
  plugin's integer-division pool is `512*(100−86)/100 =` **71 slots/layer**
  (`code`: patches 0041/0047; `docs/window-052.md` dated correction). The
  design note's prefetch schedule is one batch per layer at this budget
  (`docs/design-nvme-direct-expert-tier.md` §3).
- **Cold state: page cache dropped between arms and boots; nothing prebound**
  (the pinned slots start empty on the host and device).
- **Prompt: the 5-token France prompt** ("The capital of France is", greedy,
  temperature 0), the reference cell's own minimal request.
- **Instrument hygiene: one fresh process per arm**, sampler on the physical
  host started before the leg (`docs/sop-card-window.md` §1), 4 GiB watchdog,
  plugin `ov-0047` (`f021de51b5812ee2`, patches 0003–0047), KV u8, the served
  binary named by sha in the measurement commit.

## 1. Cold TTFT, nothing prebound — EMPTY

**The gate.** Both arms measured in one window on the cell above; the arcwell
arm required to land **at or below** the host-fed arm; and the measured cold
TTFT required to land **at or below `X`**. The prefetch depth that achieves it
is recorded (a win at a depth the serving loop cannot reach is not a win); the
`AW_IOC_STATS` delta proving `via_host_bounce = 0` is pasted.

**The pinned threshold.**

    X = T_boot + T_fill + T_prefill

| term | value | basis | evidence class |
|---|---|---|---|
| `T_boot` | **136 s** | B60 depth-4 served boot to `/props → 200` = 2 min 16 s (`docs/window-050.md` §4.9, 2026-09-13) | `measured-here` (input) |
| `T_fill`, host-fed | **0.387 s** | pinned bytes (697,958,400 B) ÷ the in-container NVMe read rate (~1.68 GiB/s) | `code` arithmetic |
| `T_fill`, arcwell | **0.240 s** | same bytes ÷ arcwell's own 2.91 GB/s — **arcwell's number, labelled, never ours** | `code` over arcwell's `measured-here` |
| `T_prefill` | **3.08 s** | B60 depth-4 cold prefill, 5 tokens (`docs/window-050.md` §4.9) | `measured-here` (input) |

The fill arithmetic, shown explicitly:

    slice            = 2,457,600 B              (code: src/exec/flash_next_offload.h:45)
    slots/layer      = 71                        (code: 512*(100-86)/100, plugin integer division)
    moe layers       = 4 in scope                (the depth-4 artifact)
    pinned bytes     = 71 x 4 x 2,457,600
                     = 697,958,400 B             (code)
    host-fed rate    ~= 1.68 GiB/s ~= 1,803,882,782 B/s
                       (measured-here: the in-container NVMe miss-feed rate,
                        docs/design-qwen-flash-next.md WP6b) = 0.387 s
    arcwell rate     = 2.91 GB/s  (arcwell's own measurement, labelled) = 0.240 s

    X = 136 + 0.387 + 3.08 = 139.467 s   ->   X = 139.5 s

**Evidence class of `X`: `code`** (arithmetic over `code` byte counts and
`measured-here` inputs). `X` is a predictively pinned threshold and is
**never** labelled `measured-here`. arcwell's 2.91 GB/s and 1.125 ms-expert
figures are arcwell's own measurements on arcwell's hardware
(`docs/campaigns/nvme-direct-expert-tier.md`, Known section); they are used
only as a labelled projection and are never quoted as arcint measurements.

**Conservative note, stated not smoothed.** The `T_boot` input is the
*non-staging* depth-4 B60 boot, which paid a 44.1 s full-table bind; the
staging twin does not copy the 26.82 GiB table, so the LISBON cell's boot is
expected **below** 136 s and `X` is an upper bound. The measured value reads
the true term. If a component is wrong, `X` is corrected in place with the
measurement's date — the gate value is not moved silently after the fact.

**Full-depth variant — arithmetic only, NOT the gate, OPERATOR DECISION.**
For the native depth-48 artifact (`qwen38-flash-next-d48n-ov`) at ratio 86 on
the B60: pinned bytes `= 71 × 48 × 2,457,600 = 8,375,500,800 B`; `T_boot` =
34.7 s (model-ready, B60 probe 2026-09-23, `measured-here`); host-fed fill
`4.64 s`; arcwell-labelled fill `2.88 s`; `T_prefill` = 11.11 s (B60 ratio-86
5-token prefill). `X_full(host-fed) = 34.7 + 4.64 + 11.11 = 50.45 s`;
`X_full(arcwell-labelled) = 34.7 + 2.88 + 11.11 = 48.69 s`. This is printed so
the operator can open the scale gate with a pinned number if they choose; it is
**not** the acceptance threshold of this commit and is **not** assumed.

| quantity | predicted (`code`) | measured |
|---|---|---|
| arcwell arm cold TTFT, depth 4 | ≤ host-fed arm; ≤ 139.5 s | **EMPTY (pending measurement)** |
| host-fed arm cold TTFT, depth 4 | ≤ 139.5 s | **EMPTY (pending measurement)** |
| prefetch depth that achieves it | recorded | **EMPTY (pending measurement)** |
| `AW_IOC_STATS` delta (`via_host_bounce`, `max_inflight`, `batches`, `batches_reads`, `segments`) | `via_host_bounce = 0`, `max_inflight > 1` | **EMPTY (pending measurement)** |
| decode t/s at the reference cell, both arms | must not regress vs the measured host-tier baseline (`docs/window-052.md`: 0.5–0.8 t/s B60; 0.526 t/s A770 same-day host comparand) | **EMPTY (pending measurement)** |

**Clause L1** — if the arcwell arm's cold TTFT exceeds the host-fed arm's, the
fill does not pay and the record says so with the prefetch depth that was
reached. **L2** — if the measured arm exceeds `X`, the pin was wrong; the row
names which term missed. **L3** — if the decode t/s regresses at the reference
cell in trade for the cold-boot number, the change does not close.

## 2. RSS bounded through boot — EMPTY

**The bound: the boot child's peak RSS must stay at or below 32 GiB.** That is
the host class the milestone's own charge names: removing the 26.82 GiB PLE pin
"frees ~26.82 GiB of host RAM (making a 32/44 GiB host viable — the class the
external 16 GB/32 GB Flash-Next runs live in)"
(`ROADMAP-0.5.x.local.md`, 0.5.3). A boot that needs more than 32 GiB fails
this row, because it has not delivered the class the PLE half exists to make
reachable. This is the row where the **freed term is read**.

**What reads it.** The `wait4` child-rusage discipline, CF-KEYSTONERSS
(`DESIGN.md` §7.0.2 area; `docs/design-qwen-flash-next.md`): the parent reaps
the boot child with `os.wait4` and reads `ru_maxrss` **for that child**; the
child's own mid-run `getrusage(RUSAGE_SELF)` is kept beside it and asserted
**not to exceed** the `wait4` value. The hole this closes is demonstrated, not
argued: a child that allocates 2 GiB after its self-read reports 0.52 GiB where
the kernel accounts 2.52 — 2.00 GiB invisible to the self-read. In parallel,
the SOP physical-host sampler records `MemAvailable` / `Shmem` / ZFS ARC every
2 s with a 4 GiB watchdog (`docs/sop-card-window.md` §1), because driver /
USM-host memory is charged to the physical host, not the container.

**Arithmetic for the freed term** (`measured-here`, PLE gate, depth 4): the
pinned twin's container `VmRSS` peak was **19.79 GiB** (20,753,068 KB); the
staged arm's was **4.93 GiB** (5,173,232 KB); Δ **14.86 GiB**. The staged
n-gram resident is **2.884 MiB** against the pinned **26.82 GiB USM host**, and
the physical `MemAvailable` minimum rose 9.89 → 32.91 GiB
(`docs/campaigns/ple-disk-backend.md`, 2026-09-23). The depth-4 arithmetic RSS
for the LISBON cell is therefore ≈ 5–7 GiB; the 32 GiB bound is the class
ceiling, not the expected value.

| quantity | bound | measured |
|---|---|---|
| boot child `ru_maxrss`, read by `os.wait4` | ≤ 32 GiB | **EMPTY (pending measurement)** |
| child `RUSAGE_SELF` vs `wait4` | self ≤ wait4 (CF-KEYSTONERSS) | **EMPTY (pending measurement)** |
| physical-host `MemAvailable` minimum under the sampler | > 4 GiB (watchdog) | **EMPTY (pending measurement)** |
| freed PLE term on the ledger | the 26.82 GiB USM-host pin is gone; staging is `T x H x 90 B` | **EMPTY (pending measurement)** |

**Clause L4** — if `ru_maxrss` exceeds 32 GiB, the boot has not reached the
32 GiB host class and the row is RED. **L5** — if the `wait4` value is below
the child's self-read, the instrument is void (the accounting was taken before
teardown); the row is re-run, not passed.

## 3. Restart determinism — EMPTY

**The gate.** Two cold boots of the same served configuration — process
restarted, page cache dropped between them, nothing prebound — the same greedy
requests, **byte-identical answers in digest form** (the digest form of the old
restore witness, DESIGN §3.4: greedy output is a pure function of the request,
never of which fetches happened to land first).

| probe | boot 1 digest | boot 2 digest | identical? |
|---|---|---|---|
| "The capital of France is", greedy | **EMPTY (pending measurement)** | **EMPTY (pending measurement)** | **EMPTY (pending measurement)** |
| the reference cell's prompt, greedy | **EMPTY (pending measurement)** | **EMPTY (pending measurement)** | **EMPTY (pending measurement)** |

A digest that differs between boots is a finding; its mechanism is measured
with the instrument that exists (`tools/boot_serving_shape.py --cut layerN/out
--repeat`), not narrated. The arcwell fill's byte-transparency is separately
enforced at the load barrier by design rule D3 (`docs/design-nvme-direct-
expert-tier.md` §4): a pinned expert whose fetch has not landed is a **load
failure**, never a silent demotion — so a boot's residency set cannot depend on
I/O timing.

**Clause L6** — any digest differs → RED, localise the layer that first
diverges, do not narrate.

## The B60 determinism caveat (operator decision 2026-09-23)

On the B60, **timing and statistics are admissible**; **byte-identity claims
are admissible only where the B60 is known readable**, otherwise they are
paired with an **A770 confirmation** (`8086:56A0` = A770 = `GPU.1`;
`8086:E211` = B60 = `GPU.0`; DRM inverted, identify by PCI id). The reason is
the recorded per-card defect: the depth-48 served path is bit-identical on the
A770 (`F_served = 0`, 0/1367 rows moved) and carries a per-forward within-kernel
nondeterminism on the B60 (KL mean 0.1361/0.1512)
(`docs/campaigns/served-prefill-determinism.md`; `docs/window-051.md` clause
(d)). Row 3's B60 reading therefore pairs with an A770 confirmation of the same
served answer; the arcwell arm itself is B60-only and its byte path is judged by
the load-barrier rule, not by cross-boot byte-identity alone.

## Dependencies — THE GATE IS BLOCKED, NOT MEASUREMENT-READY

The acceptance document must not be read as a runnable measurement plan. Three
things block it today. Every one is recorded, not worked around.

1. **The ext4 expert store is synthetic arcwell test data.** `measured-here`
   (B60 probe 2026-09-23): the store is 1,700 files of 2,457,600 B, each
   resolving to exactly one plain extent — but each file is **one deterministic
   4096-byte block repeated 600×** (600/600 blocks identical, distinct files
   differ only by seed). It carries **no expert tensors and no scales/zp**.
   Nothing can be filled correctly from it today.
   [CLEARED 2026-09-23, artifact-format step: a REAL store now exists on the
   ext4 partition — 3,408 files of 2,457,600 B (`8,375,500,800 B`), every file
   exactly ONE plain extent, `aw_fiemap` byte-verifying all 3,408 against the
   raw device with its `--mutate` leg failing on content, and a 24-expert
   byte-exactness sample exact to the u4 half-step. See the campaign's
   "Artifact-format step" section. The synthetic store itself was NOT touched.]
2. **The scales/zp store-layout precondition is OWED and belongs to the
   artifact-format step.** `code`: the plugin's device slot layout and the
   weight-file layout differ for scales/zp — device `[group][oc]`, file
   `[oc][group]` — and `maybe_transpose_scale_zp` transposes on upload
   (patches 0011:75-90, 0006:291). The weights are `[oc][ic]` in both, so a
   byte-transparent full-slice DMA would land the weights correctly and the
   scales **transposed**. Resolving it (lay files in device order, and adapt
   the host tier's scale indexing, or move the small scale/zp tensors through
   the existing host path) is a **store-layout decision** owed to the
   artifact-format step, not to this gate. No correct fill is claimed.
   [RESOLVED 2026-09-23, artifact-format step: the verdict is **weights-only,
   device order** — the 2,457,600-byte slice is the three u4 weight matrices,
   which are byte-identical file↔device, so a naive full-slice DMA is
   byte-transparent; scales/zp are EXCLUDED from the DMA slice (adding them
   gives 2,553,600 B = 623.4375 pages, not page-aligned, and the plugin's
   per-tensor scale destinations are unaligned too) and stay on the existing
   host path that transposes. The device-order alternative is implemented for
   a future integration and tested. See the campaign's "Artifact-format step"
   section, §2.]
3. **The consumer does not exist.** The `AW_IOC_SUBMIT_BATCH`/`AW_IOC_BATCH_WAIT`
   fill's timing is measured standalone (submit 20.9 ms of a 64.5 ms batch;
   2.71 GB/s at 71 experts; `via_host_bounce` delta 0), but the D2/D3
   integration that runs it inside the serving loop is **OWED**. The
   "serving step with the fill overlapping" has no number.

**Consequence:** the three rows stay **OPEN**. [UPDATED 2026-09-23,
artifact-format step: dependencies 1 and 2 are CLEARED, so the gate is no
longer blocked on "nothing can be filled correctly". It is now blocked ONLY on
dependency 3, the D2/D3 consumer integration, with its three rows still OPEN.]
The gate's own bytes cannot be laid down. This document is the criteria, not a
measurement.

## Harness and tools the measurement will use (named now)

- **served binary** `arcint` + plugin `ov-0047` (`f021de51b5812ee2`, patches
  0003–0047), KV u8, `--moe-cpu-tier`, `--offload-ratio 86`, the pinned
  `--prefill-chunk` and `--n-ctx`; artifact sha pasted at measurement.
- **device-free probe** `tools/boot_serving_shape.py` (`--stage compile` /
  `--stage forward`, `--cut layerN/out`, `--repeat`) for the determinism
  localisation.
- **OTD counters** `MOE_OTD_PERF_LOG` plateau probe (device-byte plateau,
  evictions) and the `[OTD_PERF]` lines (`gpu_hits`, `gpu_misses`,
  `evictions`, `device_slot_buffers`, `host_slot_buffers`).
- **arcwell client** derived from `~/src/arcwell`'s `stub/test/aw_async_test.c`
  (the `aw_fill_budget` shape): `AW_IOC_MAP_BUFFER` with the
  `AW_MAP_F_REQUIRE_P2P` assertion, `AW_IOC_SUBMIT_BATCH` / `AW_IOC_BATCH_WAIT`
  at DEPTH 4, `AW_IOC_STATS` read as a **delta** before/after the leg's work.
- **RSS instrument** `os.wait4` + `ru_maxrss` (CF-KEYSTONERSS), plus the SOP
  physical-host sampler (`docs/sop-card-window.md` §1) with the 4 GiB watchdog.
- **admission ledger** `--fit-ledger-dir` for the fit arithmetic, so the
  load-time probes are skipped on the second matching arm.

Nothing is run in this commit.

## Falsifiable clauses, listed

| # | clause | dies if |
|---|---|---|
| L1 | the arcwell arm's cold TTFT is at or below the host-fed arm | it exceeds the host-fed arm — the fill does not pay |
| L2 | both arms' cold TTFT ≤ `X` = 139.5 s | either exceeds `X` — the pin's arithmetic is corrected in place, dated |
| L3 | decode t/s at the reference cell does not regress | it falls below the measured host-tier baseline in trade for the cold-boot number |
| L4 | boot child `ru_maxrss` ≤ 32 GiB | it exceeds 32 GiB — the 32 GiB host class is not reached |
| L5 | `wait4` value ≥ child `RUSAGE_SELF` | the self-read exceeds the reaped value — the instrument is void |
| L6 | two cold boots are byte-identical in digest form | any digest differs — localise, do not narrate |

## Explicit NOT claims

- **No measurement of any kind is made in this commit; every measured cell
  reads EMPTY (pending measurement).**
- **arcwell's numbers are arcwell's.** 2.91 GB/s, 1.18 vs 2.91 GB/s, 16.3×
  less CPU/GiB, 1.125 ms/expert, 2.18 GB/s serial, submit comparands — none is
  an arcint measurement. The only use here is a labelled projection.
- **The store cannot be filled today** (a repeated 4096-byte block, no expert
  tensors, no scales/zp), and the scales/zp layout is an OWED precondition.
- **The consumer integration does not exist**; the standalone fill budget is
  not the integrated gate.
- **Full depth is NOT assumed.** The pinned scope is depth 4; the full-depth
  arithmetic is printed for an operator decision only.
- **No FreeToken comparison here** — that is GENEVA (0.5.8), pinned only by our
  own runs.
- **No byte-identity claim on the B60** outside the A770-paired rule above.
- `X` is arithmetic (`code`), not `measured-here`, and is not moved after the
  fact.

## Where it lives

`docs/window-053.md` (this file). Campaign:
`docs/campaigns/nvme-direct-expert-tier.md`; design note:
`docs/design-nvme-direct-expert-tier.md`; PLE half:
`docs/campaigns/ple-disk-backend.md`, `docs/design-ple-disk-backend.md`.
Precedents: `docs/window-051.md` (BERLIN-AS-LISBON), `docs/window-052.md`
(VENICE). External convention: `~/src/arcwell` (tracked, established);
`~/src/FreeToken-ref` (tracked, established). Operator-local facts (host,
paths, lock, raw output) live only in the git-ignored packet
`docs/handoff-nvme-direct-expert-tier.local.md`.

## Status

- 2026-09-23 — **LISBON-001 drafted as the 0.5.3 acceptance commit; every
  measured row EMPTY.** Criteria and the threshold `X = 139.5 s` pinned before
  any LISBON card window. Scope pinned at depth 4 (mechanism, PLE precedent);
  full depth flagged as an operator decision. The gate is recorded as
  **BLOCKED** — the synthetic ext4 store and the OWED scales/zp store-layout
  precondition mean nothing can be filled correctly today. No measurement, no
  card leg, no wake lock, no host/store/module mutation. Review before commit
  (reviewer subagent); corrections in place with dates.
- 2026-09-23, later — **artifact-format step: the store precondition is
  CLEARED.** A REAL expert store was built on the ext4 partition (3,408 files
  of 2,457,600 B, `8,375,500,800 B`), every file exactly ONE plain extent,
  `aw_fiemap` byte-verifying all 3,408 and its red leg failing on content; the
  scales/zp layout verdict is **weights-only, device order**, with scales/zp
  excluded from the DMA slice and left on the transposing host path. Dependencies
  1 and 2 are cleared; dependency 3 (the D2/D3 consumer integration) remains,
  so the three gate rows stay **OPEN** and the gate is blocked only on that
  integration. No card leg, no module load, no wake lock; the arcwell module was
  found loaded/carved (inherited) and left as found; the synthetic store was
  not touched.
