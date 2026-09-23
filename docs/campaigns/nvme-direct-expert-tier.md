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

**Recon leg 2026-09-23** (operator-approved module load; operator declared the
B60 free for the campaign's later gate). Criteria as of that date, each with
its evidence class:

| # | criterion | status | evidence class | basis |
|---|---|---|---|---|
| 1 | ext4-backed expert store | **met** | `measured-here` | 1700 files, 2,457,600 B each, one per expert, `fallocate`d on a single ext4 partition. **Every file resolves to exactly ONE extent** (`filefrag` over all 1700: `1700 1`; flags `last,eof`), and the file→absolute-LBA translation byte-verifies against the raw device for all 1700 (`aw_fiemap` GREEN), with the red leg (`--mutate`) failing on content. One extent means one *request*, not one DMA segment — see the recon note below. |
| 2 | a card (B60) | **met by operator decision** | operator decision, 2026-09-23 | The B60 is free for this campaign's later gate. Not a measurement: a seat decision. The A770 stays backlogged (arcwell README; `KERNEL_FACTS.md` — its PCIe path goes unresponsive under load). |
| 3 | module installed and loadable | **met** | `measured-here` | DKMS module installed and built for the running kernel; `modprobe arcwell` returns 0, `lsmod` shows it, `/dev/arcwell` appears, `dmesg` carries only the two load info lines (no `pr_err`/`pr_warn`), and `AW_IOC_STATS` reads (`via_host_bounce=0`). Host restored to as-found (unloaded; no carve created — `modprobe` alone does not carve). |
| 4 | a consumer | **partially met** (2026-09-23, device-free) | `measured-here` + `code` | [DATED IN PLACE 2026-09-23: NOT discharged in full. The convergence CLAUSE of RED-C-02 is answered device-free from the served window-004 routing stream — see the "Convergence measurement" status entry below: the served static-partition policy's resident set is a pure function of configuration, fixed at `bind()`, so its composition converges at position 0 with zero evictions/thrash; the one mechanism that does NOT converge is a ROLLING census (`rounds_to_plateau=None` at every budget), which the design therefore excludes (pin once from an offline census). The literal RED-C-02 HARDWARE clauses — the `MOE_OTD_PERF_LOG` plateau probe's device-byte plateau and per-forward timing at the high-80s ratios, and the async-batch upload completing inside one inference step's budget — are **OWED** (no card leg was run), and the "a consumer" integration artifact (the design note / prefetch wiring) does not exist yet; that is the next step. Original text kept as written: "The residency policy arcwell would feed is not proven to converge at Flash-Next's admission ratio: RED-C-02 is open and unmeasured (`docs/design-qwen-flash-next.md`:207, :211, :215, :496). **Do not build this tier against it.**"] |

Criterion 1's only caveat is the layout claim's second half (one *request* vs
one *segment*), corrected by arcwell's own item 24 and reflected below.

## Recon — the mapping contract and the 64 KiB question (2026-09-23)

Evidence classes stated per disposition. Header quotes are verbatim from
`~/src/arcwell/stub/include/aw_uapi.h`; usage quotes from
`~/src/arcwell/docs/USING_ARCWELL.md`. Read from the checked-out tree, whose
tree hash (`e7d326e`) is identical to the `dev` tip (`3f974fa`) — `main` and
`dev` share that tree. Host/path detail
is in the git-ignored `docs/handoff-nvme-direct-expert-tier.local.md`.

### The ioctl surface (`aw_uapi.h`; the `#define` lines verbatim, ordered by ioctl number) — evidence `code`

```c
#define AW_DEVICE_NAME	"arcwell"
#define AW_IOC_MAGIC	'W'

#define AW_IOC_MAP_BUFFER   _IOWR(AW_IOC_MAGIC, 1, struct aw_ioc_map_buffer)
#define AW_IOC_READ_BLOCKS  _IOWR(AW_IOC_MAGIC, 2, struct aw_ioc_read_blocks)
#define AW_IOC_STATS        _IOR(AW_IOC_MAGIC,  3, struct aw_ioc_stats)
#define AW_IOC_UNMAP_BUFFER _IOW(AW_IOC_MAGIC, 4, __u32)
#define AW_IOC_READ_BATCH _IOWR(AW_IOC_MAGIC, 5, struct aw_ioc_read_batch)
#define AW_IOC_SUBMIT_BATCH _IOWR(AW_IOC_MAGIC, 6, struct aw_ioc_batch_submit)
#define AW_IOC_BATCH_WAIT _IOWR(AW_IOC_MAGIC, 7, struct aw_ioc_batch_wait)

#define AW_BATCH_MAX 256
```

`AW_IOC_MAP_BUFFER` registers a buffer; `AW_IOC_READ_BLOCKS` is the blocking
single transfer; `AW_IOC_READ_BATCH` submits a batch and waits;
`AW_IOC_SUBMIT_BATCH` + `AW_IOC_BATCH_WAIT` split submit from collect
(prefetch); `AW_IOC_STATS` is observability so a host bounce cannot hide;
`AW_IOC_UNMAP_BUFFER` releases a registration (and fd close releases all).

### The destination is a dma-buf from an xe VRAM BO — evidence `code`

```c
enum aw_buffer_source {
	AW_BUF_NONE = 0,
	/* an xe GEM handle, mapped and pinned by the module */
	AW_BUF_XE_GEM = 1,
	/* a dma-buf fd exported by xe; the module attaches and maps it and uses
	 * the exporter's sg dma addresses (BAR2 + offset) */
	AW_BUF_DMABUF = 2,
};

#define AW_MAP_F_REQUIRE_P2P	(1u << 0)	/* refuse to register unless the
						 * mapping really is peer-to-peer
						 * DMA-capable; never silently
						 * fall back to a host bounce */
```

`0.0.1` implements only `AW_BUF_DMABUF` (`aw_map_buffer` returns `-EOPNOTSUPP`
otherwise). `USING_ARCWELL.md` §6 gives the BO requirements verbatim:

```
| Buffer source | a dma-buf fd exported from an xe VRAM BO |
| BO requirements | `NEEDS_VISIBLE_VRAM`, `CPU_CACHING_WC`, size a multiple of 64 KiB |
| Alignment | `in_dest_offset` and the transfer length must be page-aligned |
| Batch size | 1..`AW_BATCH_MAX` (256) requests |
| Filesystem | one that implements FIEMAP. **ZFS and btrfs do not.** ext4 does |
| Card | the GPU and the NVMe must be reachable for P2P (`pci_p2pdma_distance() >= 0`) |
```

### Geometry rules — evidence `code`

- `MAP_BUFFER.in_length`: "bytes to register; **must be a multiple of the NVMe
  logical block**" — `aw_uapi.h`.
- `READ_BLOCKS.in_start_block`: "NVMe logical block number (512 B units unless
  the namespace reports otherwise)"; `in_block_count`: "logical blocks to
  transfer into the registered buffer"; `in_dest_offset`: "byte offset within
  the registered buffer" — `aw_uapi.h`.
- §6: `in_dest_offset` **and the transfer length** must be **page-aligned**.
- Batch: 1..`AW_BATCH_MAX` (256) requests.

### File → LBA / FIEMAP — evidence `code`

FIEMAP is explicitly not part of the module's surface:

```
 * FIEMAP is NOT part of this surface. It is the standard FS_IOC_FIEMAP ioctl
 * and, where experts are addressed as files, it runs once at open as a
 * file->block-range translation. It never appears in the streaming loop, and
 * there is no aw_set_fiemap ioctl (M4_API.md, held by tests/test_m4_api.py).
```

`USING_ARCWELL.md` §5: `absolute LBA = fe_physical / 512 + <partition start
sector>`, with three rules each backed by a red cell — **add the partition
start** (omitting it reads 48 GiB early and *succeeds* with plausible garbage;
only a byte comparison catches it); **reject non-plain extents** (`UNWRITTEN`,
`DELALLOC`, `INLINE`, `ENCODED`, `UNKNOWN`; pass `FIEMAP_FLAG_SYNC`); **handle
spanning**. The recommended layout — one file per expert, `fallocate`d — gives
**exactly 1 extent per expert**, so an expert is **one request** at one
`in_dest_offset`. It is **not** one DMA segment: a bio holds at most
`BIO_MAX_VECS` (256) pages = 1 MiB, so a 2.4 MB expert is ~3 bios whatever the
layout (arcwell item 24, `results/EXPERT_OPTION_B_2026-09-16.txt`).

### `AW_IOC_STATS` fields — evidence `code`

Field names and order are verbatim; per-field comments are elided except
`via_host_bounce`, quoted in full:

```c
struct aw_ioc_stats {
	__u64 reads;
	__u64 bytes;
	__u64 us_total;
	__u32 via_host_bounce;	/* Counts transfers that WOULD have gone through
				 * system RAM and were refused. MUST stay 0.
				 * Nonzero means a caller asked for something the
				 * module could only have served by bouncing --
				 * the request failed, but the configuration that
				 * produced it is wrong.
				 * Incremented at two sites in arcwell.c: the
				 * exporter clearing attach->peer2peer, and an sg
				 * whose pages are not P2PDMA pages. There is no
				 * path that bounces WITHOUT incrementing this,
				 * because there is no path that bounces at all --
				 * both sites return an error. */
	__u32 segments;
	/* --- appended 2026-09-15: buffer lifecycle --- */
	__u32 buffers_live;
	__u32 buffers_peak;
	/* --- appended 2026-09-15: batched submission --- */
	__u64 batches;
	__u64 batch_reads;
	__u32 max_inflight;
	__u32 batches_inflight;
};
```

Two consequences the client must respect: `via_host_bounce` nonzero does **not**
mean a bounce happened (no path bounces; both refusal sites return an error) —
it means a caller asked for a configuration that could only be served through
host memory; and **the counter is module-global, not per-client**, so it is read
as a delta before/after your own work, never as an absolute.

### What the CLIENT does once at open — evidence `code`

1. FIEMAP each expert file **once**, add the partition start, reject non-plain
   extents, and keep `(absolute LBA, byte length)`. This is the caller's job,
   not the module's.
2. Create the BO with `NEEDS_VISIBLE_VRAM` + `CPU_CACHING_WC` and **size rounded
   up to a 64 KiB multiple**; export it with `DRM_IOCTL_PRIME_HANDLE_TO_FD`.
3. `AW_IOC_MAP_BUFFER` with `in_source = AW_BUF_DMABUF`, `in_length = BO size`,
   then **assert the flag** (verbatim from `USING_ARCWELL.md` §3):

   ```c
   if (!(mb.out_flags & AW_MAP_F_REQUIRE_P2P))
           abort();                   /* see §6 -- this check is not optional */
   ```

4. Read `AW_IOC_STATS` as a baseline.
5. Per step, issue fetches **ahead of need** through `AW_IOC_SUBMIT_BATCH` and
   collect through `AW_IOC_BATCH_WAIT`. `AW_IOC_READ_BLOCKS` blocks and must not
   sit on the decode path.
6. Close the fd (releases every registration and drains in-flight batches).

### The 64 KiB anomaly — RESOLVED: the store does NOT violate the contract

Evidence: `code` (header, §6, and `stub/test/aw_expert_test.c`) plus
`measured-here` (the store).

The campaign charter's own parenthetical was right: **the 64 KiB rule is on the
xe VRAM BO, not on the file.** `USING_ARCWELL.md` §6 puts it in the row
`BO requirements`; `KERNEL_FACTS.md` gives the reason — "size a multiple of
**64 KiB** (DG2 flat-CCS sets the VRAM manager `min_page_size = 64K`)" — a
`DRM_IOCTL_XE_GEM_CREATE` gate. It is `MAP_BUFFER.in_length` — the **registered
buffer**, set to the BO size, *not* the file — that must be a multiple of the
NVMe logical block. The file's size enters only as the transfer, addressed by
`READ_BLOCKS.in_start_block` / `in_block_count` in logical blocks, with §6's
rule that the transfer length and `in_dest_offset` be page-aligned.

The client rounds the BO up; the source lines below are from
`stub/test/aw_expert_test.c`, with bracketed annotations added (they are not in
the source):

```c
bo.size = (st.st_size + 0xffff) & ~0xffffULL;   /* [BO rounded to 64 KiB] */
...
mb.in_length = bo.size;                          /* [the BO, not the file] */
...
reqs[i].in_block_count = st.st_size >> 9;        /* [the file: 4800 blocks] */
```

Measured on the store (all 1700 files, one size):

```
2457600 / 4096  = 600 remainder 0      -> page-aligned: OK
2457600 / 512   = 4800 remainder 0     -> whole logical blocks: OK
2457600 / 65536 = 37 remainder 32768   -> NOT a 64 KiB multiple
```

**Verdict:** 2,457,600 B = 37.5 × 64 KiB is irrelevant to the contract. It is
page-aligned and whole-LBA, so the transfer is legal; the xe BO is rounded up to
38 × 64 KiB = 2,490,368 B by construction, satisfying the only 64 KiB rule.
There is no violation and no artifact relayout owed on this account.

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
- 2026-09-23 — **recon leg, sole writer.** Entry criteria re-examined, dated:
  criterion 1 **met** (`measured-here`: 1700 expert files, 2,457,600 B each,
every one exactly ONE extent, all `last,eof`; `aw_fiemap` byte-verifies all 1700
the raw device and its `--mutate` red leg fails on content); criterion 2 **met
by operator decision** (B60 free for the later gate; A770 stays backlogged);
criterion 3 **met** (`measured-here`: DKMS module built for the running kernel,
`modprobe arcwell` clean, `/dev/arcwell` up, dmesg clean, `AW_IOC_STATS` reads
with `via_host_bounce=0`, host restored to as-found unloaded with no carve);
criterion 4 **not met** (`code`: RED-C-02 open/unmeasured — a consumer must not
be built against it).

> [SUPERSEDED 2026-09-23: criterion 4's status was re-measured device-free — see
> the "Convergence measurement" entry below. It is now **partially met**: the
> convergence clause is answered, the RED-C-02 hardware clauses and the consumer
> integration stay OWED. This line is kept as the recon date's record.]

The **64 KiB anomaly is resolved, not a violation**: the rule is on the xe VRAM
BO (`USING_ARCWELL.md` §6 `BO requirements`; `KERNEL_FACTS.md`), which the
client rounds up to a 64 KiB multiple (`aw_expert_test.c`), while the file need
only be page-aligned and whole-LBA — 2,457,600 B is 600 pages and 4800 blocks.
The mapping contract is now quoted verbatim in the Recon section above.

Still open: the **gate itself is NOT part of this recon** — cold-TTFT
(arcwell path vs host-fed, same card/artifact/policy, one window), answer
byte-identity across both arms and two cold boots, decode non-regression at the
reference cell, and the prefetch depth that hides the 1.125 ms (arcwell's
number, not arcint's). Also owed: the design note
(`docs/design-nvme-direct-expert-tier.md`) — the prefetch schedule, where the
routing warning comes from, and the fallback for a fetch that has not landed —
and criterion 4's convergence measurement (RED-C-02). No arcint card leg was run
here; the B60 was left idle.

---

## Convergence measurement — criterion 4's convergence clause answered device-free (2026-09-23); criterion 4 partially met

[measured-here, one device-free run; NO card leg, B60 idle] Entry criterion 4
("a consumer") asked whether the residency policy arcwell would feed converges
at the admission ratio a large MoE needs. It was measured from the routing
stream already on disk, using the repository's own policy arithmetic.

**The trace.** Served window-004 (`venice-census-004`), sha256
`bf32c87401d7ac444aa1aae89eb5a641f69cd84b8c43643f7222253ad5d2d304`, 197,184
calls / 3,893,280 routed accesses, on the persistent census path (operator-local
detail in the handoff packet). The served artifact is **Flash-Next itself**
(`qwen38-flash-next-d48n-ov`): depth 48, 512 experts per layer, `top_k` 10, KV
u8, `tier=static`, provenance `offload_ratio=99`. So the routing stream is
Flash-Next's own served router, and the admission ratio is directly varied by
the slot budget.

**Instrument + cells (deliverable 3).** New `tools/expert_convergence.py`
(+ `tools/test_expert_convergence.py`, **18 red-first cells**, green; ladder now
146 cells = 81 census + 33 policy-compare + 14 census-seed + 18 convergence).
The cells pin: access-position deciles (including the sub-decile case of fewer
accesses than buckets); `convergence_decile`; pinned-only fill
and `calls_to_last_new`; a two-quiet-forward plateau (the engine's
`plateaued < 2`); TRUE-LRU promotion (FIFO is distinguishable and understates
the comparand); a rolling prefix series that reports a plateau only on an
unchanged set; and refusals for a negative LRU budget and a zero rolling
budget (a zero budget would otherwise falsely "plateau" on an empty set).
Red-first shown by mutation: dropping `move_to_end` and an off-by-one
ordinal bucket fails 5 cells (3 decile + end-to-end + TRUE-LRU); counting
unpinned ids, declaring a plateau on any quiet forward, and always reporting a
plateau fails 3 cells.
Raw: `cells-green.txt`, `redfirst-mutant{A,B}.txt` on the census path.

**Definitions.** Regimes are half-open call ranges on the served sequence:
`probe [0,288)` (the 6 load-time forwards, 48 calls each), `prefill [288,576)`
(288 batched calls, 2,735 tokens), `decode [576,end)` (196,608 single-token
calls, 4,096 tokens), `corpus [288,end)` (the served request). Position is
ACCESS position (a batched prefill call contributes many accesses). Seeds: the
incumbent `static-splitmix64` (patch 0018) and four frequency seeds calibrated
on one regime each (probe / prefill / decode / mixture). Budgets are the
plugin's integer-division slot count `512*(100-ratio)/100`.

**Budget correction, dated in place.** The brief named "ratio 75 → 16
slots/layer". Source and measurement disagree: `resident_expert_num =
const_shape[0]*(100-otd_ratio)/100` (`patches/0041-…:78`, `0047-…:62`) with
`const_shape[0]=512` gives **128** at ratio 75 — and 128 is exactly what the
measured ratio-75 sweep point pins (`window-052` rate leg, census top-128). No
integer offload ratio gives 16 at 512 experts (512·(100−r)/100 = 16 ⇒ r =
96.875; nearest integer 97 → **15**). The two SERVED budgets below are ratio 99
→ **5** and ratio 75 → **128**; **16** is included because the brief named it,
labelled as mapping to no served ratio.

**Command** (trace path operator-local, recorded in the handoff packet):

    python3 tools/expert_convergence.py --trace <w004.trace> \
        --slots 5,16,35,51,71,87,107,128 --out <w004-authoritative.json>

Raw stdout `w004-authoritative.txt` (sha256 `a4da92f8…`), JSON
`w004-authoritative.json` (sha256 `73c47bb3…`), tool sha256 `af5ac229…`, both
on the persistent census path. Accepted lines, verbatim.

**1. Resident-set composition vs position. Met, trivially.** Under the SERVED
static partition the set is a pure function of `(seed, layer_key, expert,
slots)`, fixed at `bind()` (`code`: patch 0018; patch 0046 validates the census
membership at construction and applies it at `bind()`). Membership changes =
**0** at every position; **time-to-convergence = 0 routed calls**. No history
input exists, so there is no warm-up, no oscillation and no eviction.

The ROLLING census — the set recomputed from a growing prefix, i.e. what a
periodic re-calibration would pin — does **not** converge at any budget or
regime (`code`+`measured-here`). Verbatim at ratio 75 / 128 slots:

    rolling probe    rounds_to_plateau=None plateau=False changed=[None, 1, 2, 4, 8, 16, 32, 48, 48, 30]
    rolling prefill  rounds_to_plateau=None plateau=False changed=[None, 1, 2, 4, 8, 16, 32, 48, 48, 32]
    rolling decode   rounds_to_plateau=None plateau=False changed=[None, 1, 2, 4, 8, 16, 32, 47, 48, 48, 48, 48, 48, 48]
    rolling corpus   rounds_to_plateau=None plateau=False changed=[None, 1, 2, 4, 8, 16, 32, 48, 48, 35, 24, 40, 40, 48]

Identical shape at every budget 5…128. This is the V3 failing shape, now shown
to be a property of the ROUTING distribution, not of the served policy: the
census ranking keeps moving past 4,096 decode tokens. **Design consequence:
a consumer must pin ONCE from an offline calibration census and must not
re-derive the hot set from live traffic; a rolling refresh would thrash
membership.**

**2. Hit rate over sequence position. (seed × regime) property, never one
number.** Verbatim:

    ratio 99 (5 slots)
    decode   splitmix64      early=  1.17% steady=  0.97% tail=  0.96% tail-steady= -0.01pt conv@d1 fill= 80.0% lastnew=191481 plateau=True
    decode   census-decode   early=  7.90% steady= 14.99% tail= 15.59% tail-steady= +0.61pt conv@d4 fill=100.0% lastnew=51321 plateau=True
    decode   census-mixture  early=  8.97% steady= 14.37% tail= 14.66% tail-steady= +0.29pt conv@d4 fill=100.0% lastnew=43640 plateau=True
    prefill  census-prefill  early=  9.40% steady=  8.08% tail=  7.09% tail-steady= -0.98pt conv@d9 fill=100.0% lastnew=47 plateau=True
    ratio 75 (128 slots)
    decode   splitmix64      early= 25.75% steady= 25.02% tail= 24.97% tail-steady= -0.05pt conv@d1 fill= 79.5% lastnew=196380 plateau=True
    decode   census-decode   early= 68.17% steady= 85.81% tail= 83.96% tail-steady= -1.85pt conv@d9 fill=100.0% lastnew=136653 plateau=True
    prefill  census-prefill  early= 66.58% steady= 63.60% tail= 55.68% tail-steady= -7.91pt conv@d10 fill=100.0% lastnew=286 plateau=False
    prefill  census-mixture  early= 46.44% steady= 50.52% tail= 46.56% tail-steady= -3.96pt conv@d9 fill= 98.9% lastnew=283 plateau=False
    ratio 86 (71 slots; Flash-Next admission)
    decode   census-decode   early= 51.12% steady= 70.30% tail= 68.05% tail-steady= -2.25pt conv@d9 fill=100.0% lastnew=93753 plateau=True
    decode   census-mixture  early= 55.65% steady= 66.14% tail= 62.98% tail-steady= -3.16pt conv@d9 fill= 99.4% lastnew=140591 plateau=True

Reading: (a) the incumbent `splitmix64` is flat at analytic chance `slots/512`
(≈0.98 % at 5 slots, 25.0 % at 128) — expected of a frequency-free seed, and
it converges by decile 1 because it has no signal to converge to; (b) a
regime-matched census seed rises to a steady state (`convergence_decile` 4–10,
tail within ~2–4 pt of steady); (c) a seed calibrated on the WRONG regime
drifts monotonically (e.g. `census-prefill` scored on decode at 128: 58.76 →
39.27 → 35.22) — the (seed × regime) law again. The ratio-86 point is the
design's own ~8 GiB / ~86 % row (`docs/design-qwen-flash-next.md`:346), the LOW
edge of the stated "high 80s to low 90s" admission range (`:379`); 71 is the
plugin's integer-division value for the design's 72, and 86 is the low edge of
that range, not a single "admission" ratio.

**3. Eviction / thrash per 1000 routed calls.** Static partition: **0** at
every budget and regime (no eviction path; non-resident experts take the host
tier). Demand-warm LRU comparand, verbatim:

    ratio 99 (5 slots)
    decode   lru-per-layer   hit=  2.27% evict=1921297 per1kcalls=    9772.2 per1kacc=  977.2
    ratio 75 (128 slots)
    decode   lru-per-layer   hit= 77.34% evict=439354 per1kcalls=    2234.7 per1kacc=  223.5
    prefill  lru-per-layer   hit= 81.58% evict=235701 per1kcalls=  818406.2 per1kacc=  179.5
    ratio 86 (71 slots)
    decode   lru-per-layer   hit= 65.59% evict=673170 per1kcalls=    3423.9 per1kacc=  342.4

Per-call rates are NOT comparable across regimes (prefill calls are batched,
455 accesses/call; decode calls are single-token, 10 accesses/call); the
per-1,000-ACCESS rate is the comparable figure. The static partition pays no
thrash at all; the LRU comparand pays ~180–977 evictions per 1,000 accesses.

**4. Tail (last decile vs steady).** Regime-matched census seed: within
−1.9…−4.1 pt of steady (stable); incumbent: within ±0.2 pt (flat); an
out-of-regime seed drifts further (e.g. `census-prefill` on corpus at 128:
−14.18 pt) — the (seed × regime) law.

**5. Fill / plateau-probe proxy (device-free).** The pinned-set demand is
monotone at every budget (no oscillation); `calls_to_last_new` is the
device-byte plateau point under the static partition (uploads are never freed).
At the load-time probe regime the fill is **not** complete for a seed not
calibrated on the probe: `splitmix64` reaches only **38.2 %** of its pinned set
after the 6 forwards, yet the trace's corpus begins at call 288 — the engine
terminated at 6 forwards while the demand proxy was still adding new pinned
experts. That matches the recorded two-ledger finding that the driver keeps
most of the pinned pool host-mapped, so `device_resident_bytes` flattens before
every pinned expert has been demanded. **The actual device-byte plateau, the
probe's per-forward timing and the async-batch upload budget are hardware
claims and are OWED — this measurement cannot make them, and does not fake
them.**

**Verdict — partially met; NOT discharged in full.** The convergence CLAUSE of
RED-C-02 ("does not oscillate/thrash") is answered device-free: the policy
arcwell would feed (the served static partition, pinned once from an offline
census seed) is a pure function of configuration, converges at position 0 and
evicts nothing. But two halves remain unmade: (a) the literal RED-C-02 HARDWARE
clauses — the `MOE_OTD_PERF_LOG` plateau probe's device-byte plateau and
timing at the high-80s ratios, and the async-batch upload budget — need a card
leg and are **OWED**; and (b) criterion 4 is literally "a consumer": no
integration artifact exists, and the design note is the next step. So this leg
unblocks the design note; it does NOT license building the tier against an
unmeasured engine plateau probe. Note also that a regime-matched seed's hit
series settles at `convergence_decile` 4–10 (relative to the trace end), which
is CONVERGENCE, not ADEQUACY: at the ratio-99 budget that steady value is only
~15 % of decode accesses (the measured V1 regime shortfall, `window-052`), a
separate speed-gate matter.

**OWED, named and not faked:** (a) the plateau probe's actual device-byte
plateau and per-forward timing at the high-80s ratios on the B60; (b) the
async-batch upload path completing inside one inference step's budget; (c) the
campaign GATE itself — cold TTFT (arcwell vs host-fed, one window), answer
byte-identity across arms and two cold boots, decode non-regression at the
reference cell — plus the design note
(`docs/design-nvme-direct-expert-tier.md`), the next step.

**Evidence classes.** ratio→slots arithmetic: `code` (`fit.h`; patches
0041/0047). Fixed-set composition and zero evictions: `code` (patch 0018/0046).
Every convergence number above: `measured-here` (device-free replay of the
served window-004 trace). That the trace is Flash-Next's own served router:
`measured-here` (provenance `run=venice-census-004`, depth 48, 512 experts).

---

## Design note — the prefetch schedule, the routing warning horizon, and the fallback (2026-09-23); a verdict on the miss tier

[`code` + carried `measured-here`; device-free] The note is
`docs/design-nvme-direct-expert-tier.md`. Its verdict, on the campaign's own
condition:

**The warning horizon is zero layers, so a router-driven NVMe fetch cannot pay;
as a miss tier LISBON keeps the host hop.** The serving loop makes a layer's
top-k ids host-visible only inside that layer's own MoE primitive hook
(`code`: `patches/0012-…:932,941`; patch 0017 hoists the read into one round
trip but does not move it earlier; patch 0037's prefill paths are the same
shape; patch 0044's trace has this exact granularity). There is no graph-level
pre-pass computing a future layer's router (`code`: searched the series; one
router per layer, `patches/0019-…:284-297`). So the fetch can only be issued at
the layer that already needs the expert — depth 0, against arcwell's own
1.125 ms. That is the losing configuration the Known section already describes.

**The one path the loop reaches is the load-time pinned fill.** The static
partition's membership is a pure function of configuration fixed at `bind()`
(`code`: patch 0018; patch 0046), so its warning is unbounded — the whole pinned
set is known before the first token. The note schedules it there: one batch per
layer (71 requests at ratio 86, 128 at ratio 75), 4 batches in flight over
`AW_IOC_SUBMIT_BATCH`/`AW_IOC_BATCH_WAIT`, no fetch on the decode path. Whether
that fill pays cold TTFT is a projection only (arcwell's own 2.91 GB/s applied
to arcint's bytes: 0.20/2.88/5.19 s at ratios 99/86/75) — **arcwell's numbers,
never arcint's** — and the gate decides.

**The fallback.** Two cases, decided at load (`code`: `patches/0018-…:716-780,800-870`): an expert the partition marks **non-resident** takes the host tier, at the existing static-partition `probe()` branch (`kCpuTierSentinelSlot` on decode; `acquire_one` → `std::nullopt` → `moe_cpu_expert` on prefill). A **pinned** expert whose fetch has not landed by the load barrier is a **load failure** — retried at the barrier, then refused; it is **not** silently demoted, because a demotion would make that boot's residency differ from a clean boot and break byte-identity across cold boots. A synchronous arcwell read on the decode path is **forbidden** (the campaign's *Out*).

**Fidelity, and what the note must not oversell.** FreeToken-faithful: the PLE
disk backend and an O_DIRECT-friendly container (`code`:
`~/src/FreeToken-ref/.../ple_disk.py`, `.../checkpoint/ftw.py`). **Not**
FreeToken's way: the per-forward NVMe DMA tier — the reference's runtime miss
tier is the CPU executor + host bank (`code`: `.../moe/host_banks.py`,
`expert_banks.py`, `cpu_executor.py`), i.e. exactly the host hop this campaign
keeps. The consumer is pinned once from an offline census, per `(seed ×
regime)`, never averaged across regimes; criterion 4's literal hardware clauses
(the plateau probe and the async-upload budget) and the gate remain **OWED**.

**The next leg is specified in the note (§7):** the B60 probe measures the
device-byte plateau/per-forward timing at the high-80s ratios and the
async-upload budget inside one step, with `AW_IOC_STATS` as a delta
(`via_host_bounce = 0`, `max_inflight > 1`). No card leg was run here; the B60
was left idle and the expert store and module untouched.
