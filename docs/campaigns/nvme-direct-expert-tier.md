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
| 4 | a consumer | **partially met** (2026-09-23 device-free; RED-C-02's hardware clauses MEASURED on the B60 same day) | `measured-here` + `code` | [DATED IN PLACE 2026-09-23: NOT discharged in full. The convergence CLAUSE of RED-C-02 is answered device-free from the served window-004 routing stream — see the "Convergence measurement" status entry below: the served static-partition policy's resident set is a pure function of configuration, fixed at `bind()`, so its composition converges at position 0 with zero evictions/thrash; the one mechanism that does NOT converge is a ROLLING census (`rounds_to_plateau=None` at every budget), which the design therefore excludes (pin once from an offline census). The literal RED-C-02 HARDWARE clauses — the `MOE_OTD_PERF_LOG` plateau probe's device-byte plateau and per-forward timing at the high-80s ratios, and the async-batch upload completing inside one inference step's budget — are **OWED** (no card leg was run), and the "a consumer" integration artifact (the design note / prefetch wiring) does not exist yet; that is the next step. [DATED IN PLACE 2026-09-23 (B60 probe): the two literal RED-C-02 HARDWARE clauses are now MEASURED — the plateau is 0.37 GiB at ratios 86/83 with evictions 0, and the pinned-fill async budget is measured (the "B60 probe" section below). What stays OWED is the GATE (cold TTFT both arms in one window, byte-identity across arms and two cold boots, decode non-regression), the serving step with the fill overlapping, the full-slice fill's byte-transparency (the store-layout precondition), and the design-note D2/D3 integration.] Original text kept as written: "The residency policy arcwell would feed is not proven to converge at Flash-Next's admission ratio: RED-C-02 is open and unmeasured (`docs/design-qwen-flash-next.md`:207, :211, :215, :496). **Do not build this tier against it.**"] |

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
`in_dest_offset`. [DATED IN PLACE 2026-09-24, patch 0049 leg: with the REAL
store this holds only if one destination maps the whole record. The store's
`gate|up|down` record meets three per-tensor device regions, so one expert is
THREE page-aligned requests; see the "D2/D3 plugin transport" section.] It is
**not** one DMA segment: a bio holds at most
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

---

## B60 probe — RED-C-02's hardware clauses measured; the full-slice fill stays OWED on the store layout (2026-09-23)

[`measured-here` on the B60; `code` for the layout fact] This leg ran the design
note's §7 probe. One card leg at a time; the B60 (GPU.0, PCI `8086:e211`) was
the only card touched, the A770 was idle and untouched. Protocol reproduced
from the `sub4bit-vram-kernel` precedent (one fresh process per arm, the sampler
started before the leg, plugin `ov-0047`, graceful stop so `[OTD_PERF]` dumps).
Raw logs are on the operator-local persistent card-host path (handoff packet);
the commands below use placeholders for it.

### 1. The device-byte plateau and per-forward timing (design note §7.1)

**Arm protocol** (`measured-here`): fresh process; plugin `ov-0047`
(`f021de51b5812ee2…`, patches 0003–0047); served binary tree `wt-b6dbca5`
(`f6abb402…`); artifact `qwen38-flash-next-d48n-ov` (`641fcb18…`, bin
77,492,280,673 B); B60 `GPU.0` = PCI `8086:e211`; `--moe-cpu-tier` (the static
partition is the plugin default: `MOE_CPU_TIER_STATIC_PARTITION=1`,
`seed_source=splitmix64`, `slots` = the plugin's integer division); KV u8;
chunk 512; n_ctx 8192; one lane; `MOE_OTD_PERF_LOG=1`; the same greedy 64-token
France prompt (temperature 0, `ignore_eos`) in both arms. No
`--moe-per-expert-dispatch`: the probe is the ordinary OTD slot pool.

Ratio 86 — plugin integer division `512*(100-86)/100 = 71` slots/layer:

```
lgc  load: language model ready in 34.7 s (paged); device-resident 8.06 GiB
lgc  load: expert slots: plateau probe settled at 0.37 GiB after distinct-token chunks (source: probe-static)
lgc  load: expert slots host-side: 7.91 GiB (GTT, source: config)
lgc  load: expert slot pool: pinned-pool ceiling 7.91 GiB (source: config) against 0.37 GiB measured resident (source: probe-static) -- the difference is what the driver keeps host-mapped
lgc  slot 0: prefill     5 tok in 11.11 s (  0.5 t/s) | graph 11.10 s, ...
lgc  slot 0: decode     64 tok in 115.21 s (  0.6 t/s) | graph 115.18 s, ...
[OTD_PERF] ... slots=71 static_partition=1 seed_source=splitmix64 ...
[OTD_PERF] gpu_hits=5616, gpu_misses=47310, gpu_hit_rate=10.611%, ... evictions=0, acquisitions=52926, device_slot_buffers=0, host_slot_buffers=382, alloc_fallbacks=0, staging_bytes=352256000
```
`props ready` after 1925 s (admission wall).

Ratio 83 — `512*(100-83)/100 = 87` slots/layer (the fit-ledger `ceil` = 88):

```
lgc  load: language model ready in 26.6 s (paged); device-resident 8.06 GiB
lgc  load: expert slots: plateau probe settled at 0.37 GiB after distinct-token chunks (source: probe-static)
lgc  load: expert slots host-side: 9.67 GiB (GTT, source: config)
lgc  load: expert slot pool: pinned-pool ceiling 9.67 GiB (source: config) against 0.37 GiB measured resident (source: probe-static) ...
lgc  slot 0: prefill     5 tok in 10.43 s (  0.5 t/s)
lgc  slot 0: decode     64 tok in 120.01 s (  0.5 t/s)
[OTD_PERF] ... slots=87 static_partition=1 seed_source=splitmix64 ...
[OTD_PERF] gpu_hits=6894, gpu_misses=46037, gpu_hit_rate=13.0245%, ... evictions=0, acquisitions=52931, device_slot_buffers=0, host_slot_buffers=382, alloc_fallbacks=0, staging_bytes=352256000
```
`props ready` after 1750 s.

**Verdict.** `measured-here`. The plateau is **0.37 GiB at both ratios** and
**`evictions = 0`** — no capacity thrash. This is RED-C-02's hardware clause
measured: the offload-ratio machinery does not break at Flash-Next's admission
ratio. Two limits are stated, not hidden: (a) the probe terminates on its own
two-non-increasing-reads condition and reports the high-water figure; this
build emits no per-forward device-byte series, so "does not oscillate" is the
plugin's own convergence assertion, not an independently printed series;
(b) the plateau **is not the pinned pool**. Under the static partition
`device_slot_buffers = 0`: the pool is host-mapped and the device term is a
0.37 GiB working set against a 7.91 / 9.67 GiB host-side ceiling — the
two-ledger shape the design note carries. The per-forward served time is
1.80 s/token (ratio 86; 115.21/64) and 1.88 s/token (ratio 83; 120.01/64 =
1.875 rounded); the plugin times no individual probe forward.

### 2. The async pinned-fill budget (design note §7.2)

**Protocol** (`measured-here`; chosen deliberately **without arcint** — the
serving integration the note schedules does not exist yet, so the budget is
measured standalone by arcwell's own harness shape — a documented **narrowing**
of §7.2, which asks submit/collect "while the serving stream runs"; that
integration is OWED, so what is measured here is the pinned-fill batch budget
itself. Client
`aw_fill_budget`, derived from arcwell's `stub/test/aw_async_test.c` (same
contract checks, plus `--n`/`--depth` and a full `AW_IOC_STATS` delta); run on
the card host against the ext4 store **read-only**; DEPTH=4 batches in flight,
the design note's one-batch-per-layer schedule; B60 `renderD129`.

n = 71 (ratio 86, one batch per layer):

```
MAP_BUFFER ok handle=1 out_flags=0x1
ONE BATCH n=71: submit=20919 us complete=64463 us polls=25764 eagain=1
  collected bytes=174489600 completed=71 segments=213 err=0
  submit is 32.5% of the batch wall time
  rate one batch: 2.71 GB/s
DEPTH=4 submitted in 188108 us (47027 us each); batches_inflight=4
  all 4 collected, 697958400 bytes total
```

n = 87 (ratio 83):

```
MAP_BUFFER ok handle=2 out_flags=0x1
ONE BATCH n=87: submit=28260 us complete=72802 us polls=25409 eagain=1
  collected bytes=213811200 completed=87 segments=261 err=0
  submit is 38.8% of the batch wall time
  rate one batch: 2.94 GB/s
DEPTH=4 submitted in 242459 us (60615 us each); batches_inflight=4
  all 4 collected, 855244800 bytes total
```

**Reading** (`measured-here`, against arcwell's own comparand — **arcwell's
numbers, not ours**: submit ~9.7 ms/64-expert idle, ~42 ms at 4 outstanding;
2.91 GB/s at `max_inflight=193`). Submit returns in 32.5 % / 38.8 % of the
batch wall, so the split is real and the fetch is hidden behind the transfer;
one-batch rate 2.71 GB/s is inside arcwell's measured 2.18–2.91 GB/s band, and
2.94 GB/s is marginally above arcwell's 2.91 GB/s ceiling (the larger batch, 87
experts); segments are ≈3/expert (213/71, 261/87), the bio floor. Submit cost scales with bytes
queued: 47 / 61 ms each at depth 4 for 71 / 87 experts, above arcwell's 42 ms
comparand — consistent with arcwell's "smaller, more frequent batches". The
serving step's own wall time with the fill *overlapping* is **OWED**: it needs
the integration (design note D2), which does not exist.

### 3. `AW_IOC_STATS` as a delta (design note §7.3)

Module loaded fresh (`modprobe`; carve-on-demand, no `carve_all`); the baseline
read all-zero. The client reads the delta across its own work (module-global
counter, never absolute):

```
n=71 run:  via_host_bounce 0 -> 0   max_inflight 0 -> 213   batches 0 -> 5
           batch_reads 0 -> 355   segments 0 -> 1065   bytes +872448000
n=87 run:  via_host_bounce 0 -> 0   max_inflight 213 -> 261  batches 5 -> 10
           batch_reads 355 -> 790   segments 1065 -> 2370   bytes +1069056000
```
`peer2peer=1`, `out_flags=0x1` (`AW_MAP_F_REQUIRE_P2P`); dmesg
`MAP_BUFFER … peer2peer=1`. Requirement met: `via_host_bounce` delta 0,
`max_inflight > 1`, `batches`/`batch_reads` nonzero, `segments` reported.

### 4. The store-layout precondition — **OWED** (design note §3/§7)

`code`: the plugin's device slot layout and the weight-file layout differ for
scales/zp — device `[group][oc]`, file `[oc][group]`; `maybe_transpose_scale_zp`
transposes on upload (`patches/0011-…:75-90`, `0006-…:291`). The weights are
`[oc][ic]` in both, so a byte-transparent full-slice DMA would land the weights
correctly and the scales transposed.

`measured-here`: the store as-built cannot resolve this. The ext4 expert store
is **arcwell test data**, not an arcint expert artifact: it was populated by a
synthetic builder that writes, per file, one deterministic 4096-byte block
repeated 600 times (`expert_0000.bin`: 600/600 blocks identical, 1 distinct
block; distinct files differ by seed). It carries no expert tensors and no
scales/zp, so neither a device-order store nor the transpose decision can be
exercised or verified against it.

**Disposition: the full-slice fill's byte-transparency is OWED**, with that
specific reason. Resolving it is a store-layout decision (lay files in device
order, or move the small scale/zp tensors through the existing host path) and
belongs to the artifact-format step, not this probe. No correct fill is claimed.

### 5. Determinism caveat (operator decision 2026-09-23)

The B60 is a timing/statistics card here. Every number above is admissible as a
measurement on the B60; **no byte-identity claim is made** in this leg. The two
arms' greedy answers happened to be literally identical (both
`Paris. The capital of Germany is Berlin. …`), which is an observation, not a
§3.4 discharge: the gate's byte-identity rows need the two cold boots and, where
the B60 is not known bit-readable, an A770 confirmation
(`served-prefill-determinism`).

### 6. Card, lock, module state

- Before: the two served units were inactive+disabled as found (not touched);
  `pgrep -x arcint` empty; no `/dev/dri` holders; `arcwell` not loaded,
  `/dev/arcwell` absent, no carve this boot; B60 `power/control=on`, runtime
  active.
- One card leg at a time; the B60 alone. The A770 was idle throughout.
- Wake lock on the coordinator host: found none set; taken for 6 h with the
  reason recorded (operator-local), released when card work was done — no
  foreign lock overwritten.
- Sampler per SOP on the **physical host** (driver/USM-host memory is charged
  there; the container's `MemAvailable` is cgroup-limited), watchdog
  `MemAvailable < 4 GiB`: minimum observed 20,473,296 kB = 19.5 GiB (ratio 86) /
  18,287,308 kB = 17.4 GiB (ratio 83) / 60,631,024 kB = 57.8 GiB (arcwell arms);
  **0 watchdog trips**.
- The expert store was read-only; file mtimes unchanged. No relayout, no write.
- After: no `arcint`; the `arcwell` module was **left loaded** (carved,
  `peer2peer=1`). Unloading a carved GPU is the documented half-state hazard in
  the arcwell tree's own `~/src/arcwell/KERNEL_FACTS.md` ("do not leave a carved
  GPU idle with arcwell unloaded"); the carve cannot be released short of a
  device rebind, and the wake lock's expiry lets the host clear it at its next
  sleep.

### 7. What remains (the gate, design note §7.4)

Not this leg: cold TTFT arcwell vs host-fed in one window, byte-identity across
arms and two cold boots, decode non-regression at the reference cell. The
`aw_fill_budget` numbers are a standalone pinned-fill budget, not the
integrated gate; the "serving step with the fill overlapping" and the
cold-TTFT delta (still a projection over arcwell's own 2.91 GB/s — **arcwell's
number, never arcint's**) are **OWED**. Criterion 4's "a consumer" integration
(design note D2/D3) is still the unbuilt artifact.

**Evidence classes, this leg.** Plateau/timing/OTD_PERF, the async budget, and
the `AW_IOC_STATS` delta: `measured-here` (B60). The scale/zp layout divergence:
`code`. The store's synthetic content: `measured-here`. arcwell's 1.125 ms /
2.91 GB/s / submit comparands: arcwell's own `measured-here`, labelled. The
store-layout resolution and the gate: **OWED**.

---

## LISBON-001 — the acceptance commit, criteria pinned, rows EMPTY (2026-09-23)

[`code` for the threshold; `measured-here` for cited inputs; NO measurement]
`docs/window-053.md` is 0.5.3's acceptance commit, written before the LISBON
card window exists. It pins the campaign gate's three rows — cold TTFT
(arcwell vs host-fed, both arms in one window, arcwell at or below host-fed,
**absolute threshold `X = 139.5 s`**) at a **depth-4** scope on the B60, RSS
bounded through boot (`wait4` child `ru_maxrss` ≤ **32 GiB**, CF-KEYSTONERSS),
and restart determinism (two cold boots, digest form) — and leaves every
measured cell **EMPTY**. `X = T_boot 136 s + T_fill 0.387 s + T_prefill
3.08 s`, all terms sourced (`code` byte counts over `measured-here` inputs);
arcwell's 2.91 GB/s is used only as a labelled projection. Full depth is
flagged as an **operator decision**, not assumed.

**The gate is recorded as BLOCKED, not measurement-ready.** The ext4 expert
store is synthetic arcwell test data (one repeated 4096-byte block, no expert
tensors, no scales/zp), so nothing can be filled correctly today; the
scales/zp store-layout precondition (device `[group][oc]` vs file `[oc][group]`)
is **OWED to the artifact-format step**; and the D2/D3 consumer integration does
not exist. The three gate rows stay **OPEN**. The harness the measurement will
use is named in `docs/window-053.md` (served binary + `ov-0047`, OTD_PERF
plateau probe, the arcwell `aw_fill_budget` client with an `AW_IOC_STATS`
delta, `os.wait4`/`ru_maxrss`, the SOP physical-host sampler, `--fit-ledger-dir`).
No card leg, no module load, no store mutation, no wake lock.

---

## Artifact-format step — a REAL expert store and the scales/zp layout verdict (2026-09-23); the store precondition CLEARED

[`measured-here` + `code`; NO card leg, NO module load, NO synthetic-store
mutation. The arcwell module was found loaded and carved (the probe's
inherited state) and was left as found.] This leg resolves both store
blockers LISBON-001 recorded: the ext4 store was synthetic arcwell test data,
and the scales/zp layout precondition was OWED. A real store now exists on the
ext4 partition and the layout question is decided.

### 1. What the synthetic store was, and why it blocked the fill (carried)

`measured-here` (B60 probe 2026-09-23): the store was 1,700 files of
2,457,600 B, each one plain extent — but each file was one deterministic
4096-byte block repeated 600×, so it carried no expert tensors and no
scales/zp. The full-slice fill therefore had no correct source. That is the
blocker this leg removes.

### 2. The layout verdict: the slice is the three weight tensors, device order

`code` + `measured-here`. One Flash-Next expert slice is 2,457,600 B, and that
is exactly the three u4 weight matrices (`code`:
`src/exec/flash_next_offload.h:45`; re-derived
`3 * 2560 * 640 * 0.5`). The plugin's device slot and the OTD weight-file
layout are IDENTICAL for the weights (`[oc][ic]`, row-major, even linear index
low nibble) and DIFFERENT for scales/zp — device `[group][oc]` / `[group][oc/2]`,
file `[oc][group]`, transposed on upload by `maybe_transpose_scale_zp` (`code`:
`patches/0011-…:75-90`, `0006-…:291`).

**Verdict.** The store file carries the three packed u4 weight tensors,
concatenated gate | up | down, each `[oc][ic]` with the C++ nibble contract
(`src/core/gguf_repack.h:90`, `gguf_repack.cpp:225`). These bytes are
byte-identical in the file and on the device, so a naive full-slice DMA is
byte-transparent — **given the D2/D3 integration contract that the per-expert
device BO is the concatenated record** (the plugin today keeps three separate
per-tensor memories; this contiguous gate|up|down layout is the integration's,
not an existing device layout). **Scales and zero-points are NOT in the DMA
slice**; they stay on the existing host upload path, which already applies the
transpose. That is the design note §3's second option, and the alignment
arithmetic is decisive:

    weights                         2,457,600 B = 600 pages = 4,800 x 512 B LBAs
    serving-shape scale/zp (g=128)  scales 76,800 B + zp 19,200 B = 96,000 B
    weights + scales + zp           2,553,600 B = 623.4375 pages -> NOT page-aligned

The plugin's per-tensor scale destinations are themselves unaligned
(`dst_offset = slot * per_expert_size`; at g=128 every scale tensor is 25,600 B),
so a scale/zp DMA on the existing device-slot offsets is not page-aligned
(`code`: `USING_ARCWELL.md` §6). Padding the record or giving scales/zp their
own page-aligned files is conceivable, but the chosen route is the one that
keeps the DMA ONE page-aligned request at the pinned 2,457,600-byte slice
WITHOUT changing the plugin's device-slot offsets — and the plugin's CPU kernel
and existing host path read the FILE `[oc][group]` order, so scales/zp need no
new indexing on that path (`code`: `patches/0011-…:79-80`).

The alternative — scales/zp file-resident already in DEVICE order — is
implemented for a future integration (`--with-scale-zp`;
`transpose_scale_zp_to_device`, a transcription of `maybe_transpose_scale_zp`'s
`src[o*group_count+g] -> dst[g*oc+o]`, with the f32 scale rounded to the f16
bits the artifact carries) and is exercised by a writer-level cell and a
sidecar-size cell, but it is not the default because it cannot be one
page-aligned request.

### 3. The writer tool and its red-first ladder

`code` + `measured-here`, device-free. New `tools/q4e/expert_store.py` writes
per-expert files through the existing fill machinery (`q4e.expert_fill`'s
`ExpertFiller`, `q4e.gguf_feed`): it gathers only the pinned experts BEFORE
dequantising, quantises to the u4 grouped-affine form, packs with the C++
contract, and writes with `os.posix_fallocate` first so the blocks are allocated
contiguously. Files are named `expert_NNNN.bin` — the arcwell reference tool's
own convention (`stub/tools/aw_fiemap.c` builds `expert_%04d.bin`) — and a
`manifest.json` maps each ordinal back to `(layer, expert)`.

`tools/test_expert_store.py` carries **15 cells** (green: `Ran 15 tests … OK`,
one geometry cell skipped where the temp filesystem does not report fallocated
blocks). Red-first by mutation, raw output in the store packet
(`cells-green.txt`, `redfirst-mutants.txt`):

| mutant | cell that fails |
|---|---|
| role order swapped (gate,d​own,​up) | `test_record_is_gate_up_down_in_that_order` |
| `truncate` instead of `posix_fallocate` | `test_fallocate_is_called_and_size_is_exact` |
| scale transpose made identity | `test_scale_transpose_matches_moe_otd_mapping` + `test_sidecar_scales_are_f16_bits` |
| short-payload refusal removed | `test_short_payload_is_refused` |
| sidecar scales passed through as f32 | `test_record_offsets_and_sidecar_size` + `test_sidecar_scales_are_f16_bits` |

A writer-level cell runs ONE expert through the whole writer (record offsets
gate `[0,819200)` / up `[819200,1638400)` / down `[1638400,2457600)` and the
96,000-byte f16-bit sidecar); the byte-exactness cell packs a real
quantisation, writes it, reads the FILE back, unpacks it with a transcription
of the C++ (`gguf_repack.cpp:225`), dequantises with the IR's own chain, and
asserts the reconstruction is within half a group step — AND that the bound is
tight, so it is not vacuous.

### 4. The bounded real store

`measured-here`. Pinned set: patch 0018's static partition at the gate's ratio
86 — `512*(100-86)/100 = 71` slots/layer (`code`: patches 0041/0047) × 48 layers
= **3,408 experts**, seed `0xF2A17C0DE5EED` (patch 0018's `kStaticPartitionSeed`).
Membership is a pure function of `(seed, layer_key)`; `layer_key` is the layer's
first OTD weight-file offset (`code`: patch 0018).

    slice           = 2,457,600 B
    experts         = 71 x 48 = 3,408
    pinned bytes    = 3,408 x 2,457,600 = 8,375,500,800 B = 7.80 GiB

It fits on the same ext4 partition as the synthetic store (187 G, 17 G free
after; `du` 7.9 G). The synthetic store was NOT touched — a `measured-here`
`stat` of its first and last files shows their mtimes unchanged from 2026-09-15
— and the GGUF and artifact were not modified.

**Membership caveat, stated not smoothed.** This store holds the *splitmix64*
ratio-86 set — the plugin's default seed, which is what the acceptance harness
(`docs/window-053.md`, plugin `ov-0047`, no census seed) pins. The design note
§5 prefers pinning once from an offline census seed (patch 0046); a gate run
with a census seed has a different membership and needs the writer re-pointed at
that seed file (`--pinned` / `--layer-keys`). The store is keyed by
`(layer, expert)`, so the mechanism is seed-agnostic; only the materialised
membership is not.

### 5. Geometry proof on the real store (raw)

Pasted raw output on the persistent packet; the accepted lines, verbatim:

```
## extent count over all 3408 (filefrag)
   3408 1
## non-plain flags, every 7th file (487)
0
## last,eof, every 7th file (487)
487
## verbose sample
File size of expert_0000.bin is 2457600 (600 blocks of 4096 bytes)
   0:        0..     599:   47775744..  47776343:    600:             last,eof
## st_blocks sample
expert_0000.bin 2457600 4800
## alignment
2457600/4096=600 r0; /512=4800 r0; /65536=37 r32768
```

`aw_fiemap` (repo `stub/tools/aw_fiemap.c`) byte-verifies ALL 3,408 files
against the raw device through the partition-start translation, and its
`--mutate` red leg fails on CONTENT:

```
3408 files, 3408 extents total (1.00 per file), 3408 verified against the raw device
RESULT=PASS -- FIEMAP + partition offset gives the correct absolute LBA
RED: <store>/expert_0000.bin ... -> lba=382205952 ... MISMATCH
RESULT=FAIL -- raw device at the computed LBA is not the file's bytes
```

Every file is 600 pages and 4,800 LBAs; the three weight tensors' record is
byte-transparent to the device weights layout.

### 6. Byte-exactness on the real store

`measured-here`. Manifest size+sha256 over all 3,408 files: **0 mismatches**.
A 24-expert sample (× 3 roles) recomputed from the GGUF source: the file's bytes
unpack to exactly the quantised codes, and the dequantised reconstruction sits
within the u4 half-step — worst error/bound **1.000001**, i.e. at the
representation's floor (the `_BOUND_SLACK = 1e-5` measured in
`tests/python/test_expert_fill.py`).

### 7. What this changes, and what stays OWED

This clears `docs/window-053.md` dependencies 1 and 2 (the synthetic store and
the scales/zp precondition). The gate is no longer blocked because *nothing can
be filled correctly* — a correct source now exists. What stays OWED is
unchanged and is NOT discharged here: the integrated load-time fill (the
`AW_IOC_SUBMIT_BATCH` / `AW_IOC_BATCH_WAIT` schedule wired inside the serving
loop, design note D2/D3), the red cell that a synchronous `AW_IOC_READ_BLOCKS`
on the decode path is refused, and the three gate rows themselves (cold TTFT
both arms in one window, byte-identity across arms and two cold boots, decode
non-regression). No card leg was run.

**Evidence classes, this leg.** Slice bytes and the layout divergence: `code`
(`flash_next_offload.h:45`; patches 0011/0006) plus `measured-here` arithmetic.
The writer, its ladder and mutants: `code` (the tool) + `measured-here` (the
runs). The store's geometry, extent count, raw-device verification and
byte-exactness: `measured-here` on the ext4 partition. The seed and pinned-set
arithmetic: `code` (patch 0018; patches 0041/0047). The gate and the consumer
integration: **OWED**.

---

## D2/D3 consumer integration — the load-time pinned fill's schedule, wired plugin-side; the byte destination OWED (2026-09-23)

[`code` + `measured-here`; NO card leg, NO module load, NO expert-store
mutation, NO host/store change. The arcwell module was found loaded and carved
(the B60 probe's inherited state, re-confirmed on the card host) and was left
as found. Cards idle.] This leg builds the D2/D3 consumer the design note
schedules — the last item between the real store and the LISBON gate — and
records, without smoothing, the one part that cannot land yet.

### 1. Where it lives, and why: plugin-side provider

The design note's §3.4 fixes the handshake in the **plugin**: an expert the
batch surface collected "must call the cache's `set_filled(slot)`"
(`code`: patch 0018). The pinned membership is also plugin-side
(`static_partition_resident_experts`, patch 0018; the census seed, patch 0046),
and the slot the fill must mark is the plugin's LRUCache slot. So the
integration is **plugin-side**, in `OffloadExpertWeightProvider` — not an
arcint-side runtime that would have to reach into the plugin's cache through a
new interface. The schedule's contract is tracked device-free as
`src/exec/pinned_nvme_fill.h` (the header the arcint test ladder exercises) and
shipped to the plugin byte-identically as `moe/pinned_nvme_fill.hpp` in patch
0048, because the plugin build cannot see `src/exec` (`code`; a drift-guard
cell asserts the two copies are byte-identical).

### 2. The scheduled behaviour (what the cells pin)

`paper` + `code`. Membership is taken once at `bind()` from the static
partition — never re-derived from live traffic (RED-C-02's finding). The
schedule is:

- **One batch per layer.** `FillBatch` is one MoE layer's pinned set (71
  requests at ratio 86, 128 at ratio 75; both under `AW_BATCH_MAX = 256`).
- **Four batches in flight.** The `Scheduler` fills the window to depth 4
  before collecting the oldest, so the NVMe queue depth is used (the note's
  measured four-overlap shape).
- **Collect marks filled.** Every request of a fully collected batch calls the
  caller's `on_filled`, wired to `LRUCache::set_filled(slot)`, so the existing
  first-use `fill_weights_memory` branch never fires on the decode path.
- **No fetch on the decode path.** The `Transport` interface exposes only
  setup/submit/collect; it has no synchronous read. `require_no_sync_read()`
  refuses a would-be synchronous `AW_IOC_READ_BLOCKS` once serving has begun.
- **A short batch is retried once, in full, then REFUSED.** Partial completion
  does not name which request landed, so the whole layer is re-issued; if it is
  still short, the barrier returns false with a reason naming the layer, and
  nothing from that layer is marked filled. **The refusal is terminal** — the
  phase becomes `kRefused`, and a second barrier refuses too, so a partially
  filled configuration can never resume into serving. No silent demotion to the
  host tier — a demoted expert would make this boot's residency differ from a
  clean boot and break the byte-identity the gate measures across two cold
  boots (§3.4).
- **Setup failure is a load failure.** If arcwell cannot be set up at all, the
  barrier refuses for the whole configuration; there is no host-bounce path.

In the plugin, a translation-unit global coordinator starts at the first
layer's `bind()`, enumerates the live providers in structural `layer_key`
order (not the run-to-run construction permutation), reserves every layer's
slots first (a layer whose own `bind()` has not run has no pinned slot to
mark), then runs the barrier. Any failure throws — a load failure, per D3. The
whole path is opt-in (`MOE_OTD_PINNED_NVME_FILL`); unset, patch
0018/0046/0047 behavior is unchanged.

### 3. The byte destination is OWED — measured, not guessed

`measured-here` (B60 probe, carried) + `code`. The note's D2 presupposes a
per-expert **device** destination: arcwell's contract is a dma-buf exported
from an xe VRAM BO (`code`: `aw_uapi.h`, `USING_ARCWELL.md` §6), and the
artifact-format step named "the per-expert device BO is the concatenated
record" as the D2/D3 contract (not an existing device layout). But under the
static partition the plugin's expert slot pool is **host-mapped**: the B60
probe's own `[OTD_PERF]` line reads `device_slot_buffers=0` at both ratio 86
(71 slots) and ratio 83 (87 slots), with `host_slot_buffers=382` and the
0.37 GiB device term a small working set against a 7.91 / 9.67 GiB host-side
ceiling (`measured-here`, this campaign's B60 probe section). The slot buffers
are `allocation_type::usm_host` (`code`: `moe_otd_runtime.cpp`), so there is no
dma-buf VRAM BO for the fill to land in, and arcwell has no host-bounce path by
design. **The transport therefore has no production implementation in this
patch.** An enabled fill refuses the load with that reason — exactly the note's
"arcwell cannot be set up at all is also a load failure" rule — and the three
gate rows stay OPEN.

**OWED, named and not faked:** the per-expert dma-buf BO-backed slot
destination, the arcwell ioctl `Transport`, the integrated serving step with
the fill overlapping, and the card validation of the D2/D3 wiring (design note
§7). None of these is discharged here.

### 4. Deliverable 1, build evidence

`measured-here`. Patch
`patches/0048-moe-otd-pinned-nvme-fill.patch`, sha256
`3f94609c7e8ccacdca81b39c8625cd205e40e0667248bb6f008338749c419347`, mirrored
byte-for-byte into the packaging series. It applies cleanly to a **pristine
checkout of the pin through the full sequential series 0003–0047** and then
0048:

```
$ for p in .../patches/*.patch; do git apply --check "$p" && git apply "$p"; done
applied 0003-...  ...  applied 0047-moe-per-expert-slot-pool-size.patch
$ git apply --check /.../0048-moe-otd-pinned-nvme-fill.patch && git apply ...
0048 APPLIED on the full sequential series
$ git status --porcelain | grep pinned_nvme
 M src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/expert_weight_providers.cpp
 M src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/expert_weight_providers.hpp
?? src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/pinned_nvme_fill.hpp
```

Compile-verified against the 0047 tree with the production target (rc 0; the
two changed TUs plus the include-dependent `moe_3gemm_swiglu_opt.cpp`):

```
$ ninja openvino_intel_gpu_plugin
[2/6] Building CXX object .../moe/expert_weight_providers.cpp.o
[3/6] Building CXX object .../moe/moe_3gemm_swiglu_opt.cpp.o
[4/6] Linking CXX static library .../libopenvino_intel_gpu_graph.a
[5/6] Linking CXX shared module .../libopenvino_intel_gpu_plugin.so
```

No card leg; nothing was installed; the working tree was restored to its
0047 state after the check.

### 5. Deliverable 2/3 — the red-first cells and their mutation evidence

`measured-here`, device-free. New `tests/test_pinned_nvme_fill.cpp`
(`code`), sha256 `c9a76a96077b664ac16ed57af41e018e1bc93ce2552c97d76c06ad77301f7596`,
built into `arcint-test` (`build` tree, stub backend — the schedule is pure
C++). **8 cells green**:

```
$ ./arcint-test pinned_nvme_fill
8 cases run, 0 failed, 0 skipped
```

The two cells the campaign names:

- `pinned_nvme_fill_refuses_a_synchronous_read_on_the_decode_path` — the
  losing configuration: `require_no_sync_read` does not throw during the load
  phase and **does** throw once the barrier has advanced to serving.
- `pinned_nvme_fill_refuses_a_pinned_expert_that_never_lands` — the barrier
  returns false, names the layer, never marks that layer's requests filled, and
  becomes terminally refused (`kRefused`); exactly one retry is issued, and a
  second barrier call refuses rather than resuming.

Batch accounting (`one_batch_per_layer`, `holds_four_batches_in_flight`) and
the collect→filled transition (`collect_marks_filled` in the one-batch cell;
retry-then-land in `short_batch_is_retried_then_lands`) are pinned beside them.

**Mutation evidence** (raw in the evidence packet; each mutant edits the
tracked header in place, rebuilds, runs the named cell, restores the header
byte-identically):

| mutant | named cell | result |
|---|---|---|
| `require_no_sync_read` guard removed (no throw) | `refuses_a_synchronous_read_on_the_decode_path` | **1 failed** (`expected: threw_serving`) |
| refusal replaced by a silent fill (`on_filled` for every batch, return true) | `refuses_a_pinned_expert_that_never_lands` | **1 failed** (barrier true; layer 3 filled; phase serving) |
| depth gate ignored (submit all before collecting) | `holds_four_batches_in_flight` | **1 failed** (`max_inflight` 10, expected 4) |

### 6. What this changes

`docs/window-053.md` dependency 3 ("the consumer does not exist") is **partly**
answered: the schedule exists, is wired into the plugin's `bind()`/load
barrier, applies on the full series, and compiles; the red cell the campaign
named is written and mutation-tested. But the fill cannot LAND — no
transport, no BO destination — so the gate's "serving step with the fill
overlapping" still has no number. Dependency 3 therefore **stays standing**,
updated in place with this date, and the three rows stay OPEN. The gate is now
blocked on the dma-buf BO-backed slot destination, not on a missing schedule.

**Evidence classes, this leg.** Placement and the schedule (batch count, depth,
collect, refusal, guard): `paper` (design note §3/§4) + `code` (patch 0048,
`src/exec/pinned_nvme_fill.h`). The destination finding: `measured-here` (B60
probe's `device_slot_buffers=0`) + `code` (`aw_uapi.h`; `moe_otd_runtime.cpp`).
The patch/apply/compile evidence and the cell ladder: `measured-here`. The BO
destination, transport and card validation: **OWED**.

---

## D2/D3 byte destination — SETTLED AND PROVEN: a caller-created xe VRAM BO, exported as a dma-buf (2026-09-24)

[`code` for the mechanism; `measured-here` for the proof. One card leg on the
B60 alone; NO `arcint` leg, NO module load/unload, NO expert-store write. The
arcwell module was found loaded and carved (inherited) and left as found.]
This leg settles the one item the D2/D3 integration recorded as OWED: **how the
plugin obtains a VRAM BO whose dma-buf meets arcwell's mapping contract** — and
proves it end-to-end at the smallest scale, without arcint.

### 1. The mechanism, decided from arcwell's own source

**arcwell provides NO allocator/helper. The caller creates the xe BO and
exports the dma-buf itself.** Three independent sources say so, and they agree:

- `stub/src/arcwell.c`'s own header: *"userspace creates a host-visible VRAM
  BO on xe and exports it as a dma-buf, then hands us the fd"* (`code`:
  `~/src/arcwell/stub/src/arcwell.c:7-8`).
- `M4_API.md`: *"The contract lives in the uAPI, not in a client library …
  Any thin client that wraps these ioctls exists only to issue them; it is not
  a data path and holds no bytes"* (`code`; the same record states the earlier
  client-library revision *"has been removed"*).
- `KERNEL_FACTS.md`, "The working recipe, no xe patch required", step 1:
  *"Userspace creates the BO (WC + `NEEDS_VISIBLE_VRAM` + 64K-aligned) and
  exports it with `DRM_IOCTL_PRIME_HANDLE_TO_FD`"* (`code`).

The uAPI confirms it by omission: there is no BO-create ioctl. `AW_BUF_XE_GEM`
is declared but `0.0.1` implements **only** `AW_BUF_DMABUF` — `aw_map_buffer`
returns `-EOPNOTSUPP` for any other `in_source` (`code`: `aw_uapi.h`;
`arcwell.c:486-489`). The client's once-at-open lifecycle is
`USING_ARCWELL.md` §3, verbatim: create the BO and export it;
`AW_IOC_MAP_BUFFER` with `in_length = BO size`; then *"assert `out_flags &
AW_MAP_F_REQUIRE_P2P` … this check is not optional"*.

**Chosen path: direct xe DRM ioctls from the consumer**, vendoring the DRM
constants, exactly as arcwell's own `stub/test/aw_expert_test.c` and
`aw_async_test.c` do (`code`): `DRM_IOCTL_XE_GEM_CREATE` with
`placement = 1 << DRM_XE_MEM_REGION_CLASS_VRAM`,
`flags = DRM_XE_GEM_CREATE_FLAG_NEEDS_VISIBLE_VRAM`,
`cpu_caching = DRM_XE_GEM_CPU_CACHING_WC`, size rounded to a 64 KiB multiple;
then `DRM_IOCTL_PRIME_HANDLE_TO_FD`. The vended header is `xe_drm.h`
(arcwell's tree) or the system `<drm/xe_drm.h>`; the ioctls are identical.

**Alternatives rejected, with the source line that rejects them:**

| alternative | why rejected | class |
|---|---|---|
| arcwell's own helper/library | there is none in the current contract. The tree carries an **untracked, stale** `libarcwell.a`/`arcwell.o` whose symbols (`aw_backend_stages_via_system_ram`, `aw_unbuilt_reason`) identify it as the pre-`0.0.1` userspace-staging client that `M4_API.md` says was removed; it does not create an xe BO, and `KERNEL_FACTS.md` is explicit that *"There is no userspace data path and none is to be designed"* | `code` |
| `AW_BUF_XE_GEM` (an xe GEM handle directly) | declared in `aw_uapi.h` but unimplemented at `0.0.1`; `aw_map_buffer` returns `-EOPNOTSUPP` for anything but `AW_BUF_DMABUF` | `code` |
| L0 / OpenCL export | OpenCL cannot *create* a `NEEDS_VISIBLE_VRAM`+WC BO; it can only *import* a dma-buf (`cl_khr_external_memory_dma_buf`, handle type `0x2067`), which is the **consumption** half, not the creation half (arcwell's own `aw_cl_import_test` E2E proves the import; `KERNEL_FACTS.md` "END TO END") | `code` |

**Lifetime and ownership.** The BO's GEM handle is owned by the opening
process's render-node fd; the exported `pr.fd` is a separate dma-buf reference.
`AW_IOC_MAP_BUFFER` takes the module's **own** reference (`dma_buf_get`,
`arcwell.c:495`), so the caller may close `pr.fd` immediately after a
successful map — arcwell's own tests do exactly that. The registration is
released by `AW_IOC_UNMAP_BUFFER` or by closing the `/dev/arcwell` fd
(`aw_free_buffer` detaches, unpins and drops the reference; `USING_ARCWELL.md`
§3: *"Buffers are owned by the file descriptor"*). In-flight batches hold buffer
references until collected (`aw_uapi.h`: *"Buffer references are held until the
batch is collected"*), so a buffer must not be unmapped under a live batch;
closing the fd drains.

**Can the plugin hold it inside the provider without breaking OpenVINO's
allocator?** Yes: it is a raw DRM BO with its own fd, **not** an OpenVINO
engine allocation, so it does not pass through `allocate_memory`. To let an
OpenCL kernel read it, the provider imports the dma-buf
(`cl_khr_external_memory_dma_buf`, `0x2067`) into a `cl_mem` — the path
arcwell's own E2E cell proved (`code`: `aw_cl_import_test.c`;
`KERNEL_FACTS.md`). The slot descriptor must then point at that imported
`cl_mem` rather than an engine buffer. That integration (plugin-side) is not
proven in this leg — this leg proves the byte path only.

### 2. The proof (smallest scale, non-arcint)

`measured-here` on the B60. New standalone client
`tools/arcwell_bo_dma_proof.c` (tracked; sha256
`8ee9ee5187cf4913f7d26dc981fe9aaccdaabac028d13aa35f60699c4da1a211`), built on
the card host against `<drm/xe_drm.h>` and `~/src/arcwell/stub/include/aw_uapi.h`.
It creates the BO, exports it, registers it, transfers **one real 2,457,600 B
payload from the real store** (byte-transparent artifact, the artifact-format
step's store), verifies the BO by host readback through its own xe mapping, and
reads `AW_IOC_STATS` as a delta. Re-read of the arcwell tree at this date: HEAD
`2e9257a`, **tree `e7d326e`** — the same tree the 2026-09-23 recon pinned, so
the cited sources are unchanged.

Command (`<render-node>` is the B60 node identified by PCI id `8086:E211`; the
operator-local path is in the handoff packet):

```
$ ./arcwell_bo_dma_proof --drm <render-node> --file <store>/expert_0000.bin \
      --part-start <sectors> --readback-out <packet>/readback-normal.bin
==== arcwell BO -> dma-buf -> arcwell -> host readback [normal - must PASS] ====
BO: size=2490368 B (38.00 x 64 KiB) handle=1 placement=0x2 flags=0x4
PRIME: dma-buf fd=4
MAP_BUFFER: handle=16 out_flags=0x1 (AW_MAP_F_REQUIRE_P2P honoured)
FIEMAP: <store>/expert_0000.bin extents=1 phys=195689447424 -> absolute LBA=482871296 len=2457600
SUBMIT: batch_id=1 submitted=1 err=0
poll: saw_eagain=1 polls=49 collected=1
COLLECT: bytes=2457600 completed=1 segments=3 err=0
STATS delta: via_host_bounce 0->0 max_inflight 261->261 batches 23->24 batch_reads 800->801 segments 2400->2403 bytes 1966080000->1968537600
READBACK: wrote 2457600 B to <packet>/readback-normal.bin
READBACK: 2457600 B byte-identical to the store file
RESULT=PASS -- xe VRAM BO dma-buf registered peer-to-peer, one real expert landed by controller DMA, host readback byte-identical, via_host_bounce delta 0, max_inflight 261
```

Host-readback byte identity (sha256 of the store file and of the BO's readback):

```
4a4bb0f91361e4b184d8c151fc9bdddee29626d667fda8b89928106592983f9a  <store>/expert_0000.bin
4a4bb0f91361e4b184d8c151fc9bdddee29626d667fda8b89928106592983f9a  <packet>/readback-normal.bin
```

**Reading.** One real expert landed in a caller-created VRAM BO by controller
DMA: BO 2,490,368 B (the 64 KiB round-up of the 2,457,600 B payload),
peer-to-peer registration honoured, 3 segments (the bio floor for 600 pages),
`via_host_bounce` delta **0**, `max_inflight` 261 (absolute, the module-global
high-water mark, `> 1` as the design note requires), and the BO's own
host readback **byte-identical** to the store file. This is a **DMA payload
read back to the host**, so the byte comparison is legitimate and the B60
compute determinism caveat does not apply.

### 3. Red-first cells (mutation-testable)

`measured-here`. The tool carries five mutation legs; every one MUST report
`RESULT=FAIL`, and the tool's terminal guard refuses to let a mutation leg pass
(`"the mutation did NOT change the outcome"`). Four go red as required; the
fifth is a measured counterexample to a contract claim (see §4).

**`--mutate-no-part-offset`** — drop the ext4 partition start; the DMA reads the
wrong bytes and the readback must mismatch:

```
MUTATED: partition start dropped  -> absolute LBA=382205952
RESULT=FAIL -- readback mismatch at byte 0 (expected for this mutation)
```

**`--mutate-offset-unaligned`** — `in_dest_offset=512` (not page-aligned); the
module must refuse it:

```
SUBMIT: batch_id=1 submitted=0 err=-22
SUBMIT_BATCH refused the unaligned in_dest_offset=512: submitted=0 err=-22 (out_err is the submission-time error; the ioctl return is not enough)
RESULT=FAIL -- [MUTATED] unaligned transfer geometry refused
```

**`--mutate-system-bo`** — a system-memory BO (the host-bounce configuration);
`MAP_BUFFER` must refuse it:

```
MAP_BUFFER refused a system-memory BO: Numerical result out of range
STATS after refusal: via_host_bounce 0->0 (delta 0)
RESULT=FAIL -- [MUTATED] host-bounce configuration refused
```

**`--mutate-readback`** — corrupt the expected bytes; the readback must fail:

```
RESULT=FAIL -- readback mismatch at byte 1228800 (expected for this mutation)
```

**`--mutate-bo-size`** — the BO size is **not** rounded to 64 KiB. **This leg
does not go red on the B60**, and that is the finding, not a shortcut.

### 4. Findings that correct the record

Three dated corrections follow from the raw output; none is smoothed.

**(a) The 64 KiB BO gate is NOT enforced on the B60.** `~/src/arcwell/
KERNEL_FACTS.md` states the four `DRM_IOCTL_XE_GEM_CREATE` requirements are
enforced — *"size a multiple of 64 KiB (DG2 flat-CCS sets the VRAM manager
`min_page_size = 64K`)"* — and that omission *"returns `-EINVAL`"*. That was
measured on the **A770 (DG2)**. On the B60, a 2,457,600-byte BO (37.5 × 64 KiB)
was **accepted end-to-end**:

```
NOTE: [MUTATED] GEM_CREATE ACCEPTED a non-64KiB size (2457600 B); continuing ...
BO: size=2457600 B (37.50 x 64 KiB) handle=1 placement=0x2 flags=0x4
MAP_BUFFER: handle=19 out_flags=0x1 (AW_MAP_F_REQUIRE_P2P honoured)
COLLECT: bytes=2457600 completed=1 segments=3 err=0
RESULT=FAIL -- [MUTATED] the mutation did NOT change the outcome; this leg proves nothing
```

The readback for that run is byte-identical (`readback-bo-size.bin` sha256
`4a4bb0f9…`). So the 64 KiB rule is a **client contract**, not a kernel-enforced
gate on the B60 (kernel 7.0.14); the B60 does not enforce a 64 KiB gate for this
BO on this kernel (the granule itself was not measured, only that the
acceptance shows no 64 KiB gate). The safe path is unchanged — round the BO up
per `USING_ARCWELL.md` §6 and `KERNEL_FACTS.md`, because the contract is written
for both cards and the A770/DG2 gate is card-specific. A red cell for this rule
cannot be made on the B60.

**(b) A submission-time geometry error is reported in `out_err`/`out_submitted`,
not by the ioctl return.** `AW_IOC_SUBMIT_BATCH` returned **0** yet
`out_submitted=0`, `out_err=-22` (`-EINVAL`) for the unaligned offset. The
`Transport::submit` implementation patch 0048 owes must therefore check
`out_submitted == in_count` and `out_err == 0`, not the ioctl return alone
(`code`: `aw_uapi.h`, *"`out_err`: submission-time error only"*; observed
`measured-here`).

**(c) A system-memory dma-buf is refused with `-ERANGE` before the
`via_host_bounce` increment sites.** The refusal is real (no path bounces), but
the counter stayed 0 because the failure is the carve range check
(`aw_try_carve`: `-ERANGE`, `arcwell.c:302`) — the requested range lies outside
usable VRAM — which runs before the two documented increment sites
(`arcwell.c:509` peer2peer clear; `arcwell.c:557` non-P2PDMA page). The
`via_host_bounce` delta of 0 is therefore expected for **this** refusal and does
not weaken the guard: dmesg carries `could not carve for phys … at any granule
… : -34` (`-ERANGE`), and the map failed.

### 5. Implication for the fill's destination (explicit)

The proof changes the destination from an open question to a concrete object:
**the fill's destination must be a caller-created xe VRAM BO**, created and
owned by the plugin/provider (raw DRM ioctls), exported as a dma-buf,
registered with `AW_IOC_MAP_BUFFER` peer-to-peer, and imported into OpenCL by the
same dma-buf fd for the consuming kernel. The static partition's current slot
buffers are host-mapped (`usm_host`, `device_slot_buffers=0`), so they cannot be
the destination; the provider's `Transport::begin` becomes "create the per-slot
BO, export, map", `submit` drives `AW_IOC_SUBMIT_BATCH` (checking
`out_submitted`/`out_err`), and `collect` drives `AW_IOC_BATCH_WAIT` — the
schedule already tracked in `src/exec/pinned_nvme_fill.h` / patch 0048.
The per-expert BO is the smallest scale proved here; the artifact-format step's
concatenated per-layer BO (one request per expert at a page-aligned
`in_dest_offset`) is the same contract, and either is admissible. Registering
many BOs is not a BAR2 hazard in the current tree: carves are section-sized
(128 MiB) and reused (`carve_covers`, `arcwell.c:265`; `aw_pages_present`,
`arcwell.c:348`),
and this leg registered several BOs without adding a carve.

**What this does NOT discharge:** the plugin-side transport wiring, the OpenCL
import into OpenVINO's slot descriptors, and the LISBON gate's "serving step
with the fill overlapping" number. No arcint leg was run.

### 6. Card, lock, module state

- **Before:** `pgrep -x arcint` empty; no competing service (`ollama`/`vllm`/
  `llama-server` empty); B60 (`8086:E211`) `power/control=on`, runtime
  **active**; A770 (`8086:56A0`) **suspended**, untouched; `arcwell` loaded and
  carved (inherited), `/dev/arcwell` present; physical-host `MemAvailable`
  32.9 GiB (minimum 34,471,692 kB, the re-run sampler's first row).
- **One card leg at a time:** the B60 alone; the A770 was never opened.
- **Wake lock (coordinator host):** found none set (`keine Sperre gesetzt`);
  taken for 4 h with a reason naming only the campaign and the B60 (no host
  name), and **released** when the card work was done. No foreign lock
  overwritten.
- **Sampler (SOP §1):** the physical-host sampler ran for the leg with the
  4 GiB watchdog on `MemAvailable`. Minimum observed **34,471,692 kB = 32.9 GiB**;
  **0 watchdog trips**.
- **Store:** read-only; no file written or moved.
- **After:** no `arcint`; `arcwell` **left loaded and carved** as found (unloading
  a carved GPU is the `~/src/arcwell/KERNEL_FACTS.md` half-state hazard); no new
  carve line from this leg; B60 runtime **active**, `power/control=on`; A770
  untouched.

### 7. Records changed, and what remains

`docs/window-053.md` dependency 3 is updated **in place** with this date: its
**destination** clause — "there is no destination a byte-transparent fill can
land in" — is **CLEARED** (the BO-backed destination is settled and proved),
while its **consumer-integration** clause stays standing (no plugin transport,
no overlapping-step number). The three gate rows stay OPEN. Design-note §9
records the same disposition.

**Evidence classes, this leg.** The mechanism and rejected alternatives: `code`
(`arcwell.c`, `aw_uapi.h`, `USING_ARCWELL.md`, `M4_API.md`, `KERNEL_FACTS.md`).
The end-to-end proof, the sha256 byte identity, the five mutation legs and the
three corrections: `measured-here` (B60). arcwell's own E2E OpenCL-import cell:
`code` (arcwell's tree) + arcwell's `measured-here`. The plugin integration and
the gate: **OWED**.

**Raw output** (verbatim, unredacted, with the operator-local paths only in the
git-ignored packet): `normal.txt`, `mutate-no-part-offset.txt`,
`mutate-offset-unaligned.txt`, `mutate-bo-size.txt`, `mutate-system-bo.txt`,
`mutate-readback.txt`, `sha256sums.txt`, `sampler2.log`, and the tool source on
the persistent evidence path; hashes in the packet.

---

## D2/D3 plugin transport + OpenCL slot import — built and mechanism-proven; the served gate stays OWED (2026-09-24)

[`code` + `measured-here`. Plugin patch `0049` build-verified device-free; one
B60 leg for the mechanism. NO served run; the three gate rows stay OPEN. The
arcwell module was loaded fresh (it had been unloaded by the host's sleep) and
left loaded; see the packet.] This leg supplies the last two items the
byte-destination proof recorded as OWED: the plugin-side `Transport`, and the
OpenCL import that makes the BO-backed slot the destination the resident expert
is read from.

### 1. The transport (patch `0049`, `moe/pinned_nvme_transport.hpp`)

`code`. `lgc::nvme_fill::ArcwellTransport` implements the `Transport` interface
patch `0048` injected empty. Per the destination proof's decided mechanism, it
never asks arcwell for an allocator (there is none): it creates each
64 KiB-rounded xe VRAM BO itself with the raw DRM ioctls
(`DRM_IOCTL_XE_GEM_CREATE` with VRAM placement + `NEEDS_VISIBLE_VRAM` +
`CPU_CACHING_WC`, then `DRM_IOCTL_PRIME_HANDLE_TO_FD`), registers it
peer-to-peer (`AW_IOC_MAP_BUFFER`, asserting `AW_MAP_F_REQUIRE_P2P`), and
drives `AW_IOC_SUBMIT_BATCH` / `AW_IOC_BATCH_WAIT` with the schedule's one
batch per layer, four in flight. `submit()` checks `out_submitted`/`out_err`,
not the ioctl return alone (the dated correction: an unaligned geometry returns
0 with `submitted=0`, `err=-22`). Lifetime: the BO's GEM handle is owned by the
render-node fd; the exported dma-buf is a separate reference; arcwell takes its
own `dma_buf_get` at `MAP_BUFFER`, so the caller may close the exported fd after
a successful map; registrations release by `AW_IOC_UNMAP_BUFFER` or fd close;
in-flight batches hold the buffer until collected.

### 2. The OpenCL import into the slot descriptors

`code`. `bind_pinned_nvme_pool()` imports each registered dma-buf with
`engine.import_buffer()` (`code`: `src/plugins/intel_gpu/src/runtime/ocl/
ocl_engine.cpp:111-165` — `clCreateBufferWithProperties` with
`CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR` `0x2067` + the fd, then
`cl::ExternalMemoryHelper::acquire`; `ocl_ext.hpp:347` defines the handle
constant; read on the build host) and **replaces this layer's host-mapped
`gate_w`/`up_w`/`down_w`** with a `reinterpret_buffer()` of the imported pool.
The fused GEMV kernel indexes `gate_weight_addr + expert_id * expert_wei_size`
(`code`: `moe_3gemm_swiglu_mlp.cl:516`, a build-host read), so the per-tensor
pool layout keeps the stride and the full-layout wrapper keeps
`expert_tensor_span()`'s `total_bytes/num_expert` arithmetic; `bind_pinned_nvme_
pool()` now asserts that equality rather than assuming it. What stays on the host path: the six
scale/zp tensors, which the artifact-format verdict excludes from the DMA slice
(weights + scales + zp = 2,553,600 B = 623.4375 pages, not page-aligned, and
the device wants `[group][oc]` vs the file's `[oc][group]`). They are
host-uploaded here, at load, on the engine's service stream
(`fill_weights_memory(..., include_weights=false)`), completing each pinned
slot before the first routed call. A non-resident expert never enters this
path: it still takes the host tier at the existing static-partition branch.

**Geometry correction, dated.** The store record is `gate|up|down`
concatenated; the plugin's device layout is three per-tensor slot regions, so
ONE expert is THREE page-aligned requests (each 819,200 B = 200 pages), not
one. The design note's "one expert = one request" held only for the synthetic
store whose whole file mapped to one destination. The store ordinal is
layer-major — `dense_layer * capacity + slot`, file `expert_%04u.bin` — and was
verified against the manifest (0 mismatches over all 3,408).

### 3. The B60 mechanism proof (`tools/arcwell_cl_slot_proof.c`)

`measured-here` on the B60 alone (`8086:E211`; the A770 untouched). A tracked
non-arcint client, run in the container (which now has `/dev/arcwell` and the
store mounted), creates three per-tensor VRAM BOs, DMA's two real store experts
(six requests), imports every dma-buf into OpenCL, and reads the BOs back
through the OpenCL queue. Verbatim apart from the operator-local path tokens
(`<render-node>`, `<store>`, `<packet>`):

```
drm=<render-node> store=<store> n=2 capacity=4 tensor=819200 region=3276800 bo=3276800
BO[0]: gem=1 prime_fd=5 aw_handle=22 size=3276800 out_flags=0x1
BO[1]: gem=2 prime_fd=6 aw_handle=23 size=3276800 out_flags=0x1
BO[2]: gem=3 prime_fd=7 aw_handle=24 size=3276800 out_flags=0x1
SUBMIT: batch_id=1 submitted=6 err=0
COLLECT: bytes=4915200 completed=6 segments=6 err=0 (of 6 requests)
OpenCL device: Intel(R) Arc(TM) Pro B60 Graphics (acquire/release present)
BO[0] imported into OpenCL
BO[1] imported into OpenCL
BO[2] imported into OpenCL
READBACK: wrote 4915200 B to <packet>/readback-cl.bin
STATS delta: via_host_bounce 0->0 max_inflight 6->6 batches 7->8 batch_reads 24->30 segments 24->30 bytes 19660800->24576000
READBACK: 4915200 B through OpenCL byte-identical to the store record
RESULT=PASS -- 2 real expert(s) DMA'd by controller into per-tensor VRAM BOs, imported into OpenCL on Intel(R) Arc(TM) Pro B60 Graphics, read back byte-identical, via_host_bounce delta 0, max_inflight 6, 6 requests
rc=0
```

**The STATS baseline is nonzero, and that is the measurement.** `AW_IOC_STATS`
is a module-global counter, so the absolutes carry the earlier mutation legs'
registrations (`aw_handle` 22/23/24, `batches 7->8`); this leg's own delta is
`batches +1`, `batch_reads +6`, `segments +6`, `bytes +4,915,200`, and
`via_host_bounce` **0→0**. A run against a freshly loaded module reads these
absolutes from zero (the counter is module-global, `aw_uapi.h`: read it as a
delta, never absolute); only the absolutes differ, not the per-leg delta.

Byte identity (a DMA payload read back through the OpenCL queue — legitimate
byte comparison; the B60 compute caveat does not apply). `expected-cl.bin` is
produced by concatenating, for each tensor t, the file's `[t*T, (t+1)*T)` slice
for slots 0..n-1 (`normal-cl.txt`'s own expected side; command in the packet):

```
d463d1d5fda90c6fd6364e2bf5f5c894a8ecb8810a309147146b69e0cb91587c  readback-cl.bin
d463d1d5fda90c6fd6364e2bf5f5c894a8ecb8810a309147146b69e0cb91587c  expected (store gate|up|down slices)
```

### 4. Red-first legs (all five measured RED)

`measured-here`. All five `--mutate-*` legs exit `rc=1` with their named
failure; each has its own raw transcript in the packet (`mut-*.txt`, hashes in
`sha256sums.txt`):

| leg | raw result | rc |
|---|---|---|
| `--mutate-offset-unaligned` | `RESULT=FAIL -- [MUTATED] unaligned transfer geometry refused` + `SUBMIT refused unaligned in_dest_offset: submitted=0 err=-22` | 1 |
| `--mutate-no-part-offset` | `RESULT=FAIL -- readback mismatch at byte 0 (expected for this mutation)` | 1 |
| `--mutate-system-bo` | `RESULT=FAIL -- [MUTATED] host-bounce configuration refused` + `MAP_BUFFER refused system-memory BO: Numerical result out of range (via_host_bounce 0->0)` | 1 |
| `--mutate-readback` | `RESULT=FAIL -- readback mismatch at byte 409600 (expected for this mutation)` | 1 |
| `--mutate-cl` | `RESULT=FAIL -- readback mismatch at byte 0 (expected for this mutation)` (corruption written through OpenCL) | 1 |

The campaign's named synchronous-read refusal stays covered by
`pinned_nvme_fill_refuses_a_synchronous_read_on_the_decode_path` in
`tests/test_pinned_nvme_fill.cpp` (patch `0048`), mutation-tested earlier: the
`Transport` interface has no synchronous read, and `require_no_sync_read()`
refuses a would-be `AW_IOC_READ_BLOCKS` once serving has begun.

### 5. Build evidence

`measured-here`. Patch
`0049-moe-otd-pinned-nvme-transport.patch`, sha256
`d6d3498d20fddf22b2972ba128c7b719dd8b759e4378a4b16f838296b54a630f`, mirrored
byte-for-byte (`cmp` of the two tracked copies is clean). It reverse-applies
and re-applies cleanly on the 0048 tree; raw transcript
(`apply-check-0049.txt` in the packet):

```
## reverse --check
rc=0
## reverse
rc=0
## forward --check
rc=0
## forward
rc=0
```

and compiles clean with the production target (raw transcript
`build-0049.log`, rc captured):

```
[0/2] Re-checking globbed directories...
[1/7] ... test_kernels_db_gen.py ... OK
[2/7] Building CXX object .../moe/moe_otd_runtime.cpp.o
[3/7] Building CXX object .../moe/expert_weight_providers.cpp.o
[4/7] Building CXX object .../moe/moe_3gemm_swiglu_opt.cpp.o
[5/7] Linking CXX static library .../libopenvino_intel_gpu_graph.a
[6/7] Linking CXX shared module .../libopenvino_intel_gpu_plugin.so
ninja_rc=0
```

### 6. What this leg does NOT discharge

`docs/window-053.md` dependency 3's **consumer-integration** clause stays
standing. The transport and import now exist and are proven at the mechanism
level, but the integrated **served** number — the fill running inside the
serving loop, the depth-4 gate rows, cold TTFT both arms, byte-identity across
two cold boots, decode non-regression — is **OWED**. No gate row was
half-measured. The cost named for leaving it owed: a served run needs the
depth-4 artifact's own layer keys matched to the store (the store holds the
depth-48 ratio-86 splitmix64 set, layer-major); the transport's ordinal
arithmetic is the store's, and an artifact whose layer keys differ needs the
store re-pointed.

**Evidence classes, this leg.** The mechanism, geometry, lifetime and slot
descriptor change: `code` (patch `0049`; `moe_3gemm_swiglu_mlp.cl:516`;
`arcwell.c`; `USING_ARCWELL.md`). The B60 proof, the sha256 identity and the
five red legs: `measured-here` (B60). The compile and apply checks:
`measured-here`. The served gate: **OWED**.

---

## D4 integrated served leg — the pinned NVMe fill runs inside the serving loop; the integrated number EXISTS (2026-09-24)

[`code` + `measured-here`. One B60 leg (`8086:E211`; the A770 untouched), plus
a device-free store build/verify. NO new tracked code. The `docs/window-053.md`
dependency 3 consumer half is CLEARED; the three gate rows stay OPEN.]

This leg closes the last OWED item of the D2/D3 integration: the depth-4 store
whose layer keys resolve to the served depth-4 artifact, and the integrated
served run with the fill live.

### 1. The depth-4 store (`measured-here` + `code`)

The depth-48 store holds the splitmix64 ratio-86 set keyed to the depth-48
artifact; the LISBON scope is depth 4 (`docs/window-053.md`), whose artifact has
four different layer keys. A second store was built with the tracked writer
(`tools/q4e/expert_store.py`, unchanged — driven, not rewritten).

**The four layer keys** are the artifact's `weight_0` bin offsets — the layer's
first OTD weight-file offset (`code`: patch 0013/0018; `get_const_offset`),
read from the artifact IR as the `layerN/moe/experts_gate/weight_u4` constant
offsets:

```
{"0": 284632533, "1": 2033390357, "2": 3650420373, "3": 5369067397}
```

The method is validated against the depth-48 layer-key map a served window
already produced (the plugin's own census-seed refusal named `layer_key
284636629`, the `layer0/moe/experts_gate/gridix_u8` offset of that artifact —
`measured-here`). The depth-4 plugin's own histogram (§2) reproduces exactly
`284632533 / 2033390357 / 3650420373 / 5369067397`, so the keys are not assumed.

**Geometry** (`measured-here`, raw output on the persistent evidence packet):

```
file count            284
sizes                 2457600 (all)
filefrag extents      284 -> 1   (every file exactly ONE plain extent)
non-plain flags       0          (every 7th file)
last,eof              41/41      (every 7th file)
st_blocks             4800       (fully allocated)
alignment             600 pages r0; 4800 LBAs r0; 37 x 64 KiB r32768
manifest              slice 2457600, group 128, 284 experts, seed 0xF2A17C0DE5EED, capacity 71
```

`aw_fiemap` byte-verifies all 284 against the raw device through the
partition-start translation, and its `--mutate` red leg fails on content:

```
284 files, 284 extents total (1.00 per file), 284 verified against the raw device
RESULT=PASS -- FIEMAP + partition offset gives the correct absolute LBA
RED ... MISMATCH
RESULT=FAIL -- raw device at the computed LBA is not the file's bytes
```

The manifest's ordinal→(layer, expert) map and the file sha256s match (0
mismatches); ordinal = `layer * 71 + slot`, ascending layer-key order equals
decoder order, so it is the transport's `dense_index * capacity + slot`
(`code`: patch 0049).

**Byte-transparency, verified directly** (`measured-here`): every pinned
expert's `gate|up|down` slice in the store was compared against the artifact's
own `weight_u4` constant bytes at `offset + expert * 819,200` — 4 layers × 71
experts × 3 roles = **852 slices, 0 mismatches**. The store is not merely
key-matched; it carries the served artifact's weights byte-for-byte. This is a
file read, not a served forward, so the B60 compute caveat does not apply.

### 2. The integrated served run (`measured-here`, B60)

The transport + OpenCL import (patch 0049) are live; `MOE_OTD_PINNED_NVME_FILL=1`
with the store dir, the B60 render node and the partition start. Toolchain: the
0049-built plugin (sha256 `2d83e2a6…`), served binary `a6dac5b5…`, artifact
`qwen38-flash-next-d4s-ov` xml `823997733f0b4b07`, ratio 86, `--moe-cpu-tier`, KV
u8, chunk 512, n_ctx 8192, the 5-token France prompt, greedy.

The load barrier did not refuse; the fill landed. The served run (fresh
process, first run — the graph compiled):

```
props ready=1 after 97s
lgc  load: language model ready in 19.3 s (paged); device-resident 1.88 GiB
lgc  load: ngram table STAGED: 1 port(s) of 33600 rows x 90 B = 2.884 MiB ...
lgc  load: expert slots: plateau probe settled at 0.19 GiB ... (source: probe-static)
lgc  slot 0: prefill     5 tok in  1.43 s (  3.5 t/s) | graph 1.43 s, ...
lgc  slot 0: decode      8 tok in  1.79 s (  4.5 t/s) | graph 1.66 s, ...
[OTD_PERF] gpu_hits=736, gpu_misses=4231, gpu_hit_rate=14.8178%, ... evictions=0,
  acquisitions=4967, device_slot_buffers=0, host_slot_buffers=36, staging_bytes=409600,
  cpu_tier_pairs=82904, cpu_tier_experts=4231
```

**The integrated served number.** The fill overlaps the serving boot.
`T_boot` = 97 s (served boot to `/props → 200`); `T_prefill` = 1.43 s (the
5-token prefill forward). The arcwell-arm integrated cold TTFT is therefore
**98.4 s**, below the pinned `X = 139.5 s` — but this is a single-arm number,
not the gate: the host-fed arm was not run in the same window, so
`docs/window-053.md`'s row stays EMPTY/OPEN. A second run (warm model cache)
reads boot 21 s, prefill 0.12 s, decode 0.13 s, streamed TTFT 0.25 s, and
repeats the exact same stats delta — a consistency check, not a cold run.

**`AW_IOC_STATS` delta** (module-global, read before/after; the design note's
requirement):

```
run 1: bytes 31,948,800 -> 729,907,200   (+697,958,400 = the exact pinned-store size)
       reads 39 -> 891, segments 39 -> 910, batches 12 -> 16, batch_reads 39 -> 891
       via_host_bounce 0 -> 0   max_inflight 6 -> 220
run 2: bytes 729,907,200 -> 1,427,865,600 (+697,958,400); batches 16 -> 20
       via_host_bounce 0 -> 0   max_inflight 220 -> 220
```

`via_host_bounce = 0` and `max_inflight = 220 (> 1)` as required; the bytes
delta is **exactly** the 284-expert pinned payload, so the controller DMA moved
the store and nothing else. The module's own per-client accounting records the
12 registrations and their release:

```
arcwell: MAP_BUFFER handle=49 npages=14208 bytes=58195968 peer2peer=1 (live=1 peak=1)
  ... handle=60 ... (live=12 peak=12)
arcwell: UNMAP_BUFFER handle=49 (live=11 peak=12)
  ... handle=60 (live=0 peak=12)
```

(`bytes=58,195,968` is the 64 KiB round-up of the 58,163,200-byte per-tensor
pool, 71 × 819,200; twelve BOs = three tensors × four layers. These `live`/`peak`
figures are **per-client**, `code`: `arcwell.c:1081`; an `AW_IOC_STATS` read from
a separate client fd therefore reports `buffers_live=0 buffers_peak=0`, which is
expected and is not a contradiction.)

**BO-backed-slot evidence, stated with its limit.** The fill wrote the store
bytes into the 12 per-tensor VRAM BOs; `bind_pinned_nvme_pool` imported each
dma-buf with `engine.import_buffer()` and replaced that layer's
`gate_w`/`up_w`/`down_w` (the stride assertion would throw otherwise), so the
fused GEMV kernel reads the imported pool (`code`: patch 0049; the integrated
run does not itself read a BO back — the OpenCL readback identity is the
earlier `tools/arcwell_cl_slot_proof.c` leg). The plugin's `device_slot_buffers`
counter stays 0 because it is incremented at **compile-time** `allocate_memory`
(`code`: `moe_offload_constant.cpp:161`), not at the runtime import — it is not
a BO counter and nothing in this leg reads it as one. The direct byte evidence
is the 852-slice store↔artifact identity (§1) plus the exact-bytes DMA delta.

**Key-match verification, from the plugin itself** (`measured-here`):
`MOE_OTD_ROUTING_HIST` dumped `layer,weight_offset,expert,count` — 4 layers,
`key_collisions=0`, weight offsets exactly
`284632533 / 2033390357 / 3650420373 / 5369067397`. The store and the artifact
agree on the keys the transport uses. No key-mismatch refusal was needed; none
was exposed.

**Host-RAM picture.** The PLE staging term is already banked (26.82 GiB →
**2.884 MiB**, read from the run's own `ngram table STAGED` line). The expert
host pool ceiling is `0.66 GiB (GTT)` against `0.19 GiB` resident; the run's
peak `VmRSS` was 3.53 GiB (cold) / 2.98 GiB (warm). Physical-host
`MemAvailable` minimum over the served run's own window **45.26 GiB**;
**0 watchdog trips**. (The sampler's overall minimum, 23.7 GiB, fell at
08:14:46 during the preceding failed CPU-plugin attempts, before the served
run — not during it. The 4 GiB watchdog was armed before the leg.)

### 3. The environment regression, diagnosed and worked around (operator-local)

The first attempts crashed in the OpenVINO **CPU plugin's** `cpu_info()` parse
(`free(): invalid next size (fast)`, gdb backtrace through
`ov::get_proc_type_table()` ← `IStreamsExecutor::Config::update_executor_config()`),
before any graph compile — reproducibly, with the same binary+plugin that
served on 2026-09-23. Cause: the container's
`/sys/devices/system/cpu/online` is a **sparse** list (`0,2,5,...`) while
`possible` is `0-15`; `lin_system_conf.cpp`'s parse loop only advances on
ranges, so a comma-only list corrupts the proc-type table. The leg ran the
served process inside a private mount namespace presenting a contiguous
`online`/`possible` view, which restores the 2026-09-23 behaviour
(`measured-here`: with the contiguous view the depth-48 model compiles and
loads — `lgc load: language model ready in 27.7 s`). This is operator
infrastructure, not campaign code; no tracked file changes for it.

### 4. Card, lock, module, service state

- **Before:** no `arcint`; no competing service; `arcwell` loaded and carved
  (inherited), `/dev/arcwell` present; B60 active, `power/control=on`; A770
  suspended, untouched; the coordinator's wake lock held (not touched, not
  released).
- **One card leg at a time;** the B60 alone; the A770 never opened.
- **Sampler (SOP §1):** the physical-host sampler ran before the leg with the
  4 GiB watchdog; minimum `MemAvailable` 23.7 GiB, **0 trips**.
- **After:** no `arcint`; `arcwell` left **loaded and carved** as found (the
  `~/src/arcwell/KERNEL_FACTS.md` half-state hazard forbids unloading a carved
  GPU); B60 active; A770 untouched; the leg's BO registrations released at fd
  close; the wake lock still held and untouched.

### 5. What clears, and what remains

`docs/window-053.md` dependency 3's **consumer-integration** clause is
**CLEARED**: an integrated served run exists (the fill inside the serving loop,
`via_host_bounce = 0`, the exact-bytes delta, the artifact↔store key match).
The three gate rows stay **OPEN** — they are the next leg: both arms in one
window (cold TTFT, arcwell ≤ host-fed, ≤ `X`), the `os.wait4` RSS row, and the
two-cold-boot determinism row (with the A770 confirmation where the B60 is not
bit-readable).

**Evidence classes, this leg.** Layer keys and the method: `code` (patches
0013/0018; `moe.cpp`) + `measured-here` (the artifact IR offsets and the
plugin's own histogram). Store geometry/FIEMAP/manifest/byte-transparency:
`measured-here` (ext4 partition). The integrated run, stats delta, OTD_PERF and
timings: `measured-here` (B60). The CPU-plugin regression: `measured-here`
(gdb + the namespace fix). The three gate rows: **OPEN/OWED**.
