# design-nvme-direct-expert-tier — LISBON's byte path: the prefetch schedule,
# the routing warning horizon, and the fallback when a fetch has not landed

Campaign: `docs/campaigns/nvme-direct-expert-tier.md` (0.5.3 LISBON).
Companion recon: the campaign's 2026-09-23 recon and convergence entries.
External source: `~/src/arcwell` (`dev`, tree hash `e7d326e`).
Device-free: no card leg, no module load, no expert-store mutation. The B60
probe is named at §7, not run. [UPDATED 2026-09-23, later: the B60 probe of §7
**was** run — the plateau and async-budget clauses are measured; see §7 and §9.]

**Evidence classes.** Every disposition carries `paper`, `code` or
`measured-here`. All arcwell performance numbers are **arcwell's own** on
arcwell's hardware (`measured-here` *there*), never arcint's; every use of them
below is labelled that way and is context, per `CLAUDE.md`.

## The verdict in one line

A router-driven NVMe fetch cannot be hidden: the serving loop makes a layer's
top-k ids host-visible only at that layer's own expert op, so the warning
horizon is **zero layers ahead**. arcwell is therefore usable as a
**load-time bulk-residency fill** — where the pinned set's membership is a pure
function of configuration known at `bind()` — and **not** as a runtime miss
handler. LISBON keeps the host hop for runtime moves. Whether the load-time
fill pays cold-TTFT is the campaign gate; §3 gives the schedule and §7 the
probe.

---

## §1 — the store and the DMAbuf contract, as measured (carried, not re-measured)

The recon resolved both halves; this note consumes them and adds nothing.

| disposition | class | basis |
|---|---|---|
| the expert store is one ext4 partition, `fallocate`d, **one file per expert**, 1,700 files of 2,457,600 B, and **every file resolves to exactly ONE plain extent** | `measured-here` | campaign recon 2026-09-23 (`filefrag` 1700×`1`, flags `last,eof`); `aw_fiemap` byte-verifies all 1,700 against the raw device and its `--mutate` leg fails on **content** |
| file → absolute LBA is `fe_physical/512 + partition_start`, computed **once at open**, and never appears in the streaming loop | `code` | `aw_uapi.h` ("FIEMAP is NOT part of this surface … runs once at open"); `USING_ARCWELL.md` §5; `stub/tools/aw_fiemap.c` |
| reject non-plain extents (`UNWRITTEN`/`DELALLOC`/`INLINE`/`ENCODED`/`UNKNOWN`), pass `FIEMAP_FLAG_SYNC` | `code` | `USING_ARCWELL.md` §5, each rule backed by a red cell |
| one expert = **one request** at one `in_dest_offset`; it is **not** one DMA segment (a bio holds ≤ `BIO_MAX_VECS` = 256 pages = 1 MiB, so 2.34 MiB experts floor at ~3 bios) | `code` + `measured-here` | `aw_uapi.h`, `aw_expert_test` (3.09 bios/expert at 32 experts; floor 3.00) |
| the destination is a dma-buf from an xe VRAM BO with `NEEDS_VISIBLE_VRAM` + `CPU_CACHING_WC`, size a multiple of 64 KiB; export via `DRM_IOCTL_PRIME_HANDLE_TO_FD` | `code` | `USING_ARCWELL.md` §6; `KERNEL_FACTS.md` (`min_page_size = 64K`) |
| `MAP_BUFFER.in_length` registers the **BO**, not the file; the transfer (`READ_BLOCKS.in_block_count`) is the file, and both `in_dest_offset` and the transfer length are page-aligned | `code` | `aw_uapi.h`; `aw_expert_test.c` (`bo.size = (st.st_size + 0xffff) & ~0xffffULL`) |
| the 64 KiB question is **resolved**: 2,457,600 B = 600 pages and 4,800 logical blocks; the BO rounds to **38 × 64 KiB = 2,490,368 B**. No relayout is owed | `measured-here` | campaign recon, "64 KiB anomaly" |

**The client's once-at-open sequence** (`code`, `USING_ARCWELL.md` §3): create
the BO and export it; `AW_IOC_MAP_BUFFER` with `in_source = AW_BUF_DMABUF`,
`in_length = BO size`; then **assert `out_flags & AW_MAP_F_REQUIRE_P2P`** —
verbatim from §3, "this check is not optional". Read `AW_IOC_STATS` as a
baseline. Round every BO to 64 KiB regardless of the file size.

**The counters that prove no host bounce in the gate** (`code`,
`aw_uapi.h`):

- `via_host_bounce` — read as a **delta** before/after the leg's own work; the
  counter is module-global, not per-client, so an absolute reading includes
  other clients' refusals. Nonzero means a configuration that could only be
  served through host memory was refused (both refusal sites return an error;
  there is no path that bounces at all). Requirement: **delta = 0**.
- `max_inflight` — if it stays at 1 the NVMe queue depth was not used; the
  measured batch cell asserts `max_inflight == 16` for 16 registered 1 MiB
  buffers (`results/BATCH_2026-09-15.txt`).
- `batches` / `batch_reads` — prove the fetches went through a batch surface,
  not `AW_IOC_READ_BLOCKS`. The ASYNC leg reads `batches = 6` after six
  submit/collect cycles, so the `AW_IOC_SUBMIT_BATCH` path increments it too
  (`results/ASYNC_2026-09-15.txt`).
- `segments` — makes the gather visible (the bio split), so the request count
  can be sized against the layout and the segment count against the block
  layer.

---

## §2 — where the routing warning comes from, and exactly how much there is

**Answer: it comes from the router output at the MoE primitive's own execution,
and the horizon is zero layers. The serving loop gives the top-k ids for the
current layer and no earlier point exists.**

`code`, read from the plugin patches:

1. **The ids are produced per layer, by that layer's router.** In the plugin's
   MoE graph, `MoERouterFused` produces `TOPK_INDICES`, which is a dependency
   of the `MOECompressed`/`MOEGemm` primitive, so the router runs in the layer
   that consumes it. The plugin's own topology test shows one router fused op
   per MoE layer (`code`: `patches/0019-…:284-297`, a unit-test topology, not
   the production graph; patch 0012 does not build a router — `patches/0012-…:171-173`).
   Searched the series for a layer-ahead mechanism and there is none:
   `grep -rn 'lookahead\|look-ahead\|one layer ahead' contrib/packaging/
   marfrit-openvino/patches/*.patch` returns **no matches**, and every
   `prefetch` hit is an intra-kernel tile prefetch (patches
   0022/0023/0026/0029/0030/0032/0034), never a router lookahead.
2. **The ids are host-visible only inside `on_before_batched_gemv`, the MoE
   primitive's own hook, immediately before that same layer's GEMMs.**
   `scratch.topk_id->copy_to(stream, expert_ids.data(), 0, 0, topk_count *
   sizeof(uint32_t), …)` then `try_acquire_simultaneous(expert_ids, …)`
   (`code`: `patches/0012-…:932,941`; patch 0017 only hoists the read into one
   round trip, it does not move it earlier — `patches/0017-…:862-890`).
   `topk_count = token_num × top_k`; for Flash-Next `top_k = 10`
   (`code`: `src/exec/flash_next_offload.h:48`).
3. **The prefill paths are the same shape.** `on_before_prefill` /
   `build_grouped_mask_otd` acquire at the layer's own grouped-GEMM call
   (`code`: `patches/0037-…:230-300`); the batched prefill call exposes all
   `token_num × top_k` ids for that layer at once, still only at that layer.
4. The regression trace records exactly this granularity — one line per
   `try_acquire_simultaneous` call, `<call_seq> <layer_key> <top_k> <ids…>`,
   **before** the hit/miss split (`code`: `patches/0044-…:90`). The trace
   exists because that is the earliest the ids are known.

**Consequence, stated as the campaign's own condition.** The fetch needed to
hide a single expert is 1.125 ms (arcwell's number, not ours). For a runtime
miss the fetch can only be issued once the ids are host-visible, i.e. at the
layer that already needs the expert. There is no "one layer ahead" or "one
token ahead" to overlap against: the next layer's router depends on this
layer's output, and the next token's router depends on the whole forward. So
the depth the serving loop reaches is **0 layers**, and a win at a depth the
loop cannot reach is not a win. **The router cannot hide the fetch. LISBON
keeps the host hop for runtime misses.**

This is congruent with the campaign's Known section (`code`, `USING_ARCWELL.md`
§4.1, `results/LATENCY_2026-09-15.txt`): a synchronous fetch loses to the host
CPU's 198 µs compute by ~5.7× and to the overlapped host upload's ~45 µs
critical path by ~25×. It is also exactly FreeToken's shape: the reference's
runtime miss tier is the **CPU executor and the host bank**, not a per-forward
NVMe fetch (§6).

**The one thing the router is good for.** The static partition's membership is
**not** router-driven at runtime; it is
`static_partition_resident_experts(seed, layer_key, num_expert, capacity)`, a
pure function of configuration fixed at `bind()` (`code`: patch 0018;
patch 0046 validates a census seed at construction and applies it at `bind`).
Its warning horizon is **unbounded**: the whole pinned set for every layer is
known before the first token. That is the only fetch the note schedules (§3).

---

## §3 — the prefetch schedule over `AW_IOC_SUBMIT_BATCH` / `AW_IOC_BATCH_WAIT`

Because the router gives no runtime warning, the schedule is the **pinned-set
fill**: every expert the static partition pins, fetched ahead of the first
forward, from membership known at `bind()`. This is arcwell as bulk residency —
the campaign's Known scope — never a miss handler.

### What is issued, and when

1. **At `bind()`**: the pinned set is fixed (patch 0018/0046, `code`). Translate
   every pinned expert's file to `(absolute LBA, bytes)` once, at open
   (`aw_fiemap`'s shape, `code`).
2. **Before the first routed call**: submit fetches for the pinned experts. The
   `bind()` window is the warning — the set is configuration, not traffic.
3. **Per forward, no fetches.** Under the static partition a non-resident
   expert takes the host tier and *never enqueues an upload* (`code`:
   `patches/0018-…:716-780`); a resident expert is filled once and never
   re-decided. There is therefore **no arcwell call on the decode path**, which
   is what the campaign's *Out* requires.
4. **Mark filled on collect.** Every expert collected by `AW_IOC_BATCH_WAIT`
   must call the cache's `set_filled(slot)` (`code`: `patches/0018-…:875-881`),
   so the existing first-use fill branch (`fill_weights_memory`) never fires on
   the decode path; that branch is the host hop this schedule replaces.

**The precondition that decides whether the DMA is byte-transparent.** arcwell
copies file bytes verbatim into the BO; it cannot transpose. The plugin's
device slot layout and the weight file layout are **not the same for scales and
zero points**: the device wants scales `[group][oc]`, the file carries
`[oc][group]`, and the existing upload transposes via `maybe_transpose_scale_zp`
(`code`: `patches/0011-…:75-90`, `patches/0006-…:291`). The weights
(gate/up/down) are `[oc][ic]` in both, so a naive DMA of the whole 2,457,600 B
slice lands the weights correctly and the scales transposed. The resolution is a
**store-layout decision**, and it is **not yet made**: [DATED IN PLACE
2026-09-23 (B60 probe): the store the campaign measured is **not** a laid-out
arcint expert artifact — it is synthetic arcwell test data (one deterministic
4096-byte block repeated 600× per file), carrying no expert tensors and no
scales/zp, so neither order can be exercised against it. The decision
therefore remains open and is to be built with the artifact-format step.]
When arcint lays its own expert artifact out, either
lay each expert file out in device order (and adapt the host tier's scale
indexing — patch 0011's CPU kernel reads `s` in FILE order, `[oc][group]`), or
keep arcwell to the weight tensors and move the small scale/zp tensors through
the existing host path. This is a **precondition of the fill path**, checked in
§7; until it is resolved the byte path is not byte-transparent for the full
slice.

[RESOLVED 2026-09-23, artifact-format step — evidence
`code` + `measured-here`, no card leg. A REAL store now exists on the ext4
partition (3,408 files of 2,457,600 B, one plain extent each, whole-device
byte-verified) and the resolution is DECIDED: **the slice is the three packed
u4 weight tensors only, in device order**; the weights are `[oc][ic]` in both
file and device, so a naive full-slice DMA is byte-transparent, and scales/zp
are EXCLUDED from the DMA slice and stay on the host path that already
transposes. The arithmetic is why: weights 2,457,600 B = 600 pages / 4,800
LBAs exactly, while weights + serving-shape scales/zp = 2,553,600 B = 623.4375
pages (not page-aligned, and the plugin's per-tensor scale destinations are
unaligned too). The device-order-in-the-file alternative is implemented and
tested (`tools/q4e/expert_store.py --with-scale-zp`;
`transpose_scale_zp_to_device`) but is not the default because it cannot be one
page-aligned request. See the campaign's "Artifact-format step" section. What
this does NOT discharge: the integrated fill (D2/D3) and the gate.]

### How many are in flight, and the arithmetic

**arcwell's numbers, labelled as arcwell's.** One expert is 2,457,600 B
(`code`: `src/exec/flash_next_offload.h:45`). arcwell measured 2.18 GB/s
serial (64 single-expert fetches, 1.125 ms each) and **2.91 GB/s at
`max_inflight = 193`** (`results/LATENCY_2026-09-15.txt`, `measured-here`
*there*). Its batch cell submitted 64 experts as **~192 bios ≈ 3/expert** and
collected all (`results/ASYNC_2026-09-15.txt`).

- **Latency vs depth.** At 2.91 GB/s and a 1.125 ms fetch latency, the bytes in
  flight to saturate the rate are `2.91e9 × 1.125e-3 = 3.27 MB ≈ 1.33 experts`.
  So the 1.125 ms is hidden by **two** concurrent fetches in the arithmetic
  sense; what the measured 193 in-flight shows is that the queue-depth ceiling —
  not the latency — is what buys the last 33 % (2.18 → 2.91 GB/s).
- **Batching.** `AW_BATCH_MAX = 256` requests (`code`: `aw_uapi.h`). One pinned
  expert is one request (one extent), so a layer's pinned set at the high-80s
  ratios is one batch per layer: **71 requests at ratio 86**, **128 at ratio
  75** (`code`: plugin integer division `512*(100−r)/100`; the fit ledger's
  `ceil` is the over-reserving host figure, per the campaign's dated
  correction). [DATED IN PLACE 2026-09-24, patch 0049 leg: with the REAL store
  this count is not the request count. The store record is `gate|up|down`
  concatenated while the plugin's device layout is three per-tensor regions, so
  one expert is **three** page-aligned requests: 213 at ratio 86, 384 at ratio
  75. `AW_BATCH_MAX = 256` therefore does not fit ratio 75 in one batch; the
  production transport chunks by experts (`kPerBatch = 85`, 255 requests) and
  composites the arcwell batch ids. The ratio-86 gate scope (213 requests) is
  one batch.] Keep **4 batches in flight** — arcwell measured four batches
  overlapping and all collecting (`results/ASYNC_2026-09-15.txt`) — and collect
  the oldest with `AW_IOC_BATCH_WAIT` (one collector per id; a second concurrent
  wait returns `-EBUSY`, `code`: `aw_uapi.h`).
- **Submit is a cost to design around** (arcwell's measured constraint,
  `USING_ARCWELL.md` §7): ~9.7 ms per 64-expert batch idle, ~42 ms each with
  four outstanding. arcwell's own conclusion is "smaller, more frequent
  batches"; the schedule above is one batch per layer for that reason, not one
  deep batch for the whole pinned set.

### Does the fill pay? The projection, and its label

arcwell's 2.91 GB/s applied to arcint's byte count is a **projection** (`code`
arithmetic over arcwell's `measured-here` rate), not an arcint measurement:

| served ratio | slots/layer (plugin) | pinned bytes | at 2.91 GB/s | at arcwell's host-cold 1.58 GB/s |
|---|---|---|---|---|
| 99 | 5 | 0.59 GB | 0.20 s | 0.37 s |
| 86 | 71 | 8.38 GB | 2.88 s | 5.30 s |
| 75 | 128 | 15.10 GB | 5.19 s | 9.56 s |

The submit cost is inside the transfer window, not added to it: arcwell
measured submit returning at 9.7 ms against a 53 ms batch completion, and four
submits at 168 ms against 629 MB transferred (`results/ASYNC_2026-09-15.txt`).
The note states it rather than folding it into the cells, since each batch is
submitted while earlier batches are still transferring.

The **cold-TTFT gate** decides whether the delta (~2.4 s at ratio 86, ~4.4 s at
ratio 75) survives the model's own load and prefill; that is the B60 probe's
first measurement (§7). The host-fed cold number for **this** store is
unmeasured — arcwell's 1.58 GB/s is arcwell's, and the gate must measure both
arms in one window.

---

## §4 — the fallback when a fetch has not landed

**The design forbids a synchronous arcwell read on the decode path** (the
campaign's *Out*; `USING_ARCWELL.md` §4.1). Two cases must be kept apart, and
the design fixes each once, at load.

**Case 1 — an expert the partition marks non-resident: the host tier.** The
existing static-partition branch already routes a `probe()` miss to
`kCpuTierSentinelSlot` and computes that expert with the host kernel (`code`:
`patches/0018-…:716-780`); the prefill analogue returns `std::nullopt` from
`acquire_one` and `on_load_expert_weights` runs `moe_cpu_expert` (`code`:
`patches/0018-…:800-870`, `patches/0012`). **That host tier is the fallback**,
taken at `try_acquire_simultaneous` (decode) / `acquire_one` (prefill) — never
by an arcwell call. It is deterministic because the partition's membership is a
pure function of configuration, not of arrival order.

**Case 2 — a pinned expert whose fetch has not landed by the load barrier: a
load failure.** The design retries at the barrier and, if the pinned set is
still incomplete, **refuses the load**. It does **not** silently demote the
expert: host and device are different arithmetic — patch 0011: "the divergence
this introduces is small — but it is not zero"; the V4 leg measured native
per-expert dispatch not bit-identical to the host tier. A demoted expert would
make that boot's residency set differ from a clean boot and break the gate's
byte-identity across two cold boots (§3.4). This is a **new** rule, not an
existing branch: the current static-partition code fills a pinned-but-unfilled
slot on first demand via `fill_weights_memory` (`code`:
`patches/0018-…:731-776`, `patches/0012`), which is exactly the decode-path host
hop this note moves to the barrier (§3, `set_filled` on collect).

**Distinguish the surface failure.** If arcwell cannot be set up at all —
`AW_IOC_MAP_BUFFER` returns `-EOPNOTSUPP`, `out_flags` lacks
`AW_MAP_F_REQUIRE_P2P`, the BO cannot be created/registered, or the batch ioctl
refuses — that is also a load failure, for the whole configuration, not a
per-expert case.

**Consequence.** No synchronous `AW_IOC_READ_BLOCKS` and no first-use
`fill_weights_memory` can sit on the decode path: a genuinely non-resident
expert takes the host tier (Case 1), and a pinned expert is made resident or
the load is refused (Case 2).

**Red-first cell owed.** A device-free cell that asserts a synchronous
`AW_IOC_READ_BLOCKS` on the decode path is **refused by arcint's own code**
(the campaign's red-first requirement), so the losing configuration cannot be
reached by accident. That cell is not written in this leg.
[DATED IN PLACE 2026-09-23, D2/D3 integration leg: the cell is written. It is
`pinned_nvme_fill_refuses_a_synchronous_read_on_the_decode_path` in
`tests/test_pinned_nvme_fill.cpp`, backed by `require_no_sync_read()` in
`src/exec/pinned_nvme_fill.h` (the plugin twin in patch `0048`), and it is
mutation-tested: removing the guard makes it fail.]

---

## §5 — the consumer it must not oversell

1. **Pin once, from an offline census, per `(seed × regime)`.** The convergence
   leg measured the served static partition's resident set as a pure function
   of configuration, converging at position 0 with zero thrash; the **rolling
   census never converges** (`rounds_to_plateau = None` at every budget 5…128
   and every regime, 47–48 of 48 layers changing at the largest prefix). The
   design therefore **never re-derives the hot set from live traffic**. A
   census seed is calibrated on the regime that will be served and is never
   averaged across regimes: the same leg measured a regime-matched seed rising
   to a steady state while an out-of-regime seed drifts monotonically
   (`census-prefill` scored on decode: 58.76 → 39.27 → 35.22 at 128). Every
   figure is a `(seed × regime)` property.
2. **Criterion 4's literal hardware clauses were OWED at this note's date.**
   [UPDATED 2026-09-23, later: the `MOE_OTD_PERF_LOG` plateau probe's
   device-byte plateau at the high-80s ratios and the async-batch upload budget
   are now **MEASURED on the B60** — see §7 and §9. The *consumer* integration
   and the campaign gate stay OWED; nothing in those measurements discharges
   them.] Original text kept as written: The convergence
   clause is answered device-free; the `MOE_OTD_PERF_LOG` plateau probe's actual
   device-byte plateau and per-forward timing at the high-80s ratios, and the
   async-batch upload budget, need a card leg. So do criterion 4's *consumer*
   integration and the campaign gate. Nothing in this note may be read as
   discharging them.
3. **The speed half is separate.** The static partition's steady hit fraction at
   the ratio-99 budget is only ~15 % of decode accesses (the measured V1
   shortfall, `window-052`); *convergence* is not *adequacy*. This note does not
   claim the fill changes the decode t/s; the gate's decode non-regression row
   measures that.

---

## §6 — what is FreeToken's way and what is ours

Read from `~/src/FreeToken-ref` (`code`). The note must not blur the two.

**FreeToken-faithful already:**

- **The PLE disk backend.** `python/freetoken/models/qwen4_exp/ple_disk.py`
  (docstring: "Disk-backed PLE table … batch-reads rows from the checkpoint's
  fp8 shard tensors into pinned staging; the captured `lookup` is a fixed-shape
  H2D copy + dequant"), bounds `max_graph_rows = 256`,
  `max_extend_tokens = 8192`. That is our staging bound and one
  `[S, row_bytes]` tensor with slot ids — mechanism matched; only the source
  container differs (arcint reads a GGUF shard). This is the
  `ple-disk-backend` campaign's shape, cited there.
- **A custom O_DIRECT-friendly container is FreeToken's way.**
  `python/freetoken/checkpoint/ftw.py` (docstring: "FreeToken Weight (FTW)
  checkpoint: one O_DIRECT-friendly on-disk format for a whole model"),
  sharded `freetoken-00000.ftw`, `ALIGN = 4096`, per-layer bank entries. The
  ext4 one-file-per-expert store is the same idea; the container ambition is
  **not** an arcint invention.

**Not FreeToken's way, and the note says so:**

- **The per-forward NVMe DMA tier is arcint-original.** FreeToken's disk reads
  happen at **bank fill**, not as a decode-path miss handler: `moe/host_banks.py`
  ("pin-after-fill"; "chunked multi-threaded O_DIRECT — DMA straight from disk
  into the … bank, bypassing the page cache"); `moe/expert_banks.py` (`code`).
  The runtime miss tier is the **CPU executor** (`moe/cpu_executor.py`) and the
  host bank; GPU slots hold routed experts. So arcwell's 1.125 ms/expert
  per-forward fetch is an arcint bet, and the campaign already frames it as
  "pays as bulk residency and loses as a miss handler". **LISBON is not the
  FreeToken way.** The FreeToken way is exactly the host hop — which this note
  keeps as the runtime fallback (§4).

---

## §7 — the B60 probe spec (the next leg, unambiguous; RUN 2026-09-23 — items 1–3 measured, item 4 owed)

Operator decision 2026-09-23: the B60 is free for this campaign's gate, B60 legs
are allowed **with the determinism caveat recorded** (byte-identity claims only
where the B60 is known readable, otherwise paired with an A770 confirmation),
and the sequence is design note first, B60 probe second. The probe must measure,
and paste raw output for:

1. **The device-byte plateau and per-forward timing at the high-80s ratios.**
   Run the `MOE_OTD_PERF_LOG` plateau probe against a synthetic 512-expert
   configuration (RED-C-02's own buildable form,
   `docs/design-qwen-flash-next.md`:496) at `--offload-ratio` **86 (71 slots)**
   and at least one neighbouring integer ratio (e.g. **83**: 87 slots by the
   plugin's integer division, the corrected 10 GiB row's 88 by the `fit.h`
   ceil ledger). Assert and print: `device_resident_bytes` reaches a
   plateau and **does not oscillate**, zero evictions/thrash, and the
   per-forward time across forwards. This is a RED-C-02 hardware clause, owed
   regardless of the fill decision.
2. **The async-upload budget inside one inference step.** With arcwell loaded
   on the B60, submit the pinned-fill batches through `AW_IOC_SUBMIT_BATCH` and
   collect through `AW_IOC_BATCH_WAIT` while the serving stream runs. Print:
   submit return wall time (arcwell's own 9.7 ms/64-expert, 42 ms at 4
   outstanding are the comparand — **arcwell's numbers**), bytes/completed per
   batch, `max_inflight`, and the serving step's own wall time with the fill
   overlapping. Assert submit returns well before transfer completion and the
   step does not regress.
3. **`AW_IOC_STATS` as a delta, on the leg's own work.** Print the before/after
   delta; require `via_host_bounce == 0`, `max_inflight > 1`, `batches` and
   `batch_reads` nonzero, and `segments` reported.
4. **Not this leg** (the campaign gate, after the probe): cold TTFT both arms in
   one window, served-answer byte-identity across arms and across two cold
   boots, decode non-regression at the reference cell. The probe informs those;
   it does not replace them.

The gate's byte-identity clause carries the determinism caveat: where the B60
is not known bit-readable (the recorded per-card GDN floor,
`served-prefill-determinism`), the byte-identity row pairs with an A770
confirmation.

---

## §8 — evidence classes

| claim | class | source |
|---|---|---|
| routing ids become host-visible only at the MoE primitive's own hook | `code` | `patches/0012-…:932,941`; `patches/0017-…:862-890`; `patches/0037-…:230-300` |
| one trace line per `try_acquire_simultaneous`, `<call_seq> <layer_key> <top_k> <ids…>` | `code` | `patches/0044-…:90` |
| `top_k = 10`, 512 experts, 48 MoE layers, slice 2,457,600 B | `code` | `src/exec/flash_next_offload.h:45-48` |
| static partition membership is a pure function of `(seed, layer_key, num_expert, capacity)`, fixed at `bind()` | `code` | patch 0018; patch 0046 |
| device slot layout ≠ file layout for scales/zp; a byte-transparent DMA lands weights correctly and scales transposed | `code` | `patches/0011-…:75-90`; `patches/0006-…:291` |
| a non-resident expert never enqueues an upload under the static partition; resident fill is on first use | `code` | `patches/0018-…:716-780,800-870` |
| host tier and device tier are different arithmetic | `code` + `measured-here` | patch 0011; the V4 quantification leg (native ≠ host, affine = host) |
| arcwell 1.125 ms/expert; 2.18 GB/s serial; 2.91 GB/s at `max_inflight = 193` | `measured-here` (arcwell's own hardware) | `results/LATENCY_2026-09-15.txt` |
| arcwell host cold 1.58 GB/s (the LATENCY cell's own host-cold row reads 1.59) | `measured-here` (arcwell's own hardware) | `results/GATE_2026-09-15.txt`; `USING_ARCWELL.md` §1/§7 |
| submit ~9.7 ms/64-expert idle, ~42 ms at 4 outstanding; four batches overlap and collect | `measured-here` (arcwell's own hardware) | `results/ASYNC_2026-09-15.txt`; `USING_ARCWELL.md` §7 |
| one file per expert → one extent → one request; ~3 bios/expert floor | `code` + `measured-here` | `USING_ARCWELL.md` §5; `aw_expert_test` 3.09/expert |
| the pinned-set projection (0.59/8.38/15.10 GB; 0.20/2.88/5.19 s) | `code` arithmetic over arcwell's `measured-here` rate | this note; byte counts from `fit.h`/`flash_next_offload.h` |
| rolling census never converges; served partition converges at 0 with zero thrash | `measured-here` | campaign convergence leg 2026-09-23 |
| PLE disk backend and FTW container are FreeToken-faithful | `code` | `~/src/FreeToken-ref/.../ple_disk.py`; `.../checkpoint/ftw.py` |
| the runtime miss tier is the CPU executor + host bank (FreeToken's way = the host hop) | `code` | `~/src/FreeToken-ref/.../moe/host_banks.py`, `expert_banks.py`, `cpu_executor.py` |
| a router-driven fetch can be hidden | **REFUTED** | §2; horizon 0 |
| the pinned fill pays cold TTFT | **HYPOTHESIS** (projection only) | §3; §7 decides |

---

## §9 — decisions and status

**Decisions this note takes.**

- **D1. No runtime NVMe fetch.** The routing warning horizon is 0 layers; a
  router-driven fetch cannot hide arcwell's 1.125 ms (arcwell's number). The
  decode-path upload path is forbidden; non-resident experts keep the host
  tier, taken at the existing static-partition branch.
- **D2. arcwell is the load-time pinned fill only**, scheduled from `bind()`
  membership (unbounded warning), one batch per layer, 4 batches in flight,
  `AW_IOC_BATCH_WAIT` for collection. No fetch on the decode path.
- **D3. A pinned expert that has not landed by the load barrier is a load
  failure** — retried at the barrier, then refused. No silent demotion. The
  host tier remains the fallback only for experts the partition marks
  non-resident at `bind()`. This keeps §3.4: the served answer never depends on
  I/O timing.
- **D4. The verdict on the campaign's condition stands:** as a *miss tier*,
  LISBON keeps the host hop. The bulk-residency fill is a separate, gate-decided
  question (cold TTFT), and the RED-C-02 hardware clauses are still owed.

**Status.**

- 2026-09-23 — **design note leg, device-free.** Written against the recon
  (`a657b47`) and the convergence leg (`9b63991`), both cited, not re-measured.
  No card leg, no module load, no expert-store mutation. Next: review, then
  commit, then the B60 probe of §7. The campaign gate, its byte-identity rows
  and the DESIGN §7.0.2 record remain owed.
- 2026-09-23, later — **B60 probe (§7 items 1–3) run; recorded** in
  `docs/campaigns/nvme-direct-expert-tier.md`. Results: the device-byte plateau
  is **0.37 GiB at ratios 86 (71 slots) and 83 (87 slots)** with
  **`evictions = 0`** (under the static partition the pool is host-mapped —
  `device_slot_buffers = 0`, host-side ceiling 7.91 / 9.67 GiB — so the device
  term is the small working set, the two-ledger shape). The pinned-fill async
  budget is 2.71 GB/s (71 experts: submit 20.9 ms of a 64.5 ms batch) and
  2.94 GB/s (87 experts: 28.3 / 72.8 ms) at DEPTH=4; `AW_IOC_STATS` delta
  `via_host_bounce = 0`, `max_inflight` 213/261, `batches`/`batch_reads`/
  `segments` nonzero. The full-slice fill's byte-transparency is **OWED** — the
  ext4 store is synthetic arcwell test data (a repeated 4096-byte block), not a
  laid-out expert artifact, so the scales/zp transpose (`code`:
  `patches/0011`, `0006`) cannot be resolved from it. §7 item 4 (the gate),
  the serving step with the fill overlapping, and the D2/D3 integration remain
  owed. B60 determinism caveat recorded: timing is
  admissible, no byte-identity claim made; the cold-TTFT delta stays a
  projection over arcwell's own 2.91 GB/s, labelled arcwell's.
- 2026-09-23, later — **artifact-format step (§3's precondition RESOLVED); the
  full-slice fill's byte-transparency is now decided, not owed.** A REAL ext4
  expert store exists: 3,408 files of 2,457,600 B (`8,375,500,800 B`), every
  file exactly ONE plain extent (`filefrag` 3408→1), `aw_fiemap` byte-verifies
  all 3,408 against the raw device and its `--mutate` leg fails on content; a
  24-expert sample is code-exact and dequantises to the u4 half-step (worst
  err/bound 1.000001). The verdict: the slice is the three u4 weight tensors in
  device order (byte-transparent); scales/zp are excluded and stay on the
  transposing host path (the page-alignment arithmetic in §3). New writer
  `tools/q4e/expert_store.py` + red-first `tools/test_expert_store.py` (15
  cells; mutants: role order, non-fallocate, wrong transpose, short file,
  f32-sidecar pass-through).
  Dependency 1–2 of `docs/window-053.md` are cleared; §7 item 4 (the gate)
  and the D2/D3 consumer integration stay **OWED**. No card leg, no module
  load; the synthetic store was not touched.
- 2026-09-23, later — **D2/D3 integration leg: the schedule is wired and
  compile-verified plugin-side (patch `0048`); the byte destination stays
  OWED.** The load-time pinned fill's schedule now lives in the plugin's
  `OffloadExpertWeightProvider` — one batch per layer, four batches in flight
  over the arcwell batch surface, collect→`set_filled`, a load barrier that
  retries once then REFUSES (never a silent demotion), and the red-first guard
  refusing a synchronous `AW_IOC_READ_BLOCKS` on the decode path. The schedule
  is tracked device-free as `src/exec/pinned_nvme_fill.h` and tested by
  `tests/test_pinned_nvme_fill.cpp` (8 cells green; 3 mutants each fail their
  named cell); patch `0048` applies on the full 0003–0048 series and compiles
  clean. But the note's D2 presupposes a device destination: the static
  partition's slots are host-mapped (`device_slot_buffers=0`, B60 probe) and
  arcwell needs a dma-buf from an xe VRAM BO, so there is no BO for the fill to
  land in. The transport and the per-expert dma-buf BO-backed slot destination
  are OWED, and an enabled fill refuses the load. §7 item 4 (the gate) and the
  three gate rows stay OWED/OPEN. No card leg, no module load; the arcwell
  module was found loaded/carved and left as found.
- 2026-09-24 — **byte-destination proof leg: the destination is SETTLED AND
  PROVEN on the B60; D2's device destination is no longer an open question.**
  A non-arcint client (`tools/arcwell_bo_dma_proof.c`) created a caller-owned xe
  VRAM BO (raw `DRM_IOCTL_XE_GEM_CREATE`: VRAM placement +
  `NEEDS_VISIBLE_VRAM` + `CPU_CACHING_WC`, size rounded to 64 KiB), exported it
  via `DRM_IOCTL_PRIME_HANDLE_TO_FD`, registered it peer-to-peer with
  `AW_IOC_MAP_BUFFER` (`AW_MAP_F_REQUIRE_P2P` asserted), transferred **one real
  2,457,600 B expert** from the real store by controller DMA, and verified the
  BO through its own xe mapping: **byte-identical** (sha256 `4a4bb0f9…`),
  `AW_IOC_STATS` delta `via_host_bounce = 0`, `max_inflight > 1`. The mechanism
  is decided with `code` citations: arcwell provides **no** allocator/helper
  (`stub/src/arcwell.c:7-8`; `M4_API.md` "the contract lives in the uAPI, not in
  a client library"; `KERNEL_FACTS.md` "the working recipe", step 1), so the
  plugin creates the BO itself; OpenCL can only **import** the dma-buf
  (`cl_khr_external_memory_dma_buf`, arcwell's own E2E cell), not create one;
  `AW_BUF_XE_GEM` is unimplemented at `0.0.1`. Three dated corrections: on the
  B60 the 64 KiB BO gate is **not** kernel-enforced (a 37.5 × 64 KiB BO was
  accepted end-to-end, contrary to `~/src/arcwell/KERNEL_FACTS.md`'s A770/DG2
  measurement); a submission-time geometry error is reported in
  `out_submitted`/`out_err`, not the ioctl return, so `Transport::submit` must
  check the counts; and a system-memory dma-buf is refused with `-ERANGE` at the
  carve range check, before the `via_host_bounce` sites. What this does NOT
  discharge: the plugin-side `Transport`, the OpenCL import into the slot
  descriptors, and the LISBON gate's overlapping-step number. `docs/window-053.md`
  dependency 3's destination clause is CLEARED in place; its
  consumer-integration clause stays standing. No `arcint` leg; the arcwell
  module was found loaded and carved and left as found.

- 2026-09-24, later — **plugin transport + OpenCL slot import leg: the
  destination is wired and the mechanism is proven; the served gate stays
  OWED.** Patch `0049` supplies `lgc::nvme_fill::ArcwellTransport` (raw xe VRAM
  BO creation + dma-buf export + `AW_IOC_MAP_BUFFER` peer-to-peer +
  `AW_IOC_SUBMIT_BATCH`/`AW_IOC_BATCH_WAIT`, with the store ordinal `dense_layer
  * capacity + slot`) and `bind_pinned_nvme_pool()`, which imports the dma-bufs
  with `engine.import_buffer()` and **replaces the layer's host-mapped
  `gate_w`/`up_w`/`down_w`** with BO-backed memories, so the fused GEMV kernel
  reads the controller-DMA'd bytes directly; the six scale/zp tensors stay on
  the transposing host path and are uploaded at load
  (`fill_weights_memory(..., include_weights=false)`). Dated correction to §3:
  the store record is `gate|up|down` concatenated but the device layout is
  three per-tensor regions, so one expert is **three** page-aligned requests,
  not one; "one expert = one request" held only for the synthetic store.
  Build: patch reverse-applies/re-applies on the 0048 tree and compiles clean
  (rc 0). Mechanism on the B60: `tools/arcwell_cl_slot_proof.c` DMA'd two real
  store experts into per-tensor VRAM BOs, imported them into OpenCL and read
  them back **byte-identical** (sha256 `d463d1d5…`), `via_host_bounce` delta 0,
  `max_inflight` 6; five red legs fail as required. §7 item 4 (the gate), the
  served overlapping-step number, and the depth-4-artifact↔store key match stay
  **OWED**; the three `docs/window-053.md` rows stay OPEN.
