# expert-hot-set-lru — card hot-set + host LRU for expert slots, seeded from a per-token routed-expert census

Charter: 0.5.2 VENICE (`ROADMAP-0.5.x.local.md`). The "smart half" of the
FreeToken-shaped goal — a card-resident hot set of expert slots and a host
LRU for the rest, with the hot set chosen from a measured expert-access
census, not from a random seed.

## The defect, as measured

Not a defect: a **lever**, and on the record today it is a lever whose speed
half is **blocked by a named absence**, not by a missing policy. Three
measured facts frame it:

- **The host-bound baseline is measured.** Served `d48n` on the B60
  (`--offload-ratio 99 --moe-cpu-tier`, KV u8, f16, chunk 512, native
  artifact) decodes **0.5–0.8 t/s** — every routed expert decodes its rows
  on the scalar host path (`measured-here`, `sub4bit-vram-kernel` status
  2026-09-18 night; DESIGN §7.0.2ca). Served decode on this artifact is
  therefore **host-COMPUTE-bound**, not host-feed-bound.
- **The resident-compute path does not exist yet.** Under the native
  artifact every routed expert runs on the host tier because the fused
  kernels refuse the native formats (`code`, patch 0043's in-code assert
  that every routed expert is computed on the CPU tier until the OpenCL
  decode exists `:726-731`, DESIGN §7.0.2ca; design note
  `docs/design-routing-aware-expert-execution.md` §2.3b/§2.3c). Keeping an
  expert's bytes on the card changes **no** compute in that regime — the
  host kernel still decodes and GEMVs every routed expert from host mmap.
  The rate lever is `sub4bit-vram-kernel`'s step 3, the OpenCL decode of the
  native formats in the per-expert kernel.
- **The census, as it exists, cannot choose a hot set.** Patch 0013's
  `MOE_OTD_ROUTING_HIST` counts routing per `(call, expert)`, **before** the
  hit/miss split (`code`, patch 0013 header: the counting loop is the first
  thing `try_acquire_simultaneous` does, above the dedup map) and dumps an
  aggregate CSV whose contract is `layer,weight_offset,expert,count` ordered
  by `weight_offset`, `layer` a 0-based rank of the weight-file offset
  (`code`, `patches/0013-moe-otd-routing-histogram.patch:279-281,440`). It
  answers "which experts route", not
  "in what order", so it cannot feed the per-layer LRU replay. Run over the
  acceptance prompt alone it left most of 7,360 experts at **0–2 routings**
  — "no distribution to threshold 'rarely-routed' on" (`measured-here`,
  DESIGN §7.0.2ah; the same corpus gap is `partition-seeding`'s entry
  criterion (2)).
- **The offline replay exists and is sha-pinned.** `tools/expert_lru_replay.py`
  consumes one line per `(token, layer)`, `<token_idx> <layer_idx>
  <expert_id> ...`, models both a per-layer and a global LRU, and reproduces
  WP6b's per-layer hit-rate table within ~1.4 points on a trace pinned by
  sha256 (`measured-here`/`code`, tool docstring; `--check`). What it has
  never had is a **served-path trace** in that format.

So the campaign's own first absence is the gap between patch 0013's
aggregate counts and the trace `expert_lru_replay.py` needs. That is the
census instrument this campaign's charter names as its first deliverable.

## Known against hypothesised

**Known.**

- The device slot pool, its async upload ring and the per-layer LRU cache
  exist (patches 0005–0007, 0012; `measured-here`, DESIGN §7.0.2x/§7.0.2ai).
- Patch 0018 pins the host/device split for the process life as a static
  per-`(layer, expert)` partition, ranked by `splitmix64(seed, layer_key,
  expert)` with **no routing-frequency term**, exactly so greedy output is
  history-independent (`code`, patch 0018 header; DESIGN §7.0.2ae "F2",
  §3.4).
- The routing-aware design note deliberately does **not** seed from a
  histogram: "the LRU cache here does not seed from a histogram; it warms by
  demand on the first forward" (`code`, `docs/design-routing-aware-expert-
  execution.md` §7). The demand-warm LRU is the *comparand*, not this
  campaign's policy.
- The short-corpus sparsity above is measured, not assumed.

**Hypothesised** (each falsifiable, none measured):

- A frequency-ranked hot set (top-`S` per layer by census count,
  deterministic tie-break by ascending expert id) beats the random
  `splitmix64` seed at a fixed slot budget, on served decode and on
  rounds-to-plateau.
- The per-layer LRU replay's hit rate at a given resident budget transfers
  to served decode **once the resident-compute path exists**; until then it
  predicts bytes moved, not t/s.
- Seeding the static partition from the census lowers the warm-up cost that
  `static-partition-cold-start` owns, without touching §3.4 (the seed is a
  pure function of the recorded census, not of run history).

**Named dependency, not a hypothesis.** The speed half of this campaign
cannot be measured on the native artifact until either (a)
`sub4bit-vram-kernel`'s OpenCL decode lands, or (b) a resident-compute arm
of the fused path is produced. Stated here so the gate's emptiness is
attributable, the way window-051 clause (d)'s emptiness was.

## Gate

Copied from `ROADMAP-0.5.x.local.md` 0.5.2 and `docs/window-052.md`
(committed `eaa7a06`), which is the acceptance document:

- **Speed:** warm-up decode ≥ **G × the host-bound baseline** (the gate
  copied from `ROADMAP-0.5.x.local.md` 0.5.2 and `docs/window-052.md`; the
  G basis below is an **amendment** to that copy, dated 2026-09-21). The
  baseline is fixed at the measured `d48n` host-tier rate, **0.5–0.8 t/s**
  (B60, ratio 99 + tier, KV u8, f16, chunk 512). **G is pinned in a dated
  prediction commit before the speed leg runs**, from the census-measured
  per-layer hit rate and the measured resident-compute rate; until the
  resident-compute path exists G is **UNPINNED** and the row reads EMPTY,
  not PASS. No stale figure (the 23.6 t/s HF-exported 35B control) is
  inherited.
- **Speed hold (operator decision, 2026-09-21):** the speed row waits for
  `sub4bit-vram-kernel` step 3, the OpenCL decode. Census + policy land
  first; the speed measurement is not attempted on the host-compute tier.
- **Stale-byte zero:** `digest(host-bound bytes of expert E) ==
  digest(card-bound bytes of the same E)` for every E in the hot set, with a
  **red-first mutation on eviction** (perturb one byte in the eviction path
  and watch the digest row go RED).
- **Convergence:** rounds-to-plateau printed with the census.
- **Quality:** no greedy digest change against the pre-policy served answer
  (DESIGN §3.4).

Failable clauses (from window-052, V1–V4): V1 speed shortfall, V2 stale
byte, V3 no plateau in the predicted rounds, V4 visible policy change.

## Entry criteria

1. **The census instrument exists** — a per-token routed-expert trace and an
   aggregate histogram, with **storage and format named** and a **device-free
   source** (this campaign's first deliverable; design note
   `docs/design-expert-hot-set-lru.md`).
2. **A census over a corpus long enough to threshold "hot"** — the
   partition-seeding gap; the acceptance prompt alone is not enough.
3. **G pinned before the speed measurement**, per above.
4. **A resident-compute measurement path named** — the current native
   artifact has none. Per the operator decision of 2026-09-21, VENICE's
   **speed leg is HELD** until `sub4bit-vram-kernel` step 3 (the OpenCL
   decode of the native formats in the per-expert kernel) lands: the census
   and policy paths proceed now, and **no speed measurement is taken before
   that patch**. Until then the speed row reads EMPTY, not PASS.

## Scope — in / out

**In:** the census trace/histogram instrument and its storage format; a
hot-set selection function (frequency rank + deterministic tie-break); the
eviction/refresh discipline on top of patches 0005–0019 (seed the static
partition from the census; keep the demand-warm LRU as the comparand); the
stale-byte digest proof; the convergence measurement; one card window at the
end.

**Out:** the OpenCL decode / per-expert kernel (`sub4bit-vram-kernel`);
NVMe (`nvme-direct-expert-tier`, LISBON 0.5.3); the grouped prefill split
(`static-partition-prefill`); the prefix-cache seam (DESIGN §3.4 forbids
`--moe-cpu-tier` with a prefix cache); any FreeToken comparison (GENEVA
0.5.8, and pinned only by our own runs).

## Where it lives

- Acceptance: `docs/window-052.md`.
- Instrument design: `docs/design-expert-hot-set-lru.md`.
- Engine: `src/exec/fit.h` (`expert_slot_bytes`, `expert_slot_bytes_static`),
  `src/exec/flash_next_offload.h` (per-layer slots, projection),
  `src/exec/backend_ov.cpp` (property wiring, plateau probe).
- Plugin patches: 0005–0007 (slot pool + upload), 0012 (decode split), 0013
  (routing histogram), 0018 (static partition / LRU partition), 0043 (native
  formats through the tier).
- Tools: `tools/expert_lru_replay.py`, `tools/flash_next_fit.py`,
  `tools/ref_forward_stream.py` (the device-free reference forward), and the
  new census instrument `tools/hot_set_census.py`.
- Related campaigns: `static-partition-prefill`, `partition-seeding`,
  `static-partition-cold-start`, `sub4bit-vram-kernel`.

## Pipeline for this campaign

recon (this document + the instrument design + the cited headers) → the
census instrument, **red-first and device-free first** (reference-router
trace), then the served-path trace in one card window → census run over a
long corpus; hit-rate and rounds-to-plateau → **G pin** (prediction commit)
→ hot-set selection + eviction discipline, red-first → stale-byte digest
proof, red-first mutation → one card window at the end → review before every
commit → a DESIGN `§7.0.2x` record and a CHANGELOG line when it closes.

## Invariants

DESIGN §3.4 (history-independent greedy output) and §3.8, the §5 ladder, and
`CLAUDE.md`'s measurement discipline are non-negotiable. A policy that would
trade §3.4 for a hit rate does not close; the trade is recorded as a finding
and the campaign stops. Every disposition carries an evidence class
(`paper` / `code` / `measured-here`).

## Status

- 2026-09-21: campaign opened and registered in `docs/campaigns/README.md`.
  The acceptance document is `docs/window-052.md` (committed `eaa7a06`,
  2026-09-19), whose measured rows stay EMPTY. The host-bound baseline was
  corrected in place to the measured `d48n` rate (0.5–0.8 t/s) on the same
  date. The census-instrument design note landed as
  `docs/design-expert-hot-set-lru.md`. **No measured row filled; no policy
  code yet.**
- 2026-09-21: **operator decision — VENICE's speed leg is HELD for the
  patch.** `sub4bit-vram-kernel` step 3 (the OpenCL decode of the native
  formats) is the dependency; the census instrument, hot-set selection,
  eviction/refresh discipline and the stale-byte digest proof proceed on the
  host-tier path in the meantime. The speed row stays EMPTY, and G stays
  UNPINNED, until the resident-compute path exists. Recorded here so no
  window is spent measuring a tier that cannot carry the policy.
- 2026-09-21: **the device-free census instrument landed.**
  `tools/hot_set_census.py` (format-v1 parser, canonical summary, frequency
  rank, budgeted selection, rounds-to-plateau, patch-0013 four-column parser
  and join, seed emission) and `tools/ref_forward_stream.py --router-trace`
  (the device-free reference-router source, sharing the writer). Cells:
  `tools/test_hot_set_census.py`, **31 green, no card**. A census over the
  committed 400-token fixture **does not plateau** (S = 10 per layer:
  coverage 0.636 at prefix 2 falling to 0.356 at prefix 400, and the selected
  set changed at every power-of-two prefix), which confirms entry criterion
  (2) as a measurement — the corpus is too short to threshold "hot" — not an
  assumption. **Not started:** the
  served-path trace (one B60 card window) and the stale-byte digest proof
  (it needs the engine-side host/card readback; per the design it is not
  asserted from code). Outputs on persistent paths under the operator's
  census directory.
- 2026-09-21: **`ref_forward_stream.py --router-trace` verified, and its
  long-corpus limit measured.** `measured-here`: smoke at 1 layer, T = 5,
  `--device cpu`, a format-v1 trace of 5 token-major rows with top-10 ids
  ascending (`card=none device=cpu`), parsed back by
  `tools/hot_set_census.py shape`; trace sha256 `4659806b…49c0`, log sha256
  `3c2404ab…fe4c`. The tool **dequantises every expert tensor** per layer
  (`[forward] T=5 in 119.3s (expert loads 117.8s over 1 layers)`), so the
  cost is per-LAYER and T-independent and repeats each window; the pin's own
  expert loop is sparse (`code`, `tools/q4e/ref_moe.py:18`). The 48-layer
  long-corpus cost is a **projection** (hours per window), not measured, so
  the **reference trace is a short-corpus oracle only** and the long-corpus
  census cannot come from the proxy. **Decision on 0044:** author it at
  `OffloadExpertWeightProvider::try_acquire_simultaneous` as a per-call trace
  (`<call_seq> <layer_key> <top_k> <ids...>`), converted offline to format v1
  by splitting each call's flattened ids into per-token `top_k` chunks and
  mapping `layer_key` to the decoder index by ascending weight-offset
  (export) order (assuming export order equals decoder order, as patch 0018
  already assumes, and that the served op's flattened ids are token-major;
  for decode T=1 this is exact); build to a third prefix on the build host
  against the
  pinned tree, then take **one short B60 window** over the long corpus with
  the emitter on. The measurement plugin stays untouched, and no card has
  been taken yet.
- 2026-09-21: **patch 0044 authored, built, and the offline converter
  landed.** The emitter is `MOE_OTD_ROUTING_TRACE`, a per-call trace at
  `OffloadExpertWeightProvider::try_acquire_simultaneous`, added to the
  canonical series as
  `contrib/packaging/marfrit-openvino/patches/0044-moe-otd-routing-trace.patch`
  (mirrored in `patches/`) and applied on top of the 41 patches against pin
  `71640275`; built via `ninja openvino_intel_gpu_plugin`; the third-prefix
  install reports plugin version
  `2026.4.0-22849-71640275d29-marfrit-p19` (stamp deliberately left at p19,
  which already names patches 0003–0043; the trace build is identified by its
  `routing_trace` symbol). The default-measurement plugin
  and the debug-caps install are untouched. `tools/hot_set_census.py` gained
  `parse_call_trace`, `split_topk_chunks`, `layer_key_index_map`,
  `call_trace_to_v1` and the `from-call-trace` subcommand; cells are 43
  (superseded below by the 2026-09-21 batched-prefill/provenance entry: 52)
  green, including the two silent-if-wrong assumptions (a two-token call
  splits into the right `top_k` chunks / a mis-sized call is refused; the
  `layer_key` -> decoder index map is a bijection over the artifact's own
  layer set or the conversion refuses).
- 2026-09-21: **census-card decision, stated BEFORE the leg: the A770.**
  [decision, not a measurement] A census taken on the B60 would inherit the
  B60's run-to-run varying router input (DESIGN §7.0.2cb: the GDN state
  output is nondeterministic on Xe2), so its counts could move between
  forwards; the A770's served depth-48 path is bit-identical ×8 (the 74bc082
  arm). If a B60 census is ever required, it will be run twice and the
  run-to-run SPREAD of the counts printed beside the census, as the floor
  rule requires for any other reading. A census whose stability is unstated
  is not an instrument for a policy.
- 2026-09-21: **the converter's batched-prefill gap is closed, and the
  artifact/card part of the provenance requirement is enforced.** [code,
  measured-here for the cells] A real served trace opens with a batched prefill
  call, which `call_trace_to_v1` refused, so the captured trace could not
  convert at all. `from-call-trace` now takes **`--skip-batched`**: the batched
  call is skipped and **reported** (call and token counts in the v1 header, and
  on stderr when writing to stdout), because a batch dropped silently is a
  census that under-counts. Two rules hold either way: a trace in which EVERY
  call was batched is refused rather than converted to an empty census, and an
  empty call list stays an empty conversion (not a census). The same change
  makes part of the harness-injected provenance header **mandatory**:
  `from-call-trace` requires `--provenance` and refuses a file without a
  NON-EMPTY `artifact_sha256=` (or `artifact=`) and `card=`, so a census with
  no artifact or card attribution cannot be written. That is exactly what it
  says — artifact and card, nothing more: `capture_sha256`, `kv`, `depth`,
  `chunk`, `offload_ratio`, `tier`, `run` and `utc` from §2 are still the
  harness's responsibility and are **not** machine-checked, so §4b's owed
  item is narrowed, not closed. `tools/test_hot_set_census.py` is now **52
  cells** (was 43): `skip_batched` skips and counts while keeping the decode
  rows, an all-batched trace is refused with the skip ON, the CLI refuses a
  missing, incomplete or empty-valued provenance file, both §2 artifact
  spellings are accepted, and the stdout path still emits a parseable v1
  trace. Note the shape of a skipped opening prefill **as modelled here**: the
  prefill is one batched call, so the decode rows resume mid-sequence and the
  first reconstructed token can be partial — a real depth-48 served prefill
  emits one batched call per layer, all of them skipped, which leaves the
  decode stream at layer 0. Either way the token labels are a reconstruction,
  now stated in the header (`# token_labels=reconstructed`); aggregate counts
  do not depend on them, the LRU/plateau do. The served-path trace cell (§7.3)
  still needs the A770 window and is run against the VENICE plugin prefix
  (patch 0044 built there; the operator-local prefix path is recorded in
  `CLAUDE.local.md`); the speed row stays EMPTY and G UNPINNED.
- 2026-09-21: **corpus-split decision, stated BEFORE the served window.**
  [decision, not a measurement] Two attributions have to be separated before a
  number is read, and both are stated here rather than inferred from the
  result.

  **1. Batched prefill vs decode rows.** The census is the aggregate histogram
  of routed-expert accesses over the long fixed corpus. A served trace opens
  with BATCHED prefill calls (one call per layer, each carrying many tokens'
  ids, `top_k`-flattened token-major), and `from-call-trace --skip-batched`
  drops exactly those. A census derived from the converted v1 rows would
  therefore UNDER-COUNT the corpus it claims to measure. Decision: the census
  (the canonical `layer,expert,count` aggregate) is derived DIRECTLY from the
  patch-0044 call trace, counting every id of every call including the batched
  prefill calls — an aggregate needs no token label. The converted v1 decode
  rows are reserved for the ROW-LEVEL consumers only (LRU replay,
  rounds-to-plateau, token labels), which are explicitly a reconstruction.

  **2. Corpus calls vs load-time probe calls.** With `--offload-ratio > 0` the
  engine runs a distinct-token plateau probe (up to 8 forwards of
  `probe_floor_c` tokens) at load, before the served request (`code`,
  `src/exec/backend_ov.cpp` Phase B), and those forwards route experts through
  the same provider, so the emitter records them too. They are calibration of
  the slot pool, not the corpus. Decision: the census for the hot-set seed is
  taken over CORPUS calls only, selected by `call_seq >= <the trace's call
  count at the instant the corpus request is posted>`; the harness records
  that `call_seq_start` in its provenance file. The FULL-process census is
  still computed and printed, because patch 0013's CSV counts every call
  (probes included) and the `weight_offset` agreement check must join
  like-for-like; the corpus census and the full-process census are both
  reported with their difference (the probe's share) named.

  **Consequence for the instrument.** `tools/hot_set_census.py` gains
  `census_from_call_trace` (count every call's ids; the `layer,expert,count`
  summary in decoder-layer space), a `weight_offset` join on the RAW
  `layer_key` (patch 0013's key, no export-order assumption for the
  cross-check), and a `--from-call-seq` floor on both the census and
  `from-call-trace`, with red-first cells (the ladder is now **66 cells**,
  was 52; design §7.1 and the patch README carry the new count). The corpus
  itself is the pinned KLD capture's window 0 (2735 token ids, a fixed
  published-in-digest document set) posted as a TOKEN-ID prompt with a greedy
  continuation, one A770 window, the VENICE plugin, kv u8, offload 99 + tier.
  The harness must pass `--no-logits-slice`: a load-time probe forward trips
  the default logits slice and the executor refuses to come up (a harness
  fact recorded when the first attempt of this leg failed at load).
- 2026-09-21: **the served-path trace cell (§7.3) is RUN: one A770 window,
  the corpus census, the raw-key CSV agreement, and the stability statement.**
  [measured-here] Window `venice-census-003`, Arc A770 (`GPU.1`, PCI
  8086:56a0), served depth 48, KV u8, `--offload-ratio 99 --moe-cpu-tier`, the
  VENICE plugin, prefill chunk 512, the pinned KLD capture's window 0 as a
  TOKEN-ID prompt (2735 ids) with `ignore_eos` and a 512-token greedy
  continuation (one request = the prompt plus its continuation). Trace
  25,152 calls, 8,754,436 bytes, sha256
  `553480b169c60d653c9892b72609fb4b22c67ed4a69a4b31141c833ce5702e77`;
  patch 0013's CSV sha256
  `67c70f528fe88699355cb06a82b7849ceaaf0cb360dce4c614dd7d2f55d477f5`;
  harness provenance sha256
  `f0e294a01aa4d40d2b3289f3742e03ec9ba5177a75fc3557e022c07638cd8fd2`;
  `call_seq_start=288`.
  - The trace opens with 576 batched calls (288 load-time probe calls, 288
    corpus prefill calls across 6 chunks) followed by 24,576 decode calls
    (512 tokens x 48 layers). Converted with `--skip-batched` and
    `--from-call-seq 288`: 512 tokens x 48 layers, 24,576 rows.
  - **Corpus census** (derived directly from the call trace, calls >= 288,
    batched included): **1,558,560 routed accesses over 23,813
    `(layer, expert)` cells** = 3247 tokens x 48 layers x top-10 exactly.
    The full-process census is 2,172,960 accesses over 23,925 cells; the
    difference (614,400 accesses) is the load-time plateau probe's share,
    **not** the corpus.
  - **Trace vs patch 0013's four-column CSV, joined on the RAW `layer_key`
    (= `weight_offset`): 23,925 keys, mismatches = 0.** The full-process
    totals agree exactly (2,172,960). Under the corpus floor the join shows
    9,343 keys differing by exactly the probe's share: the CSV counts every
    call (probes included), so the like-for-like join is the full one, and
    the floored difference is the probe, named rather than a nuisance.
  - **Rounds-to-plateau**: NO plateau within the 512 decode tokens at S = 6
    or S = 10 (the selected set changed at every prefix; S = 6 coverage
    0.600 at 1 token -> 0.178 at 512, S = 10 1.000 -> 0.259). That is clause
    V3's failing shape, echoed from the device-free 400-token fixture: the
    aggregate census is no longer sparse, but the decode-row ranking does
    not stabilise in this window. `rounds_to_plateau=None`.
  - **Stability statement**: the A770 served depth-48 path is bit-identical
    x8 (`74bc082`); measured here, the 2735-token prefill census counts are
    **BYTE-IDENTICAL across the two independent A770 processes** of windows
    002 and 003 (23,609 cells, 1,312,800 accesses, `diff` of the data rows
    empty), and the probe census (614,400 accesses) is identical too. The
    counts are run-to-run stable on the A770; no spread is required here (a
    B60 census would carry the two-run spread statement, per the card
    decision above).
  - **Hot-set implication** [measured-here]: at the ratio-99 budget
    (ceil(512 x 1%) = 6 slots/layer) the census's top-6 per layer covers
    **9.36%** of the corpus's routed accesses; S = 10 covers 13.48%, S = 16
    18.63%, S = 32 29.23%, S = 64 44.12%. The seed is deterministic (count
    desc, ascending id; e.g. layer 0: 269, 309, 199, 117, 306, 11). The
    speed row stays EMPTY and G UNPINNED: this leg moves no acceptance row,
    and "frequency beats splitmix64" stays HYPOTHESIS.
  - **Harness facts recorded**: the VENICE prefix needs its own
    `runtime/lib/intel64` **plus** `runtime/3rdparty/tbb/lib` on
    `LD_LIBRARY_PATH` (the libdir alone leaves `libtbb.so.12` unresolved);
    the served load requires `--no-logits-slice` (the load-time probe
    forward trips the default logits slice and the executor refuses to come
    up -- window 001 failed at load for exactly this); and a token-id prompt
    over a document that ends at EOS emits 0 decode tokens, so the
    continuation needs `ignore_eos: true` (window 002 was prefill-only for
    exactly this). Window 002's prefill census remains the stability
    comparand.
