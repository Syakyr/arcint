# window-052 — 0.5.2 VENICE acceptance (the BERLIN→VENICE gate)

Drafted 2026-09-19 as the acceptance commit for VENICE-001, BEFORE any
hot-set/LRU code, per `ROADMAP-0.5.x.local.md`'s law: *the first commit is
the acceptance criteria, with the measured rows EMPTY; prediction commits
precede measurement.* Every row below is EMPTY until measured; no number in
the tables is a result.

## Feature (roadmap 0.5.2)

Card hot-set + host LRU for expert slots — FreeToken's smart half. Hot-set
selection from an expert-access census; eviction/refresh discipline.

## Entry criteria

- **BERLIN closed.** [CORRECTED 2026-09-20: `F_served` at depth 48 **is**
  measured — window-051 clause (d) carries 0.1361/0.1512 on the B60 and 0 on
  the A770, where the row reads READABLE — so this criterion is MET, not
  pending.] The roadmap's falsifiable late-layer question instantiated and run
  (it was never designed).
- **The census instrument exists** — a per-token routed-expert histogram;
  its storage and format are this campaign's first deliverable.
- **G pinned from BERLIN's OWN measured rates, not invented:** [CORRECTED
  2026-09-20: the cited **23.6 t/s** is the **HF-exported 35B control**
  (`campaigns/serving-shape-logits.md:30`), not a d48g decode rate — no d48g
  decode rate is on the record, and the 48-layer tier measures 0.5-0.8 t/s.
  Re-pin G from a measured d48 rate before stating it.] The native
  scalar-tier route is 0.6 t/s (`d48n`) [2026-09-21: that 0.6 t/s point lies
  inside the measured 0.5–0.8 t/s band — it is not a third baseline]; G is a
  multiple of a host-bound
  baseline and is
  stated here before measuring.
  [CORRECTED 2026-09-21: the host-bound baseline is now FIXED at the measured
  `d48n` host-tier rate, **0.5–0.8 t/s** (B60, ratio 99 + tier, KV u8, f16,
  chunk 512; `measured-here`). **G itself stays UNPINNED**, and the speed row
  stays EMPTY rather than PASS, because the native artifact runs every routed
  expert on the scalar host tier (patch 0043: the fused kernels refuse the
  native formats), so residency alone moves no compute. G is pinned in a dated
  prediction commit before the speed leg, once a measured resident-compute rate
  exists — named dependency `sub4bit-vram-kernel` step 3, the OpenCL decode.
  Campaign: `docs/campaigns/expert-hot-set-lru.md`. No measured row is filled
  here.]
  [HELD 2026-09-21, operator decision: **VENICE's speed leg waits for
  `sub4bit-vram-kernel` step 3, the OpenCL decode.** The census instrument
  and the policy land on the host-tier path first; the speed row stays EMPTY
  and G stays UNPINNED until the resident-compute patch exists. No speed
  window is spent on the host-compute tier.]

## The bar in force

`bar_0.5.1 = 100 × F_ref` = **3.0905e-03 nats below row 2051 /
2.6946e-02 at or above it** (`docs/window-051.md` clause (d), BERLIN-001
`5d4dd59`; `F_ref` = the capture's uint16 reconstruction error =
3.0905e-05 nats mean). PROVISIONAL on one caveat: the rows are
f16-served. The inherited 0.0599 is SUPERSEDED BY LINEAGE and decides
nothing here. Any KL reading must print `F_served` beside it; a bar below
`F_served` is UNREADABLE, not PASS.

## Acceptance rows (EMPTY until measured)

| quantity | predicted | measured |
|---|---|---|
| warm-up decode vs the host-bound baseline | ≥ G × (measured d48n 0.5–0.8 t/s), G stated before the run; G UNPINNED today (see Entry criteria) | EMPTY |
| stale-byte zero proof | digest(host-bound bytes of expert E) == digest(card-bound bytes of the same E), every E in the hot set | EMPTY |
| convergence | rounds-to-plateau printed with the census | **MEASURED 2026-09-22** (`measured-here`): `rounds_to_plateau = None`, `plateau = False` at **S = 6 and S = 10**, at both **512 decode tokens** (window 003) and **4,096 decode tokens** (window 004); the selected resident set changed at every power-of-two prefix including 2,048 → 4,096. **V3 FIRES (2026-09-22):** the census does not plateau within the rounds this corpus admits, so the policy is not converging at this corpus length. Raw series in `docs/campaigns/expert-hot-set-lru.md` (status, 2026-09-21 night) and the window-004 plateau JSONs. |
| seed implication at the ratio-99 budget | coverage of the corpus's routed accesses by the census-seeded static partition (input to the seed decision; not a V-clause) | **MEASURED 2026-09-22** (`measured-here`): window 004 corpus census (3,278,880 accesses, 6,831 × 48 × 10): S = 6 **11.37 %**, S = 10 **16.22 %**, S = 16 **22.19 %**, S = 32 **33.84 %**, S = 64 **49.20 %**; analytic chance S/512 = 1.17 / 1.95 / 3.13 / 6.25 / 12.50 %. The incumbent `splitmix64` seed sits at chance at every budget (0.92–1.10× `slots/512`), so the census seed reaches 9.70 / 8.31 / 7.10 / 5.41 / 3.94× chance here. Coverage is a hit fraction over a non-plateauing window, not a converged steady state. **Measured correction (2026-09-22):** the plugin's actual pool at ``--offload-ratio 99`` is **5 slots/layer**, not 6 — `prepare_moe_otd_params` uses integer division `512*(100-99)/100 = 5`, while the engine's own ledger (`src/exec/fit.h`) prices `ceil(...) = 6`. The S = 6 seed is therefore REFUSED at load (measured: `census seed: layer_key … lists 6 experts but the pool has 5 slots (mismatched budget)`), and the served seed uses the corpus top-5 (coverage **9.96 %**, computed by grouping the corpus census CSV by layer, sorting `(-count, expert)` and summing the top 5: raw output `S=5: 9.9595%`, `S=6: 11.3721%`, in the session's `quality-report.txt`). The S = 6…64 numbers above are the engine-priced offline analysis; the served pool is one slot smaller. |
| quality under policy | no greedy digest change vs the pre-policy served answer | **MEASURED 2026-09-22** (`measured-here`): **PASS, no V4.** On the A770 (GPU.1, PCI 8086:56a0), native d48n artifact, `--offload-ratio 99 --moe-cpu-tier`, KV u8, chunk 512, one served window per arm, same 256-token prompt and greedy 32 tokens (temperature 0): incumbent `splitmix64` seed → greedy text sha256 `2169836b33e8bc74d7965fff867b13c1d3637388a4b52f11f639f381ce7cc36f`; corpus census seed (S = 5, the plugin's measured pool) → the **same** `2169836b…336f`. Byte-identical greedy output, so the census-seeded static partition is not visible under DESIGN §3.4. Plugin for the window: VENICE tree + patches 0003–0045 + 0046 (`d72c00bf…`); raw evidence in the session's `quality-report.txt` on the persistent census path. |
| verdict | REPORT ONLY until the tag | EMPTY |

## Falsifiable clauses (each can fail)

- **V1** — if warm-up decode < G × baseline, the hot-set did not engage (or
  the baseline moved): the row says so, not PASS.
- **V2** — if any expert's host-bound and card-bound bytes differ by digest,
  eviction/refresh is stale: RED, not PASS.
- **V3** — if the census does not plateau within the rounds the prediction
  states, the policy is not converging: the row names the rounds it took.
- **V4** — if any served answer's greeddy digest changes under the policy,
  the policy is visible: RED per DESIGN §3.4 (history-independent greedy
  output).

## Out of scope

- No FreeToken comparison here — that is GENEVA (0.5.8) and is pinned only
  by our own runs.
- No NVMe path — that is LISBON (0.5.3), blocked on the ext4 expert store.
- No segmented-chain compile; that route is `sub4bit-vram-kernel` +
  window-051 §2.

## Where it lives

`docs/window-052.md`. Candidate dependencies named in the campaign index:
`docs/campaigns/static-partition-prefill.md`,
`docs/campaigns/partition-seeding.md`,
`docs/campaigns/sub4bit-vram-kernel.md`.

## Status

- 2026-09-19: drafted as VENICE-001's acceptance commit; every measured row
  EMPTY. **NOT committed** — `CLAUDE.md` requires Fable review before every
  commit, and the tag path is the operator's.
- 2026-09-21: registered the campaign `docs/campaigns/expert-hot-set-lru.md`
  (row added to `docs/campaigns/README.md`) and the census-instrument design
  `docs/design-expert-hot-set-lru.md`. Corrected the host-bound baseline in
  place to the measured `d48n` rate (0.5–0.8 t/s) and marked **G UNPINNED**
  until a resident-compute path exists (patch 0043 runs every native-format
  expert on the host tier). This document was committed as `eaa7a06`
  (2026-09-19) — the 2026-09-19 entry's "NOT committed" is superseded by
  that commit, recorded here rather than rewritten. **No measured row filled;
  no policy code.**
- 2026-09-21: operator decision recorded — **VENICE's speed leg is HELD for
  `sub4bit-vram-kernel` step 3 (the OpenCL decode)**. The census instrument,
  hot-set selection, eviction/refresh discipline and the stale-byte digest
  proof proceed; the speed row stays EMPTY and G UNPINNED until the
  resident-compute patch lands. See `docs/campaigns/expert-hot-set-lru.md`.
- 2026-09-22 — **two rows filled from the now-complete measurements; the
  remaining three stay EMPTY, and why.** [documents the fill, `measured-here`
  for the values]

  **FILLED.**
  - *convergence.* `rounds_to_plateau=None`, `plateau=False` at S = 6 and
    S = 10, at both 512 decode tokens (window 003) and 4,096 decode tokens
    (window 004); the selected set changed at every prefix including
    2,048 → 4,096. This is a failing measurement, and it is recorded as one:
    **V3 FIRES (2026-09-22)**. Raw series and the two window-004 plateau
    JSONs are on the persistent census path; the numbers are transcribed in
    `docs/campaigns/expert-hot-set-lru.md` (status, 2026-09-21 night).
  - *seed implication at the ratio-99 budget.* On the window-004 CORPUS
    census (3,278,880 accesses, 6,831 × 48 × 10 exactly) the census top-S
    per layer covers S = 6 **11.37 %**, S = 10 **16.22 %**, S = 16
    **22.19 %**, S = 32 **33.84 %**, S = 64 **49.20 %** of the corpus's
    routed accesses — 9.70 / 8.31 / 7.10 / 5.41 / 3.94× the analytic
    S/512 chance. This is an input to the seed decision, not a V-clause; it
    is recorded because it is measured, and it is a hit fraction over a
    non-plateauing window, not a converged steady state.

  **STILL EMPTY.**
  - *warm-up decode vs the host-bound baseline.* Speed row, HELD for
    `sub4bit-vram-kernel` step 3 (the OpenCL decode); no speed measurement
    is taken on the host-compute tier. G stays UNPINNED.
  - *stale-byte zero proof.* Blocked on the engine-side host/card readback
    that does not exist; per the design it is not asserted from code.
  - *quality under policy.* Not yet measured at the moment of this entry;
    filled in the same session once the census-seeded served path has run
    twice against the incumbent seed (see the appended entry below when it
    lands).
  - *verdict.* REPORT ONLY until the tag, unchanged.
- 2026-09-22 (cont.) — **the quality row is filled: PASS, no V4.** The
  census-seeded static partition was wired (patch 0046) and served on the
  A770 twice, incumbent vs census seed, same prompt and greedy settings.
  Raw evidence, both digests and both OTD_PERF lines are in the session's
  `quality-report.txt` on the persistent census path; the summary:

  - **Card/config:** A770 (GPU.1, PCI 8086:56a0), native d48n artifact,
    `--offload-ratio 99 --moe-cpu-tier`, KV u8, chunk 512, n_ctx 8192; prompt
    256 token ids from the pinned capture, greedy `max_tokens` 32,
    temperature 0. One fresh process per arm.
  - **Incumbent** (`MOE_CPU_TIER_SEED` unset): `seed_source=splitmix64`,
    `slots=5`, `resident_checksum=0xa3dd88ad84a30530` (layer shown), prefill
    256 tok in 324.70 s, decode 32 tok in 60.84 s. Greedy sha256
    `2169836b33e8bc74d7965fff867b13c1d3637388a4b52f11f639f381ce7cc36f`.
  - **Census seed** (env `MOE_CPU_TIER_SEED` pointing at the corpus S = 5
    seed; the operator-local path is in the local notes): `seed_source=census`,
    `slots=5`, `census_seed_fp=0x3ae78e143cabbfb`,
    `resident_checksum=0x3dd7053a56e79968`, prefill 256 tok in 333.68 s,
    decode 32 tok in 60.20 s. Greedy sha256 **the same**
    `2169836b33e8bc74d7965fff867b13c1d3637388a4b52f11f639f381ce7cc36f`.
  - **Verdict:** byte-identical greedy output. The design's expectation holds:
    under the native artifact every routed expert runs on the host tier
    (patch 0043), so residency moves bytes, not arithmetic. **No V4.**
  - **Red-first refusal, measured on the card:** the S = 6 corpus seed (one
    slot over the pool) was run first and refused the load —
    `census seed: layer_key 284636629 lists 6 experts but the pool has 5 slots
    (mismatched budget)` — and the process never became ready. The device-free
    cells (`tools/test_census_seed.py`, 14) cover the malformed/mismatched
    cases; this is the same refusal on real hardware.
  - **Measured correction:** the plugin's pool at ratio 99 is 5 slots/layer
    (`prepare_moe_otd_params` integer division), while the engine's ledger
    prices 6 (`ceil`). Recorded in the seed-implication row above; it is why
    the served seed is the corpus top-5, not top-6.
  - The speed row stays EMPTY (HELD), G stays UNPINNED, and the stale-byte
    proof stays EMPTY (no engine-side host/card readback). No speed number is
    claimed.
