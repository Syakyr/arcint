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
  scalar-tier route is 0.6 t/s (`d48n`); G is a multiple of a host-bound
  baseline and is
  stated here before measuring.

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
| warm-up decode vs the host-bound baseline | ≥ G × (23.6 t/s), G stated before the run | EMPTY |
| stale-byte zero proof | digest(host-bound bytes of expert E) == digest(card-bound bytes of the same E), every E in the hot set | EMPTY |
| convergence | rounds-to-plateau printed with the census | EMPTY |
| quality under policy | no greedy digest change vs the pre-policy served answer | EMPTY |
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
