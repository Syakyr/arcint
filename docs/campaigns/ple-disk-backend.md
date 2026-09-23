# ple-disk-backend — stage the n-gram table per chunk from disk instead of pinning 26.82 GiB

## The defect, as measured

`bind_ngram_ports` (`src/exec/backend_ov.cpp`:8076–8165) binds the whole
Flash-Next n-gram table into **USM_HOST_BUFFER** tensors: one `ov::RemoteTensor`
per `ngram_table.K` port, `std::memcpy` of the entire payload, and the tensors
are retained in `ngram_table_tensors_` for the life of the process. The table is
28,800,138,240 B (26.82 GiB) over seven chunks of 47,718,400 rows x 90 B
(`docs/window-050.md`:1331).

Measured on the served path, both cards, hash ordinal 0 (`docs/window-050.md`
§4.8 run 2, R3, 2026-09-13):

    R3 | A770 / B60 | table bound in 33.03 s / 29.49 s, 26.82 GiB USM host,
         `usm_device` unchanged; peak host 27.67 GiB; no OOM | yes

Run 1's copy times were 3.6/3.6/3.5/4.3/3.7/3.5/2.6 s = 28.33 s, with the shard
pages evicting under the 28.78 GiB host peak. The user-stated figure for the
served loader is 17.2 s and 26.82 GiB; the window record's own reading is
29.49–33.03 s. Both agree on the resident bytes. **EVIDENCE CLASS:
`measured-here`.**

The pin is a CHOICE, not a constraint. The reference ships a disk backend as its
default:

- `~/src/FreeToken-ref/python/freetoken/engine/config.py`:32 —
  `ple_backend: str = "disk"`. The `"pinned"` backend (preload into page-locked
  host RAM) is the alternative, not the default. **EVIDENCE CLASS: `code`.**
- `~/src/FreeToken-ref/python/freetoken/models/qwen4_exp/ple_disk.py`
  (`DiskRowTable`) allocates **bounded** pinned staging up front —
  `alloc_pinned_tensor(max_graph_rows * token_bytes)` for decode
  (`max_graph_rows=256`) and `alloc_pinned_tensor(max_extend_tokens *
  token_bytes)` for prefill (`max_extend_tokens=8192`) — and `fill()`/`lookup()`
  stage only the rows a forward names, per fill. `source_from_safetensors`
  maps the checkpoint shards **in place, no copy** (`PleRowSource`). The
  io_uring switch (`FREETOKEN_PLE_IO_URING`, default `"1"`) changes the I/O
  backend, not the residency model. **EVIDENCE CLASS: `code`.**

For the served geometry (`num_ngram_heads = (ngram_size-1)*heads_per_ngram =
16`, 90 B/row), the reference's staging is `T x 16 x 90` B a forward — 737 KiB
at T=512, or 11.8 MB at its own `max_extend_tokens=8192` prefill bound — against
26.82 GiB pinned. The pin consumes ~26.82 of the 48 GiB host's ~44 usable GiB
(`docs/design-qwen-flash-next.md` WP6b), the same RAM `FIX E`'s host-resident
expert pool wants.

**Note on the figure `T x 7 x 90`.** The operator's working figure used H=7;
the served artifact's own id ports are `[1, T, 16]`
(`docs/window-050.md` §4.8; `src/exec/ngram_row_ids.h`:59
`(ngram_size - 1) * heads_per_ngram`). The formula is `T x H x row_bytes`; at
T=512 with the served H=16 it is **720 KiB**. The design note states both.

## Known against hypothesised

**Known (`code`, read from this tree):**

- The port contract already generalises to a chunked table. `ngram_ports.h`'s
  `PortPlan` reads the partition off the compiled model's port shapes; the ids
  are computed **host-side** and carried as ports (`ngram_chunk_ids` = which
  chunk, `ngram_local_ids` = the row within it). Nothing in the graph does
  integer arithmetic on an id (`ngram_ports.h`:11–17; `serving_shape.py`:
  `ngram_chunked_gather`).
- `PLETableBackend.lookup` is implemented (`src/exec/ngram_table.h`,
  `NGramLookup`), with the reference's row-id hash (`ngram_row_ids.h`).
- A **host-side** gather-with-dequant exists and is byte-exact tested against
  the scalar reference (`src/exec/ngram_gather.h`,
  `tests/test_ngram_gather.cpp`: 9/9 cases, AVX2 + scalar, 2026-09-10).
- Disk admission with a host-RAM fit refusal exists
  (`src/core/artifact.cpp::admit_ngram_table_from_disk`,
  `src/exec/fit.h::host_ram_fit_must_refuse`).
- The mmap path (`NGramLookup::mmap_table`) is genuinely lazy: `MADV_RANDOM`,
  rows paged in on demand. It is the `NGramLookup` path; `bind_ngram_ports` is
  the one that does the eager full copy.

**Hypothesised (unmeasured):**

- Per-forward `pread` of `T x H` rows (up to 720 KiB at T=512) is cheap enough
  on the decode/prefill path. The table's rows are random-access; the mmap
  path already reads exactly these rows. Whether plain synchronous `pread`
  (arcint scope) is fast enough without the reference's io_uring is unmeasured.
- Bounded staging does not perturb numerics: same bytes, same gather order,
  same IQ4_NL decode, no accumulation. To be proven by the gate below, not
  assumed.

## Gate

**Gate (copied from the operator's charge, 2026-09-22):** one served window,
same prompt and greedy settings, through the staged path vs the pinned path,
**byte-identical answer**, plus:

- the **freed host RAM measured** — the 26.82 GiB PLE term leaves the ledger;
- the load-time `memcpy` / 17.2 s gone.

**A numeric difference is a FINDING and blocks the change.** This touches
numerics and cannot be a silent optimisation. **EVIDENCE CLASS: `measured-here`
once run; the gate itself is a measurement that can fail.**

The mmap `NGramLookup` path is the byte-exact oracle for the device-free half;
`tests/test_ngram_gather.cpp` stays unchanged and unwidened.

## Entry criteria

1. The served IR declares a **staging** port: one `ngram_table.0` of
   `[staging_rows, row_bytes]` rows, `staging_rows >= T_max x H`, instead of
   seven full-chunk ports. (Same `ngram_table_ports(staging_rows, row_bytes)`
   helper; the caller passes the staging bound, not the table's row count.)
2. A real Flash-Next GGUF carrying `per_layer_token_embd.weight` (IQ4_NL) and
   the artifact's n-gram config, as `docs/window-050.md` §4.9 already admits.
3. A card window coordinated with the session holding the A770/GPU.1, per
   `docs/sop-card-window.md` (do not run two `arcint` legs at once; identify
   cards by PCI id).

## Scope — in / out

**In:** the host-side staging mechanism (row source, forward staging plan,
`pread` of named rows, capacity and out-of-range refusal); the fit arithmetic
term for staging bytes; the emitter port-count change; the served-window
acceptance; a DESIGN `§7.0.2` record and a CHANGELOG line on closure.

**Out (named so they are not silently implied):**

- **io_uring** (`FREETOKEN_PLE_IO_URING`). arcint scope is plain `pread`;
  the reference's io_uring is an I/O-backend optimization, not part of the
  residency model, and can be a follow-up campaign if the plain path is too
  slow.
- **Dedup / an LRU of staged rows.** The first cut stages in token order, no
  dedup (the reference's own `fill` order). A repeated row is read twice; that
  is correct and measurable, and dedup is a later lever.
- **Multiple PLE layers.** The served IR carries one PLE layer with ordinal 0;
  `bind_ngram_ports` already refuses a config with more (`REVIEW F3`).
- **The `NGramLookup` mmap path.** It is already lazy; this campaign does not
  change it, and it stays the oracle.

## Where it lives

| file | role |
|---|---|
| `src/exec/ngram_staging.h/.cpp` | the staging mechanism: row source, forward staging, `pread` |
| `src/exec/ngram_ports.h` | the port contract; a note that a staging port is a small chunk |
| `src/exec/fit.h` | `ngram_staging_bytes` and the pinned-vs-staged distinction |
| `src/exec/backend_ov.cpp` | `bind_ngram_ports` staging branch (card side) |
| `tools/q4e/serving_shape.py` | emit `ngram_table.0` at the staging bound |
| `tests/test_ngram_staging.cpp` | the device-free red-first cells |
| `tests/test_ngram_gather.cpp` | the byte-exact oracle — unchanged |

## Pipeline for this campaign

recon (this document + the cited code) → design note
`docs/design-ple-disk-backend.md` → red-first implementation (device-free) →
one coordinated card window → review before commit → DESIGN `§7.0.2` record and
CHANGELOG line on closure.

## Invariants

DESIGN §3.4 (history-independent greedy output) and §3.8; the §5 ladder;
`CLAUDE.md`'s measurement discipline. A campaign that would trade one for a
number does not close; it records the trade as a finding and stops.

## Status

- **2026-09-22** — campaign opened. Recon read: `CLAUDE.md`, `AGENTS.md`,
  `docs/campaigns/README.md`, `docs/research-freetoken-code.md`,
  `docs/design-qwen-flash-next.md` FIX D, `docs/design-routing-aware-expert-
  execution.md` §5.1, DESIGN §7.0.2 records on the PLE, the arcint mechanism
  (`ngram_ports.h`, `ngram_gather.h`, `ngram_table.{h,cpp}`,
  `admit_ngram_table_from_disk`, `bind_ngram_ports`), and the reference source
  (`ple_disk.py`, `config.py`:32). Design note written; contradicting claims
  revised in place, dated, with evidence classes. Device-free staging mechanism
  and its red-first cells in progress.
- **2026-09-22 (implementation)** — `src/exec/ngram_staging.h` (header-only:
  `StagingGeometry`, `check_staging_geometry`, `plan_staging_fill`,
  `pread_staging_rows`, `stage_from_file`) and `tests/test_ngram_staging.cpp`
  (6 cells) landed; `ngram_staging_bytes` added to `src/exec/fit.h`; the
  CMakeLists test list registers the new file.

  Green (this session, device-free build, host without AVX2):

      ./build/arcint-test ngram_staging   ->   6 cases run, 0 failed, 0 skipped
      ./build/arcint-test ngram           ->  44 cases run, 0 failed, 3 skipped
      ./build/arcint-test fit             ->  50 cases run, 0 failed, 0 skipped
      ./build/arcint-test                 -> 581 cases run, 0 failed, 5 skipped

  Red-first deletion proof (three mutations, each reverted; raw output
  captured in the session): breaking the slot order (`local[i] = 0`) fails
  `ngram_staging_places_row_i_at_the_row_the_id_names` and
  `ngram_staging_output_matches_the_pinned_gather_byte_exact`.

  **Red-first, MEASURED (2026-09-23, `measured-here`):** the file contributes
  **6 cells** to the `arcint-test` ladder, and every refusal is pinned by
  mutating the mechanism and watching that cell fail --
    * capacity refusal dropped (`if (global.size() > g.staging_rows)` ->
      `if (false)`), build rc = 0 -> FAIL
      `ngram_staging_refuses_a_forward_that_overruns_the_port` (run exit 1);
    * out-of-range refusal dropped, build rc = 0 -> FAIL
      `ngram_staging_refuses_an_out_of_range_table_id` (run exit 1);
    * the row placement shifted by one (`rows[i] = id` -> `id + 1`),
      build rc = 0 -> FAIL at `tests/test_ngram_staging.cpp:148` (run exit 134,
      the harness aborts on the first failure).
  Restored, the ladder reads **581 cases run, 0 failed, 5 skipped**, exit 0, and
  the restored header is byte-identical to the one committed. The raw mutation
  output is kept on the operator-local session evidence path (not in this
  PUBLIC repository). NOTE, stated rather than
  smoothed: the harness reports the WHOLE ladder, so "the file is 6/0/0" is not
  a run it can produce -- **6** is the file's own cell count and **581/0/5** is
  the ladder's result.

  The byte-exact cell diffs the staged gather against the pinned
gather (`gather_dequant` over the whole table) with `memcmp`, so it is the
campaign's numeric gate in device-free form.

  **Still open:** the emitter port-count change (`ngram_table_ports` at the
  staging bound), the `backend_ov.cpp::bind_ngram_ports` staging branch, and
  the one coordinated card window. The mechanism is implemented and proven
  device-free; the served acceptance needs a card and must not run beside
  another `arcint` leg.

- 2026-09-23 — **the wiring landed: the emitter can declare a staging window, and
  the runtime fills it per forward.** [code; `measured-here` for the cells]
  - **Emitter** (`tools/q4e/serving_shape.py`): `build_serving_shape_ir(...,
    ngram_staging_rows=N)` passes the STAGING BOUND to `ngram_table_ports`
    instead of the table's row count, so the IR declares ONE `ngram_table.0`
    port of `[N, 90]` — deliberately SMALLER than the source tensor, which is
    exactly how `bind_ngram_ports` recognises staging. The report carries
    `ngram_staging_rows` beside the unchanged `ngram_table_rows`. Two cells in
    `tests/python/test_serving_shape.py`: the staging shape, and the regression
    half (no bound -> the ports still cover the whole table). **RED-FIRST
    MEASURED:** against the unpatched emitter both cells FAIL (`2 failed`); with
    it they PASS (`2 passed`); the full suite is **36 passed, 1 skipped**.
  - **Runtime** (`src/exec/backend_ov.cpp`): `bind_ngram_ports` recognises a
    single port whose row count is below the source's, validates it with
    `check_staging_geometry`, opens the GGUF path for `pread`, allocates ONE
    `[S, row_bytes]` USM-host staging tensor, and SKIPS the full-table copy;
    `feed_ngram_ports` then stages exactly the rows this forward names
    (`ngram::stage_from_file`, slot `i` = the `i`-th named row) and feeds the
    slot ids with chunk id 0. **Compile-verified with the production flags:**
    `-fsyntax-only` against the OpenVINO toolchain returns **rc = 0 with no
    diagnostics** under `-Wall -Wextra -Wpedantic`.
  - **Still open, unchanged: the served card window.** Byte-identical answer
    staged vs pinned, the freed host RAM measured (the 26.82 GiB term off the
    ledger), and the load-time copy gone. It needs a card and must not run
    beside another `arcint` leg. The acceptance is a numeric gate: a difference
    BLOCKS the change. Until that window runs, the pin is still what the served
    path does — the staging path is implemented and proven device-free, not yet
    served.

- 2026-09-23 — **the export entry point exposes the staging bound.**
  [code] `tools/export_serving_artifact.py` gained `--ngram-staging-rows N`, passed
  into `build_serving_shape_ir` at the export call site, so an artifact can be
  written with the staging window declared. One avenue is CLOSED and worth
  recording: the existing full-depth artifact **cannot** be turned into a staging
  IR by editing its XML, because the artifact declares **three** chunked
  `ngram_table.K` ports (the whole table under the per-object cap) while staging
  needs **one** — the port COUNT changes, so the graph changes, so an export is
  required. The cheap acceptance path is therefore a TRUNCATED export
  (`--layers 4`), which is legitimate for this gate: the n-gram table is a
  property of the model, not of the depth, so a depth-4 staging artifact exercises
  the same mechanism and the same freed 26.82 GiB term.
