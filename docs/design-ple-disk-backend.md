# design — the n-gram table as a per-forward disk staging buffer

Campaign: `docs/campaigns/ple-disk-backend.md`. Governing defect: the served
path binds all 26.82 GiB of the n-gram table into USM host memory at load and
keeps it for the life of the process. This note names the replacement
mechanism precisely, and its geometry.

## The mechanism in one line

Host-side row ids → `pread` of just those rows into a **bounded USM host
staging buffer** → bind staging to the ports.

The graph is unchanged in its arithmetic: it still gathers rows from
`ngram_table.K` ports by host-computed ids. What changes is that a port is no
longer a fixed chunk of the whole table; it is a **staging window whose rows
are filled, in token order, from the rows this forward names**.

## Why the existing contract already carries this

`src/exec/ngram_ports.h` states the contract: the serving-shape IR carries the
table as `ngram_table.K` ports, and takes the hashed row for each token as two
ports — `ngram_chunk_ids` (which chunk) and `ngram_local_ids` (the row within
it). The decomposition happens **on the host**, because the GPU plugin was
measured gathering the wrong row for every id not exactly representable in f32
(`docs/window-050.md` §4.7). "Nothing in the graph does integer arithmetic on
an id."

A staging buffer is **a chunk whose row count is the staging bound**, not the
table's row count. The host computes the hashed global row id, and (in the
first cut) assigns it the staging slot equal to its position in the forward's
token order — exactly the reference's `DiskRowTable.fill`, which stages
per-request token runs in batch order and whose `lookup` copies that staging in
order. The local id the graph gathers is then a slot index in
`[0, T x H)`; no chunk decomposition is needed, so the served IR declares one
port and feeds `ngram_chunk_ids` as zeros (or drops the chunk port).

## Geometry

Let `H = num_ngram_heads = (ngram_size - 1) * heads_per_ngram`; for the served
Flash-Next config `H = 16` (`src/exec/ngram_row_ids.h`:59). Let `row_bytes = 90`
(IQ4_NL, five 32-element blocks of 18 B on a 160-wide row). Let `T` be a
forward's token count and `S` the port's static row count.

| term | formula | T = 512, H = 16 | note |
|---|---|---|---|
| rows named by one forward | `T x H` | 8,192 | every token, every head |
| staging bytes, one forward | `T x H x row_bytes` | 737,280 B = 720 KiB | operator's `T x 7 x 90` = 322 KB uses H=7; served H=16 |
| staging port bound `S` | `T_max x H` | e.g. 1,638 x 16 = 26,208 rows @ 90 B = 2.36 MB | `T_max` = the runtime's largest forward block |
| pinned path, resident | `table_rows x row_bytes` | 28,800,138,240 B = **26.82 GiB** | `docs/window-050.md`:1331, §4.8 R3 |

The reference's own bounds are the same shape: `max_graph_rows = 256` decode
lanes and `max_extend_tokens = 8192` prefill tokens, each `x heads x head_dim`
(`ple_disk.py`). At `max_extend_tokens = 8192` and `H = 16`, that is 11.8 MB.
`S` must cover `T_max x H`; a forward naming more rows than `S` is **refused**,
never truncated.

## The row source

The table on disk is the served GGUF's `per_layer_token_embd.weight` (IQ4_NL),
or the ARCINGRM file (`src/core/ngram_header.h`) whose payload is the same
bytes. A `PleRowSource` (the reference's name) records: the open file
descriptor, the payload base offset (`kHeaderBytes`, 24), `row_bytes`, and the
table's row count. A forward's fill is then:

1. compute the global hashed row ids host-side (`row_ids`, `ngram_row_ids.h`);
2. for each id, refuse if `id < 0` or `id >= table_rows`; assign slot `i`;
3. `pread(fd, staging + i * row_bytes, row_bytes, payload_base + id * row_bytes)`
   for each slot — the reference's batch read;
4. hand `ngram_local_ids` (the slot indices) and the staging tensor to the
   run request.

**No dedup in the first cut.** Slot `i` is the `i`-th named row, so the local
ids are the arange and the fill order carries the mapping. A row named twice is
read twice; correct, and a measurable cost. Dedup is a later lever, as is the
reference's io_uring (`FREETOKEN_PLE_IO_URING`).

## Refusals (named, before the gather)

- **Out-of-range table id.** A hashed id past the table's row count is refused
  by name, exactly as `NGramLookup::lookup` and `split_by_partition` already do.
  A Gather does not throw; it reads something, so silence here is a wrong row.
- **Staging overflow.** A forward naming more than `S` rows is refused by name.
  This is the bound the reference's static pinned allocation carries implicitly.
- **Source mismatch.** The staging port's `row_bytes` must equal the source's,
  the source type must be IQ4_NL, and the source row count must cover the
  hashed row space (`admit_ngram_table_from_disk`). A short table is refused at
  load, not discovered at decode.

## The fit arithmetic

`src/exec/fit.h` currently prices the pinned table: `ngram_table_bytes(...)`
over the full table, charged against host RAM. The staged path charges
`ngram_staging_bytes(staging_rows, row_bytes)` instead — kilobytes to a few MB,
not GiB. Admission still requires the table to exist on disk at full size
(`on_disk == kHeaderBytes + payload`), but it no longer requires host RAM for
it. **The pinned term is one configuration, not a requirement**: the reference
default (`config.py`:32) is `disk`.

## Numerics and the equivalence argument

The staged path reads the **same bytes** of the **same rows** in the **same
order** and decodes them with the **same** `gather_dequant`/`ngram_dequant_iq4nl`
path. No accumulation, no re-quantisation, no reordering of the decoded values.
The decode is exact integer arithmetic and a power-of-two codebook (measured
bit-for-bit against gguf-py, `docs/window-050.md` §4.8). Therefore staged
output must equal pinned output **byte for byte**; a difference is a finding,
not a tolerance to absorb.

## What changes, file by file

- **Emitter** (`tools/q4e/serving_shape.py`): `ngram_table_ports(staging_rows,
  row_bytes)` already yields a single `ngram_table.0` when `staging_rows *
  row_bytes` fits the cap; the caller passes the staging bound instead of
  `ngram_rows`. `ngram_chunked_gather` is unchanged (one port). The contract
  test's expected partition becomes the staging bound.
- **Runtime** (`src/exec/backend_ov.cpp::bind_ngram_ports`): recognise a single
  port whose row count is **below** the source's row count as a staging port;
  validate it against the source (type, `row_bytes`, source rows >= hashed
  space); allocate `[S, row_bytes]` USM host **once**; **no** full-table copy.
  Per forward (`feed_ngram_ports`), compute ids, `pread` the named rows into
  staging, feed the slot ids.
- **Device-free core** (`src/exec/ngram_staging.h/.cpp`): the row source, the
  forward staging plan, the `pread`, and the refusals — everything above that
  needs no card, with the red-first cells in `tests/test_ngram_staging.cpp`.
- **Oracle**: `tests/test_ngram_gather.cpp` and `NGramLookup`'s mmap path are
  untouched; the device-free equivalence cell diffs staged output against the
  pinned gather on the same synthetic table.

## Attached to a milestone — LISBON 0.5.3

This is the byte path LISBON is about: "ships arrive from disk" (git-ignored
`ROADMAP-0.5.x.local.md`, §0.5.3). LISBON streams expert bodies from NVMe into
residency; the same milestone's `nvme-direct-expert-tier` campaign asks whether
the host hop can go. The PLE pin is the other half of that footprint: a 26.82
GiB host-resident object that is not streamed at all. Staging it per chunk from
disk (a) removes the pin from the 48 GiB host budget, leaving that RAM to the
expert host pool, and (b) makes the table a **disk-resident, per-forward
read** — the same "arrive from disk" shape LISBON pins its acceptance on
(time-to-first-token from cold NVMe, RSS bounded through boot). LISBON-001's
RSS-bounded-through-boot cell is where the freed 26.82 GiB is read.

## Out of scope (named)

io_uring; dedup; multi-PLE-layer IRs; changes to `NGramLookup`'s mmap path;
changes to the bare byte path of `nvme-direct-expert-tier`.
