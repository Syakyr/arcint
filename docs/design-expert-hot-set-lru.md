# expert-hot-set-lru — the census instrument and the policy it feeds

Campaign: `docs/campaigns/expert-hot-set-lru.md` (0.5.2 VENICE).
Acceptance: `docs/window-052.md`.
Gate: warm-up decode ≥ G × the host-bound baseline (G pinned before the
speed leg); stale-byte zero proof with a red-first eviction mutation;
rounds-to-plateau printed with the census; DESIGN §3.4 output unchanged.
The speed leg is **HELD** for `sub4bit-vram-kernel` step 3 (the OpenCL
decode of the native formats) per the operator decision of 2026-09-21 —
the census instrument and the policy land on the host-tier path first.

---

## §1 — what the census is for, and why the existing instrument is not it

The policy has two consumers, and they need different facts:

1. **Hot-set selection** needs a *frequency rank per `(layer, expert)`* over
   a calibration corpus: which `S` experts per layer are worth pinning.
2. **The LRU replay and convergence** need *ordering*: which `(token, layer)`
   routed which experts, so the per-layer LRU's hit rate and
   rounds-to-plateau can be replayed offline (`tools/expert_lru_replay.py`).

Patch 0013 delivers (1) as an aggregate CSV and deliberately counts before
the hit/miss split, so it is the right *counting site* (`code`, patch 0013
header). It does not deliver (2) and its one short-corpus run left the
distribution too sparse to threshold on (`measured-here`, DESIGN §7.0.2ah).
This note names the storage and format for both, and the sources that
produce them.

## §2 — the canonical per-token trace (format v1)

One line per `(token, layer)`, byte-for-byte the format
`tools/expert_lru_replay.py::load_trace` already parses, so no consumer
changes:

```
# arcint routing trace v1
# artifact_sha256=<sha256 of the served .xml/.bin pair, or the GGUF shard set>
# capture_sha256=<sha256 of the prompt-token capture replayed>
# card=<PCI id, e.g. 8086:E211> device=<ov device> depth=<n layers>
# kv=<precision> f16=<on/off> chunk=<n> offload_ratio=<pct> tier=<static|lru|off>
# run=<run id> utc=<ISO-8601> tool=<producer>
0 0 3 17 88 210 411 455 477 501
0 1 5 9 44 121 200 300 388 490
1 0 2 3 8 17 88 155 210 477
...
```

**Grammar.** `token_idx`, `layer_idx`, then `top_k` expert ids, all
non-negative decimal integers, whitespace-separated. `#` and blank lines are
comments and are skipped by every reader. Token indices are positions in the
replayed window, `0`-based and contiguous; layer indices are `0`-based. Ids
are written **ascending** (deterministic order, independent of the router's
internal order); a repeat within one `(token, layer)` row is impossible
because a router selects distinct experts.

**Provenance.** The `#` header carries the artifact/capture digests and the
`card/device/depth/kv/f16/chunk/offload_ratio/tier` configuration exactly as
the measurement discipline requires ("name the card, the depth, the KV
precision and the configuration whenever a number moves", `CLAUDE.md`). The
trace's own sha256 is recorded wherever it is cited. A trace without a
provenance header is not a census.

**Storage.** Plain UTF-8 text; gzip is the transport (`.trace.gz`), never
the on-disk form a reader parses. One trace per calibration corpus per
configuration. Size estimate at Flash-Next geometry: 48 layers × `top_k` 10
× ~10 bytes ≈ 4.9 KB/token before gzip, so a 100k-token corpus is ~0.5 GB
raw, ~0.1 GB gzipped — acceptable on a persistent path, never `/tmp`.

## §3 — the aggregate histogram (patch 0013 retained, provenance added)

The existing plugin dump is **not** the census summary this campaign
consumes, and the two must not be conflated. They are separate schemas:

**The canonical census summary VENICE consumes** is derived from the trace
by `tools/hot_set_census.py`, in the trace's own layer-index space:

```
# <the same provenance header the trace carries>
layer,expert,count
0,3,1482
...
# total,<T>
```

`layer` is the 0-based decoder-layer index of format v1, so the summary
joins the trace directly; `expert` is ascending; the `# total,<T>` trailer
sums `count`. The counting site is the trace's router top-k.

**Patch 0013's plugin dump is a separate counter.** Its consumer contract
(its own round-two header, `code`,
`patches/0013-moe-otd-routing-histogram.patch:279-281`) is
`layer,weight_offset,expert,count` ordered by `weight_offset`, where `layer`
is the **0-based rank of the layer's weight-file offset among the layers the
process saw** (export order), not the decoder-layer index; `#` lines are
skipped and the trailer is `# total,<T>` (`:440`). It is emitted at process
exit and counts before the hit/miss split. It is retained as a plugin-side
cross-check: it can be joined to the trace on `weight_offset` once the
offset map is exported, but it is **not a pure function of the trace** (a
trace carries neither `weight_offset` nor the process's rank), so no
byte-for-byte reproduction is claimed.

## §4 — sources

**(a) Device-free reference-router trace — built first.** The exact f32
reference forward (`tools/ref_forward_stream.py`) runs the pin's own model,
including every layer's router. A `--router-trace PATH` flag taps each
layer's top-k ids and emits format v1 **on the host, with no card** — its
provenance header records `card=none device=cpu`, the device-free placeholder
the format allows (a card-derived trace records the PCI id). This is
the bootstrap source and the device-free oracle: it fixes the format, the
reader, the replay and the convergence metric before any GPU window.

Its limit is stated up front: the reference router is **not** the served
router. Quantisation and the served graph's arithmetic perturb near-tied
routers (the campaign record puts the top-10 mass at 6–27 % with margins
1e-4…1e-6, `measured-here`, `sub4bit-vram-kernel` status). The reference
trace is a **proxy** for format/consumer development and an independent
check, never the policy's calibration input.

**(b) Served-path trace — the authority.** A new opt-in dump channel
(`MOE_OTD_ROUTING_TRACE=<path>`) beside patch 0013's histogram, emitted at
process exit and flushed per chunk so a `SIGKILL` does not lose a window
(the 0013 dump learned that trap once already). The counting/emission site
is the router's own top-k, so the row carries the token index the served
graph routed. One card window produces it; the trace is the calibration
input the hot set is chosen from.

**(c) Corpus.** The acceptance prompt is too short (measured). The census
corpus is a long, fixed, published-in-digest-only document set replayed
through the served path — one window, recorded with its token count, so the
same corpus can be replayed after the policy lands.

## §5 — consumers

1. **Hot-set selection.** `tools/hot_set_census.py --trace T --slots-per-layer
   S` ranks each layer's experts by trace count, ties broken by ascending
   expert id, and emits the static-partition seed as `(layer, expert)`
   membership plus the rank in **trace layer-index space**. Patch 0018 keys
   its static partition by `layer_key` — the layer's first OTD weight-file
   offset (`code`, patch 0018 header), the same structural key patch 0013's
   CSV calls `weight_offset` — so the instrument also exports the
   decoder-index→`layer_key` map (from the served graph's own layer
   enumeration) and the seed is emitted in both spaces. `S` comes from the
   measured device-pool budget
   (`ARCINT_MOE_DEVICE_POOL_BYTES` / the fit's `expert_slot_bytes_static`),
   not from a guess.
2. **LRU replay.** The same trace feeds `tools/expert_lru_replay.py`
   unchanged: per-layer hit rate at `S`, cross-token reuse, and the
   **comparand** — the demand-warm LRU and patch 0018's random
   `splitmix64` seed — so the policy's gain is a delta against a measured
   baseline in one window, per `partition-seeding`'s gate shape.
3. **Rounds-to-plateau.** The replay is run over trace prefixes of length
   `1, 2, 4, ...` tokens; plateau is the first prefix length `r` at which
   the selected hot set per layer is **unchanged** for two consecutive
   prefixes and the per-layer hit rate moves by less than the stated
   epsilon. `r` is printed with the census. In the served run the same
   quantity is observed as the first forward after which the resident set
   stops changing.

## §6 — stale-byte zero proof

Two digests per hot-set expert `E`, taken in the same process:

- `host_digest(E)` — the bytes the host pool hands the miss tier
  (`ParallelWeightReader::mapped(offset, size)` over the same file
  `positional_read()` opens, patch 0011).
- `card_digest(E)` — the bytes read back from the device slot that E
  occupies.

Requirement: `host_digest(E) == card_digest(E)` for every `E` in the hot set.

**Red-first.** Before the equality check exists, a mutation cell perturbs one
byte on the eviction/refresh path (a wrong slot, a stale upload, an
off-by-one copy) and asserts the digest row goes **RED**. The mutation is
removed and the row must go green on the same input. A digest check that
cannot fail on a deliberately wrong byte measures nothing
(`CLAUDE.md`: a test must be able to fail).

**Evidence class.** `host_digest`/`card_digest` are engine-side readbacks;
the mutation is a test-only fault injection. When the readback path itself
does not exist yet, the proof is **not started** — it is not asserted from
code inspection.

## §7 — red-first cells (device-free first)

1. `test_hot_set_census.py` — trace parser accepts format v1, rejects a
   malformed row, and derives the canonical summary from a committed fixture
   trace deterministically (the summary is a pure function of the trace).
   A separate cell parses patch 0013's four-column plugin CSV (skips `#`,
   reads `# total,`) and joins it to the trace on `weight_offset` from an
   exported offset map.
2. `test_hot_set_selection.py` — frequency rank with the id tie-break is
   deterministic; the selected set has exactly `S` experts per layer; a
   synthetic trace with one clearly hot expert selects it over the random
   seed.
3. `test_rounds_to_plateau.py` — a fixture whose hot set is stable from
   prefix `k` returns `k`; one that never stabilises returns "no plateau in
   the predicted rounds" (the failing cell V3).
4. `test_stale_byte_digest.py` — the red-first mutation of §6.
5. served-path trace cell — one card window, provenance header verified,
   histogram/trace agreement checked.

## §8 — evidence classes

| claim | class | source |
|---|---|---|
| patch 0013 counts before the hit/miss split | `code` | patch 0013 header |
| patch 0013's CSV is `layer,weight_offset,expert,count`, ordered by `weight_offset` (`layer` a 0-based rank), trailer `# total,` | `code` | `patches/0013-…:279-281,440` |
| the short corpus left most experts at 0–2 routings | `measured-here` | DESIGN §7.0.2ah |
| `expert_lru_replay.py` reproduces WP6b within ~1.4 pts on the sha-pinned trace | `measured-here`/`code` | tool `--check`, docstring |
| the served d48n host-tier rate is 0.5–0.8 t/s | `measured-here` | `sub4bit-vram-kernel` status; DESIGN §7.0.2ca |
| every native-format expert runs on the host tier (no resident compute) | `code` | patch 0043's in-code assert `:726-731`; DESIGN §7.0.2ca |
| the reference router is not the served router (near-tied margins) | `measured-here` | `sub4bit-vram-kernel` status |
| hot set beats random seed; replay hit rate transfers to served t/s | `HYPOTHESIS` | untested |
