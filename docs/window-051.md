# window-051 — 0.5.1 BERLIN-AS-LISBON acceptance (the first 0.5.1 commit; every measured row EMPTY)

Recorded 2026-09-13, before any segmentation code exists in this repository
and before the 12-layer artifact exists. This file is the acceptance commit
of 0.5.1 in the form the roadmap's law demands: the criteria first, the
measured rows empty, the predictions dated, every clause falsifiable. The
commit that fills a row is a measurement commit and pastes the raw output.

The markers are `docs/window-050.md`'s (`RUN@<sha>`, `RUN@wt+<sha>`,
`RUN@unrecorded`, `DRY`, `UNTESTED`); a row with no marker is EMPTY, and
EMPTY is the honest state of every measured column in this commit.

## 0. What 0.5.1 is, in one sentence

All 48 layers of the Qwen3.8 Flash-Next serving-shape IR served on one card
by a **segmented forward** — K compiled sub-models, the hidden state carried
between them as ports, the expert bodies of one segment at a time streamed
from the GGUF mmap into a single host buffer set — so that the model's own
token for "The capital of France is" exists at full depth, on this host, with
its residency bounded by REAL host headroom. Paris ships inside it because the
answer IS the depth proof (operator CR #2, 2026-09-13).

Route (a) of the HANDOFF's change request, priced there; route (b), the
12-layer rung, is its first number and prices the depth ladder before the
segmentation lands.

## 1. The device facts this design is built WITH (measured; not re-litigated here)

| fact | value | where measured |
|---|---|---|
| A770 per-object cap | 4,294,959,104 B (4.00 GiB) | window-050 §4.4 |
| B60 per-object cap | its whole 24,385,683,456 B | window-050 §4.4 |
| prefill block cap under tiled MoE | 1,638 tokens (`[512, M, 2560]` tile object) | window-050 §4.7 |
| the plugin stages EVERY constant in USM-host memory at compile | measured on the A770: `drm-resident-gtt` to 11.8 GiB, `vram0` flat at 7.6 MiB, `CL_OUT_OF_HOST_MEMORY` at one expert body with the host at 2.6 GiB free | window-050 §4.10 F2 |
| cgroup / systemd memory fences do not charge driver / USM-host memory | the `MemoryMax=44G` fence never engaged while the physical host ran out | window-050 §4.10 F2; fleet mneme 372 |
| host RAM | 48 GiB, ~46 usable; the n-gram table takes 26.82 GiB of it as pinned USM host while bound | window-050 §4.8 R3 |
| the n-gram table | seven `ngram_table.K` USM-host PORTS (26.8 GiB, ~1.9 s bind), the IQ4_NL bytes decoded in-graph — ports, NOT constants | window-050 §4.8 |
| state constructs | slice with negative indices (the ShapeOf-swallow workaround, pinned by a cell) | window-050 §4.7 |
| GDN emission | perchunk (default) | window-050 §4.2 |
| QSA boundary | 2051; price 0.0 below, 2.385560e-02 over 29/2080 rows above | window-050 §8 |
| paged ports | 13, all from `SDPAToPagedAttention` (a pattern over emitted constructs; nothing to declare by hand); the pass asserts at least one `v13::ScaledDotProductAttention`, so a segment must hold at least one full-attention layer (index ≡ 3 mod 4) | window-050 §4.6, CHANGELOG 0.5.0 |
| served binary needs | `--ngram-gguf`; `position_ids` at port rank 1; `qwen_sparse_attention` counted as attention | CHANGELOG 0.5.0 |
| depth-4 served, both cards | warm decode 38 / 47 t/s (A770 / B60); first token 5613 `ramework` — MECHANISM, never an answer | window-050 §4.9 |
| full-depth constants | 1,030 dense f32 tensors 17,169,341,952 B (15.99 GiB); 144 u4 expert bodies 64,644,710,400 B (60.2 GiB), 448,921,600 B each; `.bin` 81,948,724,009 B; fill 82 min at 32.8 GiB peak host | window-050 §4.10 |
| the KLD reference | llama.cpp master `56b9eb28`, capture `af7993b7…`, n_ctx 2735, 684 rows below / 683 at-or-above 2051 per window, min clamped at max−16, uint16 reconstruction, bit-reproducible across runs | window-050 §4.11 |
| the served logits' own floor at depth 4 | KL(A‖B) 2.1e-4 nats mean (one window bit-identical, one with 41 of 1,367 argmaxes moved), a per-window event; mechanism: window-050 §4.11 | window-050 §4.11 |
| the bar | 0.0599 nats, **PROVISIONAL** = 1.5 × another model's R0 (window-050 §7); re-derived in row (d) below | window-050 §7 |

## 2. The route, as this commit fixes it (design, not measurement)

- **K sub-models** over the 48 layers, each a serving-shape IR of its own
  layer range with at least one full-attention layer inside it; the
  candidate cut is 4 × 12 (3 SDPA each); 6 × 8 and 12 × 4 are the fallbacks
  the arithmetic in row (a) prices, and the measured 12-layer staging (row
  (b), WP3) decides between them.
- **The hidden state** `[1, T, 10240]` (the hc-width residual stream) is an
  output port of segment k and an input port of segment k+1; segment 0 takes
  `inputs_embeds`, the last segment carries the final mixer and the head.
  Rope positions are recomputed per segment from `position_ids` (the tables
  are shared constants of 134 MB); the GDN / conv state tables and the KV
  pools are per segment, bound once per lane, exactly as the single model's
  are today.
- **Expert bodies as ports**: u8-packed USM-host ports (a u4 port is
  converted and copied to the device by the plugin; u8 shares). The MoE
  fusion matches a Constant, not a Parameter, so every expert computes for
  every token inside a segment: ~5 GFLOP / token / layer, "slow" is the
  accepted regime of 0.5.1 (Venice buys speed on top of a model that answers).
- **The runtime** drives the K compiled models per forward in order and
  refills ONE expert buffer set (the 15 GiB class at 12 layers) per segment
  from the GGUF mmap before that segment runs — so at most one segment's
  bodies are host-resident at any time, and the compile of a segment stages
  only that segment's dense constants.
- **Nothing is pinned by a cgroup**: the staging budget is REAL host
  headroom (`free`'s available minus the table minus the buffer set minus
  the process), measured before every compile, refused with the numbers.

## 3. Acceptance rows — EMPTY at this commit

### (a) Per-segment staging arithmetic against REAL host headroom

Per-layer terms from the measured full-depth artifact: dense f32 per layer
= (17,169,341,952 − 2,542,796,800 head) / 48 ≈ 304.7 MB (the PLE and final
mixer ride in the first and last segment; the head, 2.54 GB f32, in the last);
expert bodies per layer = 3 × 448,921,600 B = 1,346,764,800 B.

| cut | layers / segment | dense f32 staged per compile (predicted) | expert buffer set, host (predicted) | table, host | predicted host peak while compiling one segment (table + buffer + staging + ~3 GiB process) | measured peak | fits ~46 GiB? (predicted → measured) |
|---|---|---|---|---|---|---|---|
| 4 × 12 | 12 | 3.66 GB (+2.54 GB head, last segment) | 16.16 GB (15.05 GiB) | 26.82 GiB | ≈ 26.8 + 15.1 + 3.4 (+2.4) + 3 = **48.3–50.7 GiB** — predicted **NOT to fit** unless the table leaves pinned host or the buffer set is halved | EMPTY | predicted no → EMPTY |
| 6 × 8 | 8 | 2.44 GB (+2.54 GB head) | 10.77 GB (10.03 GiB) | 26.82 GiB | ≈ 26.8 + 10.0 + 2.3 (+2.4) + 3 = **42.1–44.5 GiB** — predicted to fit with 1.5–4 GiB to spare | EMPTY | predicted yes → EMPTY |
| 12 × 4 | 4 | 1.22 GB (+2.54 GB head) | 5.39 GB (5.02 GiB) | 26.82 GiB | ≈ 26.8 + 5.0 + 1.1 (+2.4) + 3 = **35.9–38.3 GiB** | EMPTY | predicted yes → EMPTY |

The n-gram table stays PORTS (26.8 GiB pinned, shared by every segment that
carries the PLE layer — only segment 0 does); it is never re-emitted as a
constant, or segment 0 dies the way the 48-layer compile died. The cut is
DECIDED by the measured 12-layer staging in (b): if 4 × 12 measures over
the headroom, 6 × 8 is the cut and this table's row says so with the
number.

### (b) Depth ladder prices

| depth | fill (predicted → measured) | compile, served binary (predicted → measured) | device-resident after compile | warm decode t/s | per-forward NVMe reads (predicted → measured) | card |
|---|---|---|---|---|---|---|
| 4 | measured 556.8 s (window-050 §4.9) | measured: server up in 1 min 58 s on a quiet host | 6.79 GiB | 38.2 / 47.1 | 0 (all resident) | A770 / B60 |
| 12 | **~25 min predicted** (¼ of 82 min, the per-layer share of the 48-layer fill) → EMPTY | predicted: minutes; staging ≈ 3.7 GiB dense + 16.2 GB expert bodies **as constants** at this rung (the single-model rung, not yet segmented) = ~19 GB staged in host next to the 26.8 GiB table → **predicted to compile on the B60** with ~2 GiB of host to spare, or to name the new edge → EMPTY | predicted ≈ 6.79 + 8 × 1.27 ≈ 17 GiB + activations at chunk 512 (against 22.71 GiB on the B60) → EMPTY | EMPTY | 0 (resident) | B60 |
| 48, segmented (4 × 12 or the cut (a) decides) | measured 82 min (the artifact exists) | per segment: EMPTY | per segment: dense f16 ≈ 1.8 GiB + one segment's expert ports (host) + KV + state | EMPTY | route (a) estimate **60 GB / forward at 1.8 GB/s ≈ 35 s** (cold page cache) → EMPTY; warm-cache case: EMPTY | A770 or B60 |

### (c) Cold-boot determinism

Two cold boots of the segmented 48-layer service (process restarted, page
cache dropped between them), the same greedy requests, byte-identical
answers in digest form:

| probe | boot 1 digest | boot 2 digest | identical? |
|---|---|---|---|
| "The capital of France is", 8 greedy tokens | EMPTY | EMPTY | EMPTY |
| KLD capture window 0, greedy token | EMPTY | EMPTY | EMPTY |

A digest that differs between boots is a finding, and its mechanism is
measured with `tools/boot_serving_shape.py --repeat` (window-050 §4.11's
instrument), not narrated.

### (d) The KLD bar, re-derived from THIS model's own reference round-trip

**The pair, DECIDED here** (BF16 of this model fits nowhere this repository
can run): **P_ref** = llama.cpp master `56b9eb28`, CPU, f32 accumulation,
over the SAME UD-Q3_K_XL shards — the pinned capture `af7993b7…` (n_ctx
2735, two windows); **P_cand** = arcint's segmented 48-layer served path
over the same shards (f16 kernels, the IQ4_NL table decoded in-graph, the
experts dense per token). The two forwards read the same quantised bytes, so
the ideal KL is 0 and what the instrument measures is arcint's
implementation divergence.

**The reference's own rounding error** (the doctrine's base): the capture
stores each row as an f32 scale, an f32 min log-prob and n_vocab uint16
steps (`log_softmax(int, const float*, uint16_t*, int)`, perplexity.cpp),
the min clamped at max−16 — so a row's reconstruction differs from the
writer's own f32 log-probs by at most half a step (≤ 8/65535 nats per logit)
plus the clamp's tail. **F_ref** is that error measured at 248,320-wide on
real rows: served f32 rows round-tripped through the writer's transcription
(`tests/python/test_kld_served.py::llama_row`) and compared with
`kl_ref_vs_served` (the 257-wide cell measured 2.9e-6 nats). llama.cpp's own
run-to-run floor is measured **0** (bit-reproducible, window-050 §4.11).

**The bar, stated before any number**: `bar_0.5.1 = 100 × F_ref` below row
2051, and `bar_0.5.1 + the measured QSA price` (2.385560e-02 over the rows
above the boundary, window-050 §8) at or above it — a stated multiple of the
reference's own rounding error, never an imported multiplier. The multiple
is 100 because two decades over the instrument's resolution is where "the
same distribution to the instrument's precision" stops being a defensible
sentence; it is a choice, and it is written down before F_ref exists. The
served path's own floor **F_served** (KL(A‖B) over replays, §4.11's
instrument at depth 48) is printed beside every reading; a bar below
F_served is unreadable and the row says UNREADABLE, not PASS. The inherited
0.0599 is printed beside it for continuity and decides nothing.

| quantity | predicted | measured |
|---|---|---|
| F_ref (uint16 reconstruction error at 248,320-wide, mean over the capture's rows) | between 2.9e-6 (the 257-wide cell) and 8/65535 = 1.2e-4 nats | EMPTY |
| F_served at depth 48 (KL(A‖B) mean, ≥ 2 replays, both windows) | of the order of depth 4's 2.1e-4, per-window event | EMPTY |
| bar_0.5.1 below 2051 = 100 × F_ref | 3e-4 .. 1.2e-2 nats | EMPTY |
| bar_0.5.1 at/above 2051 = bar + QSA price | + 2.385560e-02 | EMPTY |
| mean KL(P_ref‖P_served) below 2051 | not predicted (the first full-depth number) | EMPTY |
| mean KL(P_ref‖P_served) at/above 2051 | not predicted | EMPTY |
| verdict (REPORT ONLY at this commit; the gate is the tag's) | — | EMPTY |

### (e) THE PARIS LINE — EMPTY, with its falsifiable clause

| probe | served model | answer (raw, pasted) | token id | date |
|---|---|---|---|---|
| "The capital of France is" (ids `760,6511,314,9338,369`), greedy | the segmented 48-layer served path, both cards | EMPTY | EMPTY | EMPTY |
| "What is the capital of France? Answer in one word." (chat form), greedy | same | EMPTY | EMPTY | EMPTY |

**The clause**: the 48-layer served answer to "The capital of France is" is
the token whose surface form is `Paris` (leading space or not; its id is
recorded when measured, from the GGUF's own vocabulary). The clause dies if
the greedy token is anything else.

**The named-refusal fallback, written now**: if the token is not `Paris`,
the measurement commit does NOT explain; it localises with the instrument
that exists — `tools/boot_serving_shape.py --cut layerN/out --repeat` over
the segmented graphs — and names WHICH segment / layer first departs from the
llama.cpp reference's hidden state for the same prompt (the reference's
per-layer activations captured once with `llama-eval-callback` or the
equivalent at `56b9eb28`), or which knowledge refuses: a wrong token with
argmax agreement near 1 against the reference elsewhere is a head/mixer
defect; a wrong token with the per-layer divergence starting at the first
segment boundary is the boundary's; one starting inside a segment is that
segment's kernel. The refusal row names the layer and the measurement that
named it.

### (f) Explicit NOT claims

- A 12-layer rung answer (row (b)) is mechanism, never an answer: 36 layers
  are missing, whatever token it returns.
- A depth-4 token is mechanism, never an answer (window-050 §4.9).
- Nothing in this file compares to FreeToken; the comparison row is
  GENEVA's, by our own runs.
- "Fits" in row (a) is a measured host peak under a real compile, never a
  cgroup reading.
- No KL mean is quotable without F_served beside it.

## 4. Falsifiable clauses of this commit, listed

| # | clause | dies if |
|---|---|---|
| C1 | the 12-layer fill takes ~25 min | it takes under 15 or over 40 |
| C2 | the 12-layer single model compiles on the B60 through the served binary | the host refuses its staging (~19 GB next to the table) — then the new edge is the number WP4 designs with |
| C3 | 4 × 12 does NOT fit the host while compiling one segment | its measured peak sits under ~46 GiB |
| C4 | 6 × 8 fits | its measured peak sits over ~46 GiB — then 12 × 4, and the row says so |
| C5 | a cold-boot pair is byte-identical | any digest differs |
| C6 | F_ref lies in [2.9e-6, 1.2e-4] | it does not — the writer transcription is then re-checked against perplexity.cpp before anything else |
| C7 | Paris: the 48-layer greedy token is `Paris` | it is not — the named-refusal fallback runs |
| C8 | one full forward at 48 layers costs ~35 s of NVMe reads cold | measured under 15 s (page cache) or over 90 s (a mechanism other than bandwidth) |
