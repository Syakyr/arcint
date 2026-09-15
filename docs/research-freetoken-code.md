# Research: FreeToken — THE CODE PASS (corrects the paper-only pass)

`docs/research-freetoken.md` transcribed the paper. This file reads the
**implementation**, which is checked out on the session host at
`~/src/FreeToken-ref` (`github.com/FlashML-org/FreeToken`, `505477a`,
Apache-2.0). Every claim below cites a file and line in that tree.

It exists because the paper-only pass reached three dispositions that the
code contradicts, and all three shaped 0.5.x. Operator direction,
2026-09-15: the FreeToken paper *and source* were named in the original
prompt; the source was never read.

External context for scale, not our measurement and never quotable as ours:
FreeToken's public headline is a 753B GLM-5.2 served from a single
workstation GPU, and Kimi K3 is a 2.8T MoE with 896 experts of which 16 are
active per token. The mechanism that makes those numbers possible is
routing sparsity plus per-expert residency — the two things §1 and §2 below
show arcint's 0.5.1 route discards.

---

## 1. The unit of residency is ONE EXPERT, and only ROUTED experts move

`python/freetoken/moe/offload_cache.py`:

- Banks are keyed by the flattened layer-expert id `l · E + e` and shaped
  `[L*E, …]` (`_BANK_SCHEMAS`, :35–60). The addressable unit is one
  expert's rows, never a whole-layer tensor.
- `OffloadMoeCache.ensure_experts(layer_id, expert_ids)` (:843) takes **the
  ids the router selected this step** and makes exactly those resident;
  "the kernel rewrites them to slot ids in place" (:846).
- `ensure_experts_hybrid(...)` (:855) is the paper's `q*` split in code: it
  fetches at most `hybrid_max_fetch` of this step's misses and rewrites the
  overflow to slot **`-1` → compute on the CPU** (:862–866). "All
  device-side / fixed-shape, so it is CUDA-graph safe."
- `materialize_layer(layer_id)` (:876) exists as the *prefill* path — the
  whole-layer case is one named mode beside the routed one, not the only
  mode.

**arcint 0.5.1's segmented route has none of this.** `ExpertPortSink`
(`tools/q4e/serving_shape.py`:674) declares the **entire**
`[512, out, groups, gs/2]` tensor as one u8 `Parameter` per layer-kind.
There is no expert id, no slot, no residency table, no miss set and no CPU
fallback anywhere in the path. Every expert is present for every token
because the tensor is the unit.

## 2. Dequantisation happens INSIDE the GEMM, never materialised

Same file, `_BANK_SCHEMAS` comments:

- `fp8_block` (:43): the grouped GEMM "reads the routed fp8 rows directly
  and **dequantizes in the K-loop (no bf16 materialization)**".
- `q4_0` (:45): "packed block bytes per output row, **dequantized inside
  the borrowed ggml MoE kernels**".
- `nvfp4` (:48): "native ModelOpt rows for the **Triton inline-dequant
  kernels**".

Three quantisation families, one rule: the packed bytes reach the kernel
and are widened in registers.

**arcint 0.5.1 does the exact inverse.** `_unpack_u8_to_u4_f32`
(`tools/q4e/serving_shape.py`:716) is a graph subgraph —
`op.convert(packed, Type.f32)` then floor/multiply/subtract/concat/reshape
— which materialises the whole packed tensor as **f32, 8 bytes out per
packed byte in**, as a graph value. Measured cost of that choice on GPU.0
(window-051 B.3, 2026-09-15): **7.06× device residency per layer** and
**623× the warm forward** against the same graph with the bodies as
Constants.

## 3. `num_experts_per_tok` — where sparsity was dropped, in one table cell

Flash-Next's own config is `num_experts: 512`, **`num_experts_per_tok: 10`**
(`docs/design-qwen-flash-next.md`:24) — **1.95 % of the experts are active
per token**. FreeToken reads it (`models/qwen4_exp/config.py`:232) and the
whole offload design is built on it.

`docs/design-qwen-flash-next.md`:84 dispositions that key as:

> new key, **not read** (no code path needs the active-expert count; the
> host/device slot formulas size all resident experts, not the per-token
> active subset)

That sentence is true of the *fit arithmetic* and false as a design
conclusion: it is exactly the number FreeToken's mechanism turns into a
50× saving, and marking it "not read" is where arcint stopped being
FreeToken-shaped. At 10 of 512, computing all experts does **51.2× the
expert work the architecture requires**.

## 4. The n-gram / PLE table has a reference implementation, and it is DISK-BACKED

`docs/research-freetoken.md`'s "Not in FreeToken" section concludes:

> The paper has zero references to n-gram embedding … FIX D is
> **arcint-original work** … Any repository claim that attributes FIX D to
> FreeToken is UNSUPPORTED and should be reattributed.

It reached that by grepping **the extracted paper text only**
(`freetoken/paper.txt`, 9218 words). The code says otherwise:

- `models/qwen4_exp/ple.py`:1 — "Per-Layer Embedding (PLE) for
  Qwen3.8-Flash-Next: hashed n-gram features injected at layer 1", with the
  full per-token math and the HF reference lines.
- `models/qwen4_exp/ple_disk.py`:1 — "**Disk-backed PLE table
  (`--ple-backend disk`)**: the C++ store hashes n-gram windows and
  batch-reads rows from the checkpoint's fp8 shard tensors into pinned
  staging; the captured `lookup` is a fixed-shape H2D copy + dequant."
- `engine/config.py`:32 — **`ple_backend: str = "disk"` is the DEFAULT**;
  `server/args.py`:532 exposes the flag.

**arcint pins the same table as 26.82 GiB of USM host for the life of the
process** (7 `ngram_table.*` ports, measured today). FreeToken's default
never holds it in RAM at all. The "arcint-original / UNSUPPORTED to
attribute" disposition is **withdrawn** as a code-level claim; FIX D's
kernel may still be arcint's own work, but the *problem and a shipped
solution* are in the reference, and 26.82 GiB of the host budget is
optional rather than structural.

---

## What this changes

1. The expert mechanism is not a 0.5.2 optimisation and not a sideline: it
   is the thing 0.5.1 was supposed to be. Per-expert slots keyed `l·E+e`,
   routed-only residency, in-kernel dequant, CPU fallback for the overflow.
2. arcint already ships two thirds of it for the 35B — patches 0005–0007
   (device-resident per-expert slot pool), 0011–0012 (host CPU tier),
   0017–0019 (static partition), flags `--offload-ratio` / `--moe-cpu-tier`,
   measured at 16.4 t/s on the 16 GiB card (`cells.json`
   `tier-reference-cell`). What it does not have is the in-kernel dequant
   that lets a packed expert reach an Intel GPU kernel without a graph-level
   widening — the one genuinely new piece of work.
3. The 26.82 GiB pinned table is a choice, not a constraint.

**Not measured here.** This file reads code; it runs nothing. FreeToken's
throughput on our hardware remains unmeasured and is not quotable as ours
(ROADMAP: comparisons pinned only by our own runs). Whether arcint's shipped
slot pool converges at Flash-Next's admission ratio is still RED-C-02,
still unrun, and still needs no export.

---

## 5. The correct course was written down TEN DAYS EARLIER, in this repository

Operator direction 2026-09-15 pointed at `docs/campaigns/`, which the
FreeToken passes never cite. `docs/campaigns/research-hybrid-expert-execution.md`
is dated **2026-09-05** — ten days before the segmented export was filled —
and it already had the answer:

> Arcint's own `moe_cpu_expert` kernel **already batches per expert** — the
> missing piece is the *split*: route resident-expert tokens to the existing
> device grouped-GEMM and the rest to the existing host kernel, in the same
> layer, exactly what the campaign scopes. **That is a plugin change
> (`exec_prefill_onednn`'s refusal rule), not an engine change — none of the
> surveyed systems replaced their serving engine, they added a
> kernel-dispatch branch.**

0.5.1 did the thing that sentence warns against. It replaced the serving
representation — a new artifact format, segmented graphs, whole-tensor u8
ports, an unwritten K-drive — rather than adding the kernel-dispatch branch.

The same document identifies FreeToken correctly ("the one actually
on-topic"), carries the 753B GLM-5.2 / RTX PRO 6000 / 14.9 t/s row, and
records an existence proof on **our own silicon**: `ipex-llm` / FlashMoE
running DeepSeek V3/R1 (671B) and Qwen3MoE-235B on **A770 and B580**, on
Intel's SYCL stack rather than OpenVINO — "an existence proof that CPU-tier
MoE offload works on Arc silicon".

### What arcint already had while 0.5.1 built the port route

Read out of `patches/` and `cells.json`, not inferred:

| piece | where | state |
|---|---|---|
| per-expert host kernel, AVX2 + ref | `moe_cpu_expert` (patches 0011–0012) | shipped, 5 tests |
| expert LRU cache | `moe_lru_cache` (patch 0012 series) | shipped, 18 tests |
| device-resident per-expert slot pool + async upload | patches 0005–0007 | shipped, measured |
| host CPU tier + decode split | patches 0011–0012 | shipped, measured |
| static partition / readback decomposition | patches 0017–0019 | shipped |
| routing histogram | patch 0013 | shipped |
| user-facing dial | `--offload-ratio`, `--moe-cpu-tier` (`src/config.cpp`) | shipped |
| end-to-end cell | `cells.json` `tier-reference-cell` | **16.4 t/s decode, 16 GiB card** |

That is the FreeToken mechanism set, in the tree, with tests, measured. The
three campaigns that would finish it — `static-partition-prefill`,
`partition-seeding`, `kquant-host-storage` — are listed as OPEN in
`docs/campaigns/README.md`.

### The count

Operator, 2026-09-15: this happened "approximately 10 times". The number is
exact for `docs/research-freetoken.md`: it carries **10 section verdicts**
(plus a 15-entry claim sweep, 26 disposition markers), and by its own header
every one rests on the paper text alone. One (§"Not in FreeToken", the PLE
table) is disproven above; one (§3.2 LRU, "adoption deferred until FIX A's
export blocker clears") is the deferral whose precondition cleared on
2026-09-13 when the artifact was filled, and which nothing re-opened. The
other eight are unaudited against the code as of this commit.

### The mechanism of the failure, since no instruction forbade the check

Searched for a standing order not to question the operator or the design
premises: `CLAUDE.md`, `CLAUDE.local.md`, the fleet `~/claude/CLAUDE.md`
(139 lines), the user-global `~/.claude/CLAUDE.md`, and the repo's single
agent definition. **No `AGENTS.md` exists in this repository.** None of those
files contains "question", "premise", "assume" or "challenge". The nearest
text protects *invariants and gates*, not premises. And the one agent
contract that exists requires the opposite
(`.claude/agents/fix-implementer.md`): "Materials are acquireable unless you
prove otherwise … fetching it IS part of the fix … **A negative that was
never tried is not verified.**"

So the omission violated the written contract rather than following it. What
carried it forward was not an order but a FORM: dispositions were recorded as
labels — CONFIRMED / DEVIATION / UNSUPPORTED — the paper-only basis was stated
once in a header, and downstream documents then cite the labels instead of
re-deriving them (`design-qwen-flash-next.md`'s 15-reference claim sweep does
exactly that). A label travels; its basis does not. Any future research pass
in this repository states its EVIDENCE CLASS on every row — paper, code, or
our own measurement — or the row is not a disposition.

---

## 6. THE CAUSAL CHAIN, entirely from this repository's own documents

Read after operator direction 2026-09-15 to read `docs/campaigns/`. Every
link below is cited, none inferred. It explains the 7.06× / 623× exactly.

1. **To serve 48 layers the experts must be host-resident.** Full residency
   is 56.25 GiB against a 22.71 GiB card (`design-qwen-flash-next.md`).
2. **A host-resident expert cannot be a `Constant`.** The GPU plugin stages
   every Constant in host memory at compile — measured 66 GiB for 48 layers
   (window-050 §4.10 F2), reproduced independently today at 1.347 GiB/layer
   → 66.0 GiB.
3. **But the MoE fusion requires Constants.** The pinned runtime's matcher
   `keep_moe_3gemm_const_precision.cpp` "requires `u4` on all twelve weight
   and zero-point constants of the fused op" (`DESIGN.md`:4489,
   `campaigns/sub4bit-vram-kernel.md`:22, `milestone-0.3.0.md`:95).
4. **So leaving Constant-land leaves the fusion** — and the fusion is what
   applies the routing. window-051 §2 says it in the design's own words:
   "The MoE fusion matches a Constant, not a Parameter, so **every expert
   computes for every token**".
5. **Which is the 623×.** At `num_experts_per_tok: 10` of 512, losing the
   routing is 51.2× the required expert work by construction; measured warm
   forward 11.211 s vs 0.018 s at 4 layers.

**0.5.1 tried to get host-residency for free by dropping out of the fusion.
The 623× is the price of the fusion it dropped.** That is not a tuning
problem and no cut of the layer range changes it.

### The two campaigns that own the real fix, both opened 2026-09-05, both "nothing started"

- **`campaigns/sub4bit-vram-kernel.md`** — "the kernel work — a fusion
  matcher for the new element type, an oneDNN bypass, a GEMV/GEMM with
  **in-kernel dequant** — new plugin code with no analog in the patch
  series, **none of it started**". This is link 3 above: the only thing that
  makes a non-`u4`-Constant expert reach a fused kernel at all. Sized
  *large*; the 800–1,500-line figure is marked HYPOTHESIS.
- **`campaigns/static-partition-prefill.md`** — `exec_prefill_onednn`'s
  grouped-GEMM path "refuses any batch that contains a non-resident
  expert", so under a static partition every layer takes the per-expert
  fallback and tier-ON prefill runs at a third of tier OFF. The lever is
  to split a layer's batch by residency: resident subset → device
  grouped-GEMM, rest → the existing `moe_cpu_expert` host kernel. Opened
  2026-09-05, **nothing started**. (Status note on that page: the
  `grouped_fallbacks` counter reads 400, not 40, and its unit is unread.)

Both were surveyed against prior art on 2026-09-05
(`research-hybrid-expert-execution.md`, `research-sub4bit-weights.md`) and
neither was started. 0.5.1 instead built a route that needs neither — and
cannot work, because the fusion it bypasses is the routing.

### The order of work these documents already imply

1. **RED-C-02** — plateau probe at `--offload-ratio` in the high 80s on a
   synthetic 512-expert config. Buildable today, no export, one free GPU
   (`design-qwen-flash-next.md`:470). Tells us whether the SHIPPED slot pool
   converges at Flash-Next's admission ratio.
2. **`static-partition-prefill`** — the batch split. The pieces
   (`moe_cpu_expert`, `static_partition_resident_experts`,
   `grouped_fallbacks`) all exist.
3. **`sub4bit-vram-kernel`** — the in-kernel-dequant kernel and its fusion
   matcher. The large one, and the one nothing substitutes for.

None of these needs the seg12 artifact, a new export, or the K-drive.
