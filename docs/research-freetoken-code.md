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
