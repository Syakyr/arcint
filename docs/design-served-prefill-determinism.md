# design-served-prefill-determinism — the run-to-run floor of the served
# Flash-Next path, and why it is not §3.2

Companion to `docs/campaigns/served-prefill-determinism.md`. Recon first, then
the hypothesis ranking and the discriminators. No code decided yet.

## Recon — what is actually there

**§3.2 is a different defect, and it is already fixed.** DESIGN.md §3.2
isolates the chunk non-bit-exactness to the GDN recurrent scan ("advancing the
state by two tokens in one call is a different computation from advancing it
twice by one … a property of `ocl::gated_delta_net::ref`, not of arcint"), and
the shipped fix is the **ABSOLUTE GRID**: prefill chunks and cache
checkpoints are placed on multiples of the chunk size counted from position 0,
so "**no two paths ever hand the model a different split of the same
tokens**". That makes different *splits* equal. It says nothing about two runs
of the *same* split.

**Our defect is same-split.** Two replays of the same 2,735-token window, in
one process, on the same absolute grid, differ by `KL(A‖B)` mean **0.1361
(w0) / 0.1512 (w1)**, max |logit diff| **13.2 / 17.8**, and the argmax agrees
at only **0.850 / 0.902** of scored positions. §3.2's own split effect is
"up to ~2.8" absolute; ours is 13–18. So it is **not** the §3.2 family, and the
absolute grid does not touch it.

**The host CPU expert tier is a deliberately different numeric path.** Plugin
patches 0011/0012/0013/0017/0018/0043 implement it; 0011 states verbatim:

> "the GPU accumulates each lane's within-fake-group partial products in
> `half` … the CPU reference (`moe_cpu_expert_ref`) accumulates the whole dot
> product in f32 throughout … **the divergence this introduces is small — but
> it is not zero**, and it is why G1b (patches/0012) is a tolerance gate rather
> than a byte-exact one."

So the same expert, on the GPU and on the host, yields **different** outputs,
by design. The served path is `--offload-ratio 99 --moe-cpu-tier`, i.e. almost
everything is host-computed, but the **slot pool still decides what is
device-resident** — and which experts run where is exactly the state that can
differ between replays.

**The knobs exist** (no code needed to discriminate): `--offload-ratio`
(0 = all resident), `--prefill-chunk` (0 = unchunked), and
`--moe-cpu-tier-threads N` (0 = `hardware_concurrency()-1`).

## Hypotheses, ordered

- **H1 — residency mix (leading). [CORRECTED 2026-09-20: REFUTED — the
  force-the-tier arm leaves 0.081553, so the GPU/host mix is not the cause;
  the leading shape is now a within-kernel GDN nondeterminism on Xe2.]**
  Which experts are device-resident vs
  host-tier computed differs between the two replays (slot-pool/LRU state, or
  a warm-up ramp). The same expert then crosses the f16/f32 divergence above,
  and over 48 layers with routing the difference compounds to 13–18 absolute.
  It also explains why the *first* replay and the *second* differ while each
  is internally consistent.
- **H2 — kernel/config selection.** GPU kernel or tuning selection varying
  between forwards, or the `ocl:ref` fallback path mixing with the fast path.
- **H3 — thread order.** The tier's worker threads reordering a reduction.
  Weak by construction: 0011's kernel accumulates each expert's dot in a
  **fixed** chain per column (multiple accumulators combined in a fixed
  order), so thread count should not change one expert's bytes.
- **H4 — §3.2 family.** Ruled out *as the whole cause*: the absolute grid is
  in force and the same split still differs.

## Discriminators — all existing flags, one card window

| arm | command delta | what it decides |
|---|---|---|
| D1 | `--offload-ratio 0` (tier off, all resident) | H1: if the floor collapses to ~0, the GPU/host mix is the cause |
| D2 | `--warmup N` before the two counted replays | H1-warm-up: if the floor collapses once residency settles, it is ramp state |
| D3 | `--moe-cpu-tier-threads 1` | H3: expected null; if it bites, the kernel's combination is not fixed |
| D4 | `--prefill-chunk 0` (unchunked) | H2/H4: if the floor collapses, the chunk path carries it |

D1 is the decisive one and is also the cheapest to interpret. Note its caveat:
**tier-on and tier-off are not byte-identical** (0011 says so; 0012's G1b is a
tolerance gate). D1 therefore measures the *floor*, not an equality.

## Gate

`F_served ≤ bar_0.5.1` on both capture windows, by the existing floor
instrument (`KL(A‖B)` over ≥ 2 replays), with the decided bound printed beside
it. Today the floor is ~44x the bound, so the gate is RED by clause (d)'s
UNREADABLE rule. **The gate is about reproducibility, not fidelity.**

## Entry criteria

- The campaign's `-001` acceptance commit lands first (this document is the
  design note, not that commit).
- The card window ritual: sampler first, abort if not growing; hold the host;
  restore nothing (the units are inactive by design between legs).
- A replay `--timeout` that exceeds the per-window cost (~3,600 s), or the leg
  is cut before a second replay exists — measured, and the failure mode of
  every earlier `F_served` attempt.

## Scope — in / out

**In:** the determinism of *which* numeric path an expert takes (residency /
slot-pool state); a single-chunk gate arm; a red-first cell that fails when
two forwards of the same ids differ.
**Out:** the tier kernel's arithmetic (fixed and deliberate); the model-quality
term (inside the floor); the f16-state/attention hypotheses (falsified by the
artifact IR); #37607's cache path (closed-not-fixed, its own workaround).

## Where it lives

Executor backend prefill chunking and the absolute grid; plugin patches
0011/0012/0017/0018/0043 (slot pool + tier); the floor/compare instrument in
`tools/`; the upstream tracker openvinotoolkit/openvino **#38099** (OPEN,
acknowledged, unfixed) for the GPU chunked-GDN family.

## Pipeline

recon (this document) → design decision (pin the residency set per layer per
step, or serialise the tier path for the gate arm) → red-first implementation
(a cell that goes RED when two identical forwards differ) → one card window →
review before commit → DESIGN §7.0.2x record + CHANGELOG line on closure.

## Invariants

DESIGN §3.4 (history-independent greedy output); LISBON-001's restart
determinism ("two cold boots, byte-identical answers"); the §5 ladder; the
measurement discipline in CLAUDE.md. A change that trades determinism for a
number does not close.

## Status

**Evidence classes:** an entry below is `[measured-here]` unless it names a
different class (`[code]`, `[paper]`).

- 2026-09-19: recon written; **H1 (residency mix across the tier's
  deliberate f16/f32 divergence) is the leading hypothesis**; D1–D4 listed as
  no-code discriminators for one card window; §3.2 distinguished and its fix
  noted as insufficient for this defect; upstream #38099 confirmed OPEN.
  Not committed — CLAUDE.md requires Fable review before every commit.
- 2026-09-20 (**D2 read**; table in the campaign's dated entry): the warm-up
  **halves** the floor — 0.1361 (no warm-up) → 0.0726 / 0.0853 (warmup vs
  replay 0 / 1) — but the warmed pair (replay 0 vs replay 1) is still
  **0.073234**, ~**24x** the decided bound, with **114/1367 (8.3 %) argmax
  flips** and `bit_identical False`.
- 2026-09-20 — **CORRECTION (same day): that reading is confounded.**
  `--repeat` is rep-major, so the earlier leg's 0.1361 pair straddled a
  window-1 forward; D2's pairs are consecutive same-content forwards, and
  D2 shows the warm-up adds **nothing** (warmup vs replay 0 = 0.072601 ~=
  replay 0 vs replay 1 = 0.073234). **H1 (cold->warm ramp) is NOT
  supported.** The floor tracks **content/residency churn between forwards**
  (a different window in between: 0.136105, and 0.085348 across a
  same-content replay), over an intrinsic **~0.073** floor for two identical
  consecutive forwards. **H2 — steady-state per-launch variance (kernel/
  config selection, or the tier's steady-state GPU/host mix) — is the
  leading candidate**, and the force-the-tier arm (below) is the binary
  that splits it.
  1. **force the tier** (`ARCINT_MOE_DEVICE_POOL_BYTES` small, so every
     expert takes the host kernel): if the warmed floor collapses, the
     GPU/host MIX is the cause; if it persists, it is intrinsic kernel
     variance (and #38099's family becomes the suspect).
  2. **D4 unchunked** (`--prefill-chunk 0`, warmed pair): chunk path or not.
  3. **D3 threads=1**: expected null — 0011's per-expert accumulation chain
     is fixed, so thread count should not change one expert's bytes.
  4. `--cut layerN/out` on the boot driver for the first divergent node —
     needs the 4 GiB watchdog relaxed or a smaller working set.
- 2026-09-20 (**force-the-tier read**): pool forced to 16 MiB (no expert fits)
  -> `warmup vs replay 0` mean **0.081553**, 137/1367 moved, argmax 0.8998,
  `bitid False`. **The residency mix is refuted**; the divergence survives with
  every expert on the identical host path.
- 2026-09-20 — **the prior art this note should have opened first**:
  `docs/window-051.md`'s cut table already bisects this. **The variable is the
  card.** On the B60 (Xe2) layer 0 carries a **per-forward 1-f16-ulp step at or
  after row 32** (1-3 of 1,024 rows), amplified by depth to 11 % argmax
  agreement at 12 layers; on the **A770 the same bytes are bit-identical**
  (x8, x12), and at depth 4 the A770's floor pair is bit-identical x3. The
  kernel is not localised: *"the kernel-level cut needs names the emitter does
  not set yet"*. So the two live attributions are (i) **the chunked path**
  (window-051 clause (d): chunk-boundary non-exactness, fix = a single-chunk
  prefill or a bit-exact chunk-carry) and (ii) **the B60's layer-0 kernel**
  (fix = name it, or measure on the A770). **D4 `--prefill-chunk 0` is the
  discriminator**; it is running (d48n, ratio 99 + tier, B60 GPU.0, warmup 1
  repeat 2 windows 0, launch 09:47Z). `config.h:97` states the claim being
  tested: a prompt inside a single chunk is *"bit-identical to an unchunked
  run"*.
- 2026-09-20 (**D4 read**): **the chunk attribution is refuted — unchunked is
  worse, not better.** `--prefill-chunk 0` on the same served config gives
  `warmup↔r0` mean **0.095497** and `r0↔r1` mean **0.202143** (**284/1367
  moved**, argmax **0.7922**), against chunk-512's 0.072601 / 0.073234 and
  force-the-tier's 0.081553. So the depth-48 served floor is **not §3.2's
  chunk non-exactness**, and the fix is **not a single-chunk prefill**.
  **Attribution (ii) stands: the B60/Xe2 layer-0 per-forward step** (one f16
  ulp at or after row 32; the A770 bit-identical). Consequence for
  BERLIN-001: clause (d) cannot be closed by a serving config — it closes by
  taking the measurement on the **A770**, or by naming and pinning the kernel.
  Next: the depth ladder (**4 -> 2 -> 1**) and the card A/B, per
  `docs/handoff-served-prefill-determinism.md`.
- 2026-09-20 (**A770 floor = 0** [measured-here]): the A770 depth-48 served
  path is bit-identical (r0↔r1, 0/1367 moved, argmax 1.0000, maxdiff 0.000),
  so `F_served = 0` there and BERLIN-001 clause (d) reads READABLE on the A770
  as the measurement card. Caveats kept: it is **×2** (the d4/d12 evidence is
  x8/x12) and §4.11's "A770 steps once" is not refuted by two forwards — a
  repeat-8 A770 arm is queued.
- 2026-09-20 (**CORRECTION: the width is a correlate, the pin is DEAD**
  [code, measured-here]): `xe2` requires subgroup size 16
  (`intel_reqd_sub_group_size(8)` fails to compile on every Xe2 target:
  `bmg-g21`, `bmg-g31`, `lnl-m`, `ptl-h`), and the disassembled
  `sub_group_reduce_add` is a fixed tree at both widths. So the width explains
  the card-to-card VALUE difference, NOT the run-to-run variance.
- 2026-09-20 (**the mechanism narrows to a within-kernel nondeterminism**
  [measured-here]): the peer session refuted inter-kernel ordering (`clFinish`
  after each of 233 enqueues), the JIT (`ocloc` twice -> byte-identical), the
  launch geometry (static dispatch, `!params.is_dynamic()`), and the write
  mapping (scattered, no contiguous overlap); the minimal same-shape gemm is
  bit-identical x8. Location `layer0/mixer_out` with input-bit-identity across
  8 repeats; fingerprint `dim0 = row 0`, heads
  [3,5,6,7,10,13,17,22,31,39,41,42,43,47], one f16 ulp, flip count
  2423..3924. The sibling report is posted to
  `openvinotoolkit/openvino#38099` (`issuecomment-5751935449`).
