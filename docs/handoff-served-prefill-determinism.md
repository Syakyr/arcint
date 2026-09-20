# handoff — served-prefill-determinism: the B60 run-to-run defect and the A770 gate path

**For:** the next session that picks up `docs/campaigns/served-prefill-determinism.md`.
**Read first (RTFM mandate):** this file, then the campaign doc, then
`docs/design-served-prefill-determinism.md`, then `docs/window-051.md`
**clause (d)** *and* its **cut-bisect table**, then `DESIGN.md` §3.2, then
`patches/0011` and `patches/0043`.
**Operational packet** (hosts, card map, paths, commands, traps):
`docs/handoff-served-prefill-determinism.local.md` — git-ignored, because this
repository is public and operator-local facts do not belong in it.

---

## 0. The defect, in one paragraph

The served Flash-Next path (**B60 / `GPU.0` / `bmg-g21`**, `--offload-ratio 99
--moe-cpu-tier`, f16, chunk 512) is **not run-to-run deterministic at long
context**. Two forwards of the *same* 2,735-token window in one process differ
by `KL(A‖B)` mean **0.1361 / 0.1512**, max |logit diff| 13.2 / 17.8, argmax
agreement **0.850 / 0.902**. The decided KLD bound (3.0905e-03) sits ~44x
*below* that floor, so on the B60 clause (d)'s row reads **UNREADABLE, not
PASS**. **On the A770 the same bytes, request and harness are bit-identical**,
so the gate closes there instead — see §3.

## 1. What is established (evidence class in brackets)

| finding | number | where |
|---|---|---|
| served floor, both windows, 4 replays [measured-here] | 0.1361 / 0.1512, argmax 0.850 / 0.902 | `window-051.md` clause (d) |
| warm-up is NOT the cause [measured-here] | warmup↔r0 **0.072601** ≈ r0↔r1 **0.073234** | `d48n-d2-warm.bin` |
| GPU/host residency mix is NOT the cause [measured-here] | force-the-tier floor **0.081553**, 137/1367 moved | `d48n-tieronly.bin` |
| chunking is NOT the cause [measured-here] | unchunked `--prefill-chunk 0` is **worse**: **0.095497 / 0.202143** | `d48n-d4-unchunked.bin` |
| **the A770 d48 served floor is ZERO** [measured-here] | r0↔r1 **-0.000000**, **0/1367 moved**, argmax 1.0000, `bit-identical True` (×2) | `d48n-a770-d48.bin` |
| the defect is card-dependent [measured-here, code] | B60/Xe2 steps; **A770/acm is bit-identical** (×8/×12 at **depth 4**, ×2 at depth 48) | `window-051.md` cut table |
| depth brackets it [measured-here] | present at **depth 4** (layer2/out, logits) and **depth 12** (layer0/out, ple/out) | `window-051.md` cut table |
| location, input-bit-identity proven [measured-here] | `layer0/mixer_out`; nine input ports bit-identical across 8 repeats, both state tables the all-zero hash, output differs every forward | peer session, `2026-09-20` |
| **fingerprint** [measured-here] | `dim0 = row 0`; heads `[3,5,6,7,10,13,17,22,31,39,41,42,43,47]`; one f16 ulp `9.7656e-4`; flip count 2423..3924; head set invariant | peer session, `2026-09-20` |
| the width is a correlate, NOT the mechanism [code, measured-here] | `xe2` **requires** subgroup size 16 (`intel_reqd_sub_group_size(8)` fails on `bmg-g21`/`bmg-g31`/`lnl-m`/`ptl-h`); the lowered reduce is a fixed tree at both widths | peer session, posted to `openvinotoolkit/openvino#38099` |
| the JIT is not the source [measured-here] | `ocloc` compiles the bucket twice to **byte-identical** binaries (`be20f259…dc78`, 64,712 B) | peer session |
| serialization does not fix it [measured-here] | `clFinish` after **every** of 233 enqueues: repeats still differ | peer session |
| the upstream dense gemm is clean [measured-here] | minimal same-shape f16 MatMul bit-identical ×8 | peer session |
| upstream | **#38099 OPEN** (chunked GDN unroll, wrong values chunk≥2, no fix); our sibling comment posted `2026-09-20` | GitHub |
| bar status | 3.0905e-03 is an **instrument-resolution bound**, not an acceptance bar; per-row max 1.0533e-02; between-implementations floor median 0.0649 / 0.0283 | `window-051.md` amendments |

**The mechanism, stated as it stands.** Every candidate is now excluded by
measurement or by code: warm-up, the GPU/host residency mix, chunking, launch
geometry (`get_dispatch_data_func` is static and `!params.is_dynamic()` is
asserted), the source itself (no `atomic`/`barrier`/`__local`/`volatile`), the
reduction lowering (fixed tree at both widths), the JIT (byte-identical), and
inter-kernel ordering (serialization leaves it varying). What remains is a
**within-kernel nondeterminism in the GDN arithmetic on Xe2**, at execution
level, below the kernel-choice level. The GDN state digest is **stochastic**
(5 distinct hashes in one process; 1–4 among repeats in cold processes; the
**first** forward is reproducible across cold processes), while the co-resident
conv state is stable every repeat. The defect's shape is the fingerprint above:
a stable, data-dependent head set with a varying count of one-ulp flips.

**Sharp consequence:** the `ref` GDN kernel cannot be selected in the shipped
plugin build (`OV_GPU_FORCE_IMPLEMENTATIONS` needs `ENABLE_DEBUG_CAPS`, absent),
so in-tree selection is impossible until a debug-caps build exists. The
in-tree options are therefore the **A770 measurement card** (already measured
clean) or upstream. A debug-caps rebuild is the only in-tree route to testing
`ref` as a workaround.

## 2. The job: turn hours into seconds

The 3,600 s per forward is **not the kernel under test** — it is 48 layers ×
512 experts × the host tier × 2,735 tokens, plus a ~37-minute artifact load
(the A770 leg, 2026-09-20: launch 14:22:12Z to its first counted forward).
One block at the same shape is milliseconds. So build a **standalone graph of
the suspect block** and run it twice, bit-exact-compared.

**Bisect the depth DOWNWARD — start below 4.** The defect is present at depth 4,
so the ladder is **4 → 2 → 1**. At depth 12 `layer0/out` already differs; if a
**1-layer** artifact shows it, the repro is a single block: seconds to load and
milliseconds to forward.

**The chunk axis is a probe axis, not the cause.** D4 refuted chunking as the
*floor* cause (unchunked is worse), so a chunk sweep is only useful to find
which cached-kernel regime carries the defect — do not read it as the
mechanism.

**Hold these fixed or the defect vanishes** (the compiler will legitimately
pick a different kernel): device (`GPU.0`/B60), dtype (f16), layout, the
chunk/unroll count, and the compile options.

**The zeros idea — right instinct, wrong lever.** A dense GPU kernel does not
skip zeros (no shortcut without structured-sparsity metadata this plugin does
not use). Worse, the weights are **Constants** in the IR, so an all-zero
Constant MatMul gets **constant-folded away** and the op disappears. Where
zeros help is as an **oracle**: with a trivially-known input the correct output
is exactly 0, so any run-to-run difference is unambiguously the kernel.

**The sharpest artefact is the card A/B at the minimal shape:** same tiny
graph, same tokens, same dtype — **B60 steps, A770 bit-identical**. That is the
reproducer for the upstream report, and it needs no engineer's attention to be
credible. **Reproduce the fingerprint or you are measuring something else** —
`dim0 = row 0`, the same 14 heads, one f16 ulp.

**State the pair, not just the number.** With the first forward reproducible
and later ones stochastic, every floor must name its **transition**
(`#1↔#2`, `#2↔#3`, …). A warmup+2-repeat arm and a 0-warmup 2-repeat arm are
different observations on this card.

**#38099's existing reproducer is a DIFFERENT defect** — deterministic wrong
values for chunk ≥ 2. Ours is run-to-run at execution level. Siblings in the
same unroll, not the same bug.

## 3. What "fixed" means — and the odds (judgment, not measurement)

| level | requirement | P(we do it, no Intel) |
|---|---|---|
| gate readable (BERLIN-001 clause (d) closes) | **DONE 2026-09-20**: A770 d48 served floor **0** → clause (d) reads READABLE on the A770 as the measurement card; the B60 stays a per-card caveat until the real mechanism is found | **met** |
| B60 deterministic | the width pin is **DEAD** (`xe2` requires 16). Find and fix the within-kernel nondeterminism in the GDN arithmetic (open), or test `ref` via a debug-caps rebuild if it proves deterministic and tolerable in rate | **~0.35, mechanism OPEN** |
| upstream kernel fixed | the sibling report is posted; a fix from Intel is the only repair of the B60's arithmetic | **~0.15–0.2** |

Risk on the middle row: if the variance is in **IGC/the driver** rather than in
a patchable kernel, we cannot fix it — the route-around is the A770 card, the
`ref` kernel, or upstream.

## 4. Floor read (the one code snippet that belongs here)

Do NOT use `--compare` for a single-window arm:

```python
# rows 1367..2733 = the tool's own scored subset; floor_pair on the dump's
# OWN window-0 replays, NOT across capture windows
from kld_served import read_dump, dump_windows, window_rows
import kld_bar
N_CTX, N_VOCAB = 2735, 248320; FIRST = N_CTX//2; N_ROWS = N_CTX-1-FIRST
ws = [w for w in dump_windows(read_dump(P)) if sum(r[3] for r in w
      if r[1] < N_CTX and r[1]+r[3] <= N_CTX) == N_CTX]
rows = [window_rows(P, w, N_CTX, N_VOCAB)[FIRST:FIRST+N_ROWS] for w in ws]
print(kld_bar.floor_pair(rows[-2], rows[-1]))
```

## 5. Traps (paths and commands in the `.local.md` packet)

1. **`--compare` misaligns a single-window arm** (it takes the LAST `n_chunk`
   replays as the capture's windows) — read the dump's own replays.
2. **`--repeat` is rep-major**, so a "w0 pair" straddles a w1 forward unless
   `--windows 0` is pinned.
3. **The server is `timeout -s KILL ${KILL}`-wrapped**; a long load can cut the
   last replay. The usable pair is (warmup, r0).
4. **Sampler watchdog at 4 GiB free**: the depth-48 **boot driver** dies there;
   the **served path** survives because it reuses state.
5. **The `data` host sleeps without a wake lock** — take one before any leg.
6. **`ARCINT_MOE_DEVICE_POOL_BYTES` must be exported** on tier windows, or the
   measurement is void.
7. **`pgrep -f` self-matches** — a guard matching its own watcher never clears.
8. Analysis outputs go to persistent paths, never `/tmp`.

## 6. House rules

- **A commit requires the external review.** `CLAUDE.md`: Fable review before
  every commit, without exception; this harness has no Fable agent and
  `.claude/agents/fix-implementer.md` substitutes the external reviewer.
- Campaign docs are a **dated, appended status log**; corrections go in place
  with a date.
- Every disposition carries an **evidence class**: `paper`, `code`, or
  `measured-here`.
- A negative never tried is not verified; a label is not evidence.
- Operator-local facts (hosts, paths, unit names) stay in `*.local.md`.
