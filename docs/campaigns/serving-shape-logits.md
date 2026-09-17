# serving-shape-logits — the served Flash-Next artifact's logits carry no information about the model; find the layer, or the fill, that loses it

## The defect, as measured

`measured-here`, 2026-09-18, Arc Pro B60, the 48-layer artifact in the
fused shape (`qwen3.8-flash-next-d48f`, tree da52858) served with
`marfrit-openvino +p18` at `--offload-ratio 99 --moe-cpu-tier`, KV u8,
f16 inference, one lane, from the NVMe: the llama.cpp logits capture of
the same GGUF (`qwen4exp-c2735-chunks2`, two windows of 2,735 tokens)
replayed through the server — mean KL(reference ‖ served) **12.42 / 12.16
nats** (window 0, below / above the midpoint) and **12.40 / 12.34**
(window 1), max 26–35, argmax agreement **0.0007 and 0.0000**. For scale, ln(248320) = 12.42 nats is
the KL a reference of zero entropy would score against a uniform served
distribution; with an argmax agreement of zero the served logits carry no
information about the model's own. The depth-12 rung read 11.93 / 11.63 on
2026-09-13 (`docs/window-051.md`) and the depth-4 rung about the same;
those were read as "44 (36) layers missing". A truncated prefix of a deep
model scores like that too, so those rungs could not decide between
"missing layers" and "a broken artifact"; the full-depth rung decides:
the artifact is broken, and nothing says at which depth.

The served output is deterministic: France raw, greedy, 8 tokens is
byte-identical across two cold starts and across two storage paths, and
warm repeats are identical. Whatever is wrong is wrong the same way every
time.

## Known against hypothesised

Known (`measured-here`): the fused MoE route computes the HF-exported 35B
correctly on the same plugin and card (Paris, 23.6 t/s); the CPU tier
matches the device kernel at depth 12 (KL 0.020 on the France prefill,
decode steps within 0.4 logits); the unfused and fused routes disagree at
depth 12 (KL 0.147 on the same prefill) — both are approximations and
neither has been compared to a reference at that depth. Known (`code`):
every emitted piece (GDN, attention, hyper-connections, PLE, MoE, the
n-gram gather) has a numpy reference and a bit-exact contract test at the
suite's reduced geometry (`tests/python/`, `q4e/ref_*.py`), on synthetic
weights; the real-geometry fill (`q4e/gguf_feed.py`, `q4e/expert_fill.py`,
`tools/export_serving_artifact.py`) has its own cells (the expert fill's
codes are pinned bit for bit against the C++ unpack; the dense fill is
checked against the GGUF's own dequantisation in `test_expert_fill.py`'s
shard-gated cells) but no cell compares a served forward at real geometry
against any reference. Known (`measured-here`, 2026-09-13): the n-gram
table's binding and the PLE hash ordinal were measured as mechanism, never
as values.

Hypothesised (unranked; none measured): the real-geometry structural
parameters the reduced geometry cannot exercise (rope span and theta, the
GDN geometry at real head counts, the hyper-connection lowrank, the
attention interval, `layer_types`); the dense fill's key mapping at real
geometry (a transposed or misnamed tensor passes shape checks and fails
values); the PLE / n-gram table's row addressing at the real vocabulary
(a wrong hash ordinal reads the wrong rows and produces deterministic
noise); the expert fill's expert ordering across layers; the served
binary's port feeding (`inputs_embeds` from the embedding model, the
position ids, the conv mask) against what the emitter expects.

## Gate

A served forward at depth 48 whose logits match the reference: mean KL
within the 0.5.1 bar (re-derived from this model's own reference
round-trip, not the 35B's 0.0599), argmax agreement above 0.9 on both
capture windows, and the Paris line — the model's own token for the
capital of France — or a named refusal that says which layer refuses.
Byte-identical across two cold starts (§3.4) stays required.

## Entry criteria

Met: a full-depth artifact that compiles and serves on one card
(8.06 GiB device at ratio 99 with the tier, +p18), an NVMe copy that makes
a 2,735-token replay a 20-minute cell, the reference capture staged where
the container reads it, the replay and compare tools (`tools/kld_served.py`),
a two-dump diff (`tools/logits_dump_diff.py`), and the cut localiser in
the boot driver (`--cut layerN/out`). Unmet: a per-layer reference at real
geometry — the numpy `ref_backbone` fed with the real GGUF at depth 1 (the
reduced-geometry tests already wire the same modules), or a llama.cpp
hidden-state tap.

## Scope — in / out

In: localising the loss layer by layer (depth 1 first: the embedding, the
first GDN block, the first MoE, the PLE layer) against a real-geometry
reference; the fill's key mapping and the structural parameters at real
geometry; the port feeding of the served binary; the fix in the emitter or
the fill; the re-export; the gate measurement.

Out: the MoE route's own numerics (fused against unfused at depth 12 is a
separate, smaller question once the artifact computes the model); the
residency stream's speed; the B60/A770 kernel differences.

## Where it lives

`tools/q4e/` (the emitter and the fill), `tools/export_serving_artifact.py`,
`tools/boot_serving_shape.py` (`--cut`, `--dump-logits`, `--stage forward`),
`tools/kld_served.py`, `tools/logits_dump_diff.py`, `docs/window-050.md`
and `docs/window-051.md` (the depth-4 and depth-12 records that were read
as depth), `docs/design-qwen-flash-next.md`; this campaign's own status
below; `sub4bit-vram-kernel.md` (status 2026-09-17/18, how the full-depth
serve was reached).

## Pipeline for this campaign

Recon: read the depth-4 KLD record and the feed-the-ports measurement of
2026-09-13 again with this verdict in hand → the reference at depth 1: the
numpy backbone over the real GGUF for one short prompt, against the served
cut at `layer0/out` (values, not shapes) → walk forward layer by layer
until the first divergence, then inward (embedding, norm, attention or
GDN, hyper-connection, MoE, PLE) → the fix, red-first in the reduced
geometry where the same bug can be planted → re-export at depth 4, KLD
against the reference (a depth-4 rung of a correct emitter must beat the
uniform floor by a wide margin even with 44 layers missing — that is the
cheap signal) → full depth, the gate → review, DESIGN record, CHANGELOG.

## Invariants

DESIGN §3.4 (history-independent greedy output) and §3.8; the measurement
discipline in `CLAUDE.md`; a served number without the reference beside
it is mechanism, not an answer.

## Status

- 2026-09-18 — opened from the full-depth KLD of the night before: the
  served logits carry no information at depth 48 (KL 12.4 nats, argmax
  agreement 0); the depth-12 and depth-4 rungs read the same and could not
  distinguish "missing layers" from "broken" — full depth does. Nothing
  localised yet.
