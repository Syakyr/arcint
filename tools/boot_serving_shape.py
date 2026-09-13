#!/usr/bin/env python3
"""ITEM 4: the serving-shape IR through the SERVED PATH's own load sequence,
on a card, one stage at a time, every verdict captured verbatim.

This is the reproducer the window manifest's §4.6 prediction names. It does
not fix anything and it does not fill anything: the expert bodies and every
dense weight are the sparse arena's unwritten pages (zeros), exactly as in the
`RUN@be57428` boot. What it adds over that boot is the SERVED PATH: the same
sequence `load_paged` runs in `src/exec/backend_ov.cpp`, transcribed stage by
stage so that the first stage to refuse names itself.

    stage   what runs                                   transcribed from
    build   build_serving_shape_ir(n_layers, seq_len)   (the IR under test)
    protos  state prototypes read off the STATEFUL       backend_ov.cpp:2565-2577
            graph's Variables (rank 3 -> conv table,
            rank 4 with a static tail -> GDN table)
    pass    ov::pass::SDPAToPagedAttention               backend_ov.cpp:2582
    compile compile_model(device, KV_CACHE_PRECISION)    backend_ov.cpp:2719, :2965
    request one InferRequest; f16 state rows bound       :3019, alloc_la_rows
            per la port; KV pools per key/value port     alloc_kv_pools
    forward inputs_embeds, position_ids, then the nine   :6141-6151
            index ports, in that order, then infer()

Since feed-the-ports the graph is dynamic in T and takes `inputs_embeds`,
so the served feed order is accepted end to end; the driver adds the two id
ports (q4e.ngram_ids) and conv_mask itself and says so. `--long N` runs one
more forward of N tokens on a fresh request (rope span, blocks past 2051).

`--no-pass` is the CONTROL: the stateful graph compiled and run directly, the
`RUN@be57428` form (INFERENCE_PRECISION_HINT f32, every declared port fed),
which proves the structure still lights up on the card before the served
path is asked to. `--probe` continues past the served forward's first refusal
with ONE labelled substitution (input_ids for inputs_embeds) so the next
signature is measured instead of guessed; it is not the served path and its
output says so.

Nothing here is a test. Run one leg per process:

    <venv>/bin/python tools/boot_serving_shape.py --layers 1 --stage pass
    <venv>/bin/python tools/boot_serving_shape.py --layers 1 --device GPU.1 \\
        --no-pass --ids 760,6511,314,9338,369
    <venv>/bin/python tools/boot_serving_shape.py --layers 4 --device GPU.1 \\
        --ids 760,6511,314,9338,369
"""
import argparse
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

KV_BLOCK_TOKENS = 16          # backend_ov.cpp:7674 kv_block_tokens_
ROWS_PER_LANE = 3             # backend_ov.cpp:2637 drafts_max_ + 3, MTP off
PAGED_KV_DEFAULT = "u8"       # config.h:175


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def one_line(exc):
    """The exception text on one line, the way `RUN@be57428`'s boot.log kept
    it: newlines become ' | ' so a grep finds the whole signature."""
    return f"{type(exc).__name__}: " + " | ".join(
        s.strip() for s in str(exc).splitlines() if s.strip())


def say(stage, text):
    print(f"BOOT [{stage}] {text}", flush=True)


def parse_ids(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def dims(port):
    ps = port.get_partial_shape()
    if ps.rank.is_dynamic:
        return None
    return [d.get_length() if d.is_static else -1 for d in ps]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--device", default=None,
                    help="GPU.0 / GPU.1 / CPU; absent = device-free stages only")
    ap.add_argument("--stage", choices=("pass", "compile", "forward"),
                    default="forward", help="how far to go (default: all)")
    ap.add_argument("--ids", default="",
                    help="prompt token ids, comma-separated; seq_len = their "
                         "count unless --seq-len is given")
    ap.add_argument("--shards", default=None,
                    help="REAL WEIGHTS: a directory of the model's GGUF shards. "
                         "The built layers, the PLE, the final mixer and the "
                         "head are filled from them (q4e.gguf_feed), the expert "
                         "bodies through q4e.expert_fill, the n-gram table is "
                         "bound from the GGUF's own IQ4_NL bytes, and the prompt "
                         "is embedded on the host from token_embd. Without it "
                         "every weight is an unwritten page (zeros).")
    ap.add_argument("--long", type=int, default=0,
                    help="after the prompt forward: ONE forward of this many "
                         "tokens (ids 1000, 1001, ...) on the same request -- "
                         "the graph is dynamic in T, so this measures a block "
                         "past the query block's length and, past 2051, the "
                         "rope span (feed-the-ports)")
    ap.add_argument("--no-pass", action="store_true",
                    help="CONTROL: skip the transformation, compile the "
                         "stateful graph directly, feed every declared port")
    ap.add_argument("--probe", action="store_true",
                    help="after the served forward's first refusal, continue "
                         "with input_ids in place of inputs_embeds (LABELLED)")
    ap.add_argument("--paged-kv", default=PAGED_KV_DEFAULT)
    ap.add_argument("--artifact", default=None,
                    help="FULL-DEPTH: instead of building, read the language "
                         "model of this ARTIFACT DIRECTORY (tools/export_serving_"
                         "artifact.py) -- the exact bytes the served binary "
                         "loads -- and run the same stages over it. --shards "
                         "still supplies the table and the embedding rows")
    ap.add_argument("--zeros", type=int, default=0,
                    help="prompt = this many tokens of id 0 (the served "
                         "activation probe's own feed, backend_ov.cpp "
                         "`embed_paged(lane, zeros)`); overrides --ids")
    ap.add_argument("--cut", default=None,
                    help="LOCALISER: after the pass, keep only the graph up to "
                         "the node with this friendly name (a Result is put on "
                         "it, the rest is dropped) and compile THAT. Names the "
                         "emitter sets: ple/gathered, ple/out, layerN/mixer_out, "
                         "layerN/out. Not the served path; its output says so.")
    # THE FLOOR'S MECHANISM (REVIEW ba2d5de F1 rider, 2026-09-13): the served
    # logits of a 2,735-token window are not run-to-run deterministic; these
    # four options turn the driver into the bisect that finds the first node
    # whose output differs between two identical forwards.
    ap.add_argument("--capture", default=None,
                    help="prompt = the first --take ids of window --window of "
                         "this llama.cpp --kl-divergence-base capture (the KLD "
                         "gate's own tokens; overrides --ids)")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--take", type=int, default=1024)
    ap.add_argument("--repeat", type=int, default=2,
                    help="run the prompt forward this many times and compare "
                         "every repeat against the first: bit-identical or "
                         "not, the first differing row, rows differing, max "
                         "|diff|, argmaxes moved (default 2: the second "
                         "forward that pays no kernel jit)")
    ap.add_argument("--fresh", action="store_true",
                    help="each repeat on a FRESH request (new state rows and "
                         "KV pools, table ports shared) instead of the served "
                         "path's re-used one")
    ap.add_argument("--precision", default=None,
                    help="INFERENCE_PRECISION_HINT for the post-pass compile "
                         "(f32 / f16); absent = the plugin's default, which is "
                         "what the served binary gets")
    args = ap.parse_args(argv)

    import openvino as ov
    from q4e import serving_shape as ss

    ids = parse_ids(args.ids)
    if args.zeros:
        ids = [0] * int(args.zeros)
    if args.capture:
        from kld_served import read_capture
        _n_ctx, _nv, n_chunk, ctoks, _rows = read_capture(args.capture)
        ids = [int(t) for t in ctoks[args.window][:args.take]]
        say("capture", f"{args.capture}: window {args.window} of {n_chunk}, the "
                       f"first {len(ids)} of {_n_ctx} ids")
    if not ids:
        ids = list(range(1000, 1008))
    T = len(ids)                       # the FEED's length; the graph is dynamic

    say("env", f"openvino {ov.get_version()} layers={args.layers} T={T} "
               f"device={args.device} pass={not args.no_pass} probe={args.probe}")

    # ---- build ----------------------------------------------------------------
    arena = ss.SparseArena()
    feed_ = filler = None
    if args.shards:
        from q4e import gguf_feed as gf
        from q4e import expert_fill as ef
        t0 = time.time()
        feed_ = gf.GgufFeed(args.shards)
        filler = ef.ExpertFiller(ef.gguf_expert_source(feed_), ss.EXPERT_GROUP_SIZE)
        say("shards", f"feed over {args.shards} in {time.time() - t0:.1f}s; "
                      f"REAL WEIGHTS for every layer built")
    t0 = time.time()
    if args.artifact:
        # the served binary's own bytes: read, not built. The report is
        # composed from the model and the artifact's manifest so every later
        # stage reads the same keys.
        import json
        core0 = ov.Core()
        xml = str(Path(args.artifact) / "openvino_language_model.xml")
        try:
            model = core0.read_model(xml)
        except Exception as exc:                                  # noqa: BLE001
            say("build", "READ FAIL " + one_line(exc))
            arena.close()
            return 0
        man = {}
        try:
            man = json.loads((Path(args.artifact) / "serving-shape.json").read_text())
        except Exception:                                         # noqa: BLE001
            pass
        tports = [p for p in model.get_parameters()
                  if p.get_friendly_name().startswith("ngram_table.")]
        rep = {
            "n_layers": man.get("layers", args.layers), "gdn_layers": man.get("gdn_layers", -1),
            "attn_layers": man.get("attn_layers", -1), "nodes": len(model.get_ordered_ops()),
            "graph_const_bytes": 0,
            "ngram_table_rows": man.get("ngram_table_rows", sum(dims(p.output(0))[0] for p in tports)),
            "ngram_row_bytes": man.get("ngram_row_bytes", dims(tports[0].output(0))[1] if tports else 0),
            "ngram_chunk_cap_bytes": man.get("ngram_chunk_cap_bytes", 0),
            "ngram_table_ports": [(p.get_friendly_name(), dims(p.output(0))[0],
                                   dims(p.output(0))[0] * dims(p.output(0))[1]) for p in tports],
            "inputs": [(p.get_node().get_friendly_name(), dims(p), str(p.get_element_type()))
                       for p in model.inputs],
            "outputs": [(r.get_node().get_friendly_name(), dims(r), str(r.get_element_type()))
                        for r in model.outputs],
        }
        say("build", f"ARTIFACT {xml} read in {time.time() - t0:.2f}s (tree "
                     f"{man.get('tree', '?')}, written {man.get('written_utc', '?')}); "
                     f"the served binary's bytes, not a build")
    else:
        try:
            model, rep = ss.build_serving_shape_ir(arena=arena, n_layers=args.layers,
                                                   filler=filler, feed=feed_)
        except Exception as exc:                                  # noqa: BLE001
            say("build", "FAIL " + one_line(exc))
            arena.close()
            return 0
    if feed_ is not None and not args.artifact:
        dense = rep["dense_fill_census"]
        say("shards", f"dense fill: {len(dense)} tensors, "
                      f"{sum(b for _, b in dense) / 2 ** 30:.2f} GiB written; "
                      f"expert fill: {rep['fill_census']}; arena written "
                      f"{rep['arena_written_bytes'] / 2 ** 30:.2f} GiB")
    say("build", f"OK {time.time() - t0:.2f}s nodes={rep['nodes']} "
                 f"declared_GiB={rep['graph_const_bytes'] / 2 ** 30:.2f} "
                 f"layers={rep['n_layers']} ({rep['gdn_layers']} GDN + "
                 f"{rep['attn_layers']} attn) peak_host_GiB={peak_rss_gib():.2f}")
    for name, shape, et in rep["inputs"]:
        say("build", f"input  {name:16s} {shape} {et}")
    say("build", f"ngram table: {rep['ngram_table_rows']:,} rows x "
                 f"{rep['ngram_row_bytes']} B over {len(rep['ngram_table_ports'])} "
                 f"port(s) under cap {rep['ngram_chunk_cap_bytes']:,}: "
                 + ", ".join(f"{n}[{r:,}]={b:,}B" for n, r, b in rep["ngram_table_ports"]))

    # ---- protos (backend_ov.cpp:2565-2577) -----------------------------------
    conv_proto, gdn_proto = [], []
    for var in model.get_variables():
        ps = var.get_info().data_shape
        if ps.rank.is_dynamic:
            continue
        tail = [d for d in list(ps)[1:]]
        if not all(d.is_static for d in tail):
            continue                                  # attention KV: dynamic seq dim
        sh = [1] + [d.get_length() for d in tail]
        (conv_proto if len(sh) == 3 else gdn_proto if len(sh) == 4 else []).append(sh)
    say("protos", f"conv={conv_proto[:1]}x{len(conv_proto)} "
                  f"gdn={gdn_proto[:1]}x{len(gdn_proto)} "
                  f"variables={len(model.get_variables())} sinks={len(model.get_sinks())}")

    # ---- pass (backend_ov.cpp:2582) ------------------------------------------
    if not args.no_pass:
        from openvino._offline_transformations import (
            paged_attention_transformation as pat)
        try:
            pat(model)
        except Exception as exc:                                  # noqa: BLE001
            say("pass", "REFUSED " + one_line(exc))
            arena.close()
            return 0
        say("pass", "OK ports after: " + ", ".join(
            f"{p.get_any_name()}{dims(p)}" for p in model.inputs))
        hist = {}
        for node in model.get_ordered_ops():
            tn = node.get_type_name()
            if tn.startswith("Paged"):
                hist[tn] = hist.get(tn, 0) + 1
        say("pass", f"paged ops {hist}")
    if args.cut:
        hits = [n for n in model.get_ordered_ops() if n.get_friendly_name() == args.cut]
        if len(hits) != 1:
            say("cut", f"{args.cut!r} names {len(hits)} node(s); nothing cut")
            arena.close()
            return 0
        res = ov.opset13.result(hits[0].output(0))
        model = ov.Model([res], model.get_parameters(), f"cut_at_{args.cut}")
        say("cut", f"LOCALISER: graph cut after {args.cut!r} "
                   f"({hits[0].get_type_name()}, {dims(hits[0].output(0))}); "
                   f"{len(model.get_ordered_ops())} ops remain; NOT the served path")
    if args.stage == "pass" or args.device is None:
        arena.close()
        return 0

    # ---- compile ----------------------------------------------------------------
    core = ov.Core()
    dev = args.device
    if args.no_pass:
        props = {"INFERENCE_PRECISION_HINT": "f32"} if dev.startswith("GPU") else {}
    else:
        props = {"KV_CACHE_PRECISION": getattr(ov.Type, args.paged_kv)}
        if args.precision:
            props["INFERENCE_PRECISION_HINT"] = args.precision
    t0 = time.time()
    try:
        compiled = core.compile_model(model, dev, props)
    except Exception as exc:                                      # noqa: BLE001
        say("compile", f"FAIL after {time.time() - t0:.2f}s "
                       f"peak_host_GiB={peak_rss_gib():.2f} " + one_line(exc))
        arena.close()
        return 0
    prec = "?"
    try:
        prec = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception:                                             # noqa: BLE001
        pass
    resident = "n/a"
    if dev.startswith("GPU"):
        try:
            st = dict(core.get_property(dev, "GPU_MEMORY_STATISTICS"))
            resident = f"{sum(v for k, v in st.items() if k in ('usm_device', 'cl_mem')) / 2 ** 30:.2f}"
        except Exception:                                         # noqa: BLE001
            pass
    say("compile", f"OK {time.time() - t0:.2f}s prec={prec} "
                   f"peak_host_GiB={peak_rss_gib():.2f} device_resident_GiB={resident}")
    say("compile", "ports: " + ", ".join(
        f"{p.get_any_name()}{dims(p)}:{p.get_element_type().get_type_name()}"
        for p in compiled.inputs))
    if args.stage == "compile":
        arena.close()
        return 0

    # ---- request ----------------------------------------------------------------
    req = compiled.create_infer_request()
    declared = {p.get_any_name(): p for p in compiled.inputs}
    n = T
    fed = []

    def feed(name, tensor):
        """set_tensor exactly as the served path does: unconditional, and a
        missing name THROWS. The throw is captured and returned, not hidden."""
        try:
            req.set_tensor(name, tensor)
        except Exception as exc:                                  # noqa: BLE001
            say("forward", f"set_tensor({name}) THREW " + one_line(exc))
            return exc
        fed.append(name)
        return None

    def gpu_mem(label):
        """GPU_MEMORY_STATISTICS by allocation type, GiB, so a table that the
        plugin silently copied to the device shows up as usm_device growth."""
        if not dev.startswith("GPU"):
            return
        try:
            st = dict(core.get_property(dev, "GPU_MEMORY_STATISTICS"))
            say("gpu-mem", f"{label}: " + " ".join(
                f"{k}={v / 2 ** 30:.2f}GiB" for k, v in sorted(st.items()) if v))
        except Exception as exc:                                  # noqa: BLE001
            say("gpu-mem", f"{label}: unavailable ({one_line(exc)})")

    # ---- the n-gram table: bound ONCE per request, from host memory ------------
    # Increment 5. The table is `ngram_table.K` ports, one per chunk under the
    # A770's per-object cap. On a GPU each chunk is a USM-host tensor from the
    # device's own context: the plugin shares such a tensor with the graph
    # without copying it to the device (sync_infer_request.cpp `prepare_input`,
    # `is_usm_host_tensor && !convert_needed`), which is what "host-mmap tier"
    # means for a compiled graph. The rows are whatever the allocation holds --
    # unwritten, like every weight here. `gpu-mem` lines before and after say
    # where the bytes went.
    table_ports = sorted((nm for nm in declared if nm.startswith("ngram_table.")),
                         key=lambda nm: int(nm.split(".")[1]))
    if table_ports:
        gpu_mem("before table")
        tctx = core.get_default_context(dev) if dev.startswith("GPU") else None
        t0 = time.time()
        total = 0
        raw = None
        if feed_ is not None:
            # the GGUF's own bytes: (rows, 90) u8, mmapped, copied chunk by
            # chunk into the USM-host tensors -- no conversion, one copy
            raw = np.asarray(feed_.raw_table())
            say("table", f"binding the REAL table: {raw.shape} {raw.dtype} from the shards")
        off = 0
        for name in table_ports:
            sh = dims(declared[name])
            et = declared[name].get_element_type()
            try:
                t = (tctx.create_host_tensor(et, ov.Shape(sh)) if tctx
                     else ov.Tensor(et, ov.Shape(sh)))
            except Exception as exc:                              # noqa: BLE001
                say("table", f"{name}{sh}: ALLOC FAIL " + one_line(exc))
                arena.close()
                return 0
            if raw is not None:
                t1 = time.time()
                np.copyto(t.data, raw[off:off + sh[0]])
                say("table", f"{name}: rows {off:,}..{off + sh[0]:,} copied in "
                             f"{time.time() - t1:.1f}s")
                off += sh[0]
            total += int(np.prod(sh))
            if feed(name, t) is not None:
                arena.close()
                return 0
        say("table", f"bound {len(table_ports)} port(s), {total:,} B "
                     f"({total / 2 ** 30:.2f} GiB) of "
                     f"{'USM host' if tctx else 'host'} memory in "
                     f"{time.time() - t0:.2f}s peak_host_GiB={peak_rss_gib():.2f}")
        gpu_mem("after table")

    if not args.no_pass:
        ctx = core.get_default_context(dev) if dev.startswith("GPU") else None
        la_i = kv_i = 0
        nblk = (n + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS + 1
        for name, port in declared.items():
            if name.startswith("conv_state_table."):
                sh = list(conv_proto[la_i % max(len(conv_proto), 1)]) if conv_proto else None
            elif name.startswith("gated_delta_state_table."):
                sh = list(gdn_proto[la_i % max(len(gdn_proto), 1)]) if gdn_proto else None
            elif name.startswith(("key_cache.", "value_cache.")):
                d = dims(port)
                sh = [nblk] + d[1:]
                t = (ctx.create_tensor(port.get_element_type(), ov.Shape(sh), {})
                     if ctx else ov.Tensor(port.get_element_type(), ov.Shape(sh)))
                feed(name, t)
                kv_i += 1
                continue
            else:
                continue
            if sh is None:
                say("request", f"{name}: no prototype on the stateful graph")
                continue
            sh[0] = ROWS_PER_LANE
            t = (ctx.create_tensor(ov.Type.f16, ov.Shape(sh), {}) if ctx
                 else ov.Tensor(ov.Type.f16, ov.Shape(sh)))
            feed(name, t)
            la_i += 1
        say("request", f"state rows bound: {la_i} (f16, rows={ROWS_PER_LANE}); "
                       f"KV pools: {kv_i} (blocks={nblk})")

    # ---- forward ----------------------------------------------------------------
    H = 2560
    rep_vocab = lambda: rep["outputs"][0][1][-1]                            # noqa: E731
    i64 = lambda a, sh: ov.Tensor(np.array(a, dtype=np.int64).reshape(sh))   # noqa: E731
    i32 = lambda a: ov.Tensor(np.array(a, dtype=np.int32).reshape(-1))       # noqa: E731

    def flat_or_2d(name, values):
        """The pass rewrites `input_ids` and `position_ids` to rank 1
        (`sdpa_to_paged_attention.cpp`: `set_partial_shape({-1})` + an
        Unsqueeze the graph owns), so post-pass they take the flat token
        vector; pre-pass (the control) they are the emitter's [1, T]. Follow
        the declared rank rather than assume either."""
        d = dims(declared[name]) if name in declared else None
        m = len(values)
        return i64(values, (m,) if d is not None and len(d) == 1 else (1, m))

    # the prompt's embedding, on the host as the served path does it
    # every row this run will need is cut out now and the 2.5 GB f32 table
    # dropped: the host budget is the table's 26.8 GiB of pinned USM plus the
    # compile peak, and a second copy of token_embd is not in it
    _emb_rows = {}
    if feed_ is not None:
        t0 = time.time()
        embed_w = feed_.fitted("embed_tokens.weight", (rep_vocab(), H))
        say("shards", f"embed_tokens {embed_w.shape} dequantised in {time.time() - t0:.1f}s")
        need = set(ids)
        if args.long:
            need |= set(range(1000, 1000 + int(args.long)))
        for tid in need:
            _emb_rows[int(tid)] = np.array(embed_w[int(tid)], np.float32)
        del embed_w

    def embeds_for(tokens):
        if feed_ is None:
            return np.zeros((len(tokens), H), np.float32)
        out = np.empty((len(tokens), H), np.float32)
        for j, tid in enumerate(tokens):
            out[j] = _emb_rows.get(int(tid), 0.0)   # a token no leg asked for: zero, said so
            if int(tid) not in _emb_rows:
                say("shards", f"token {tid} had no cut row; fed zero")
        return out

    # THE DRIVER'S OWN FEEDS (feed-the-ports): the hashed row ids, split at
    # the port partition by the host, and the padding mask. Computed once for
    # the prompt; the runtime has no site for either yet (that is the C++
    # half of the increment), so these are labelled as the driver's.
    from q4e import ngram_ids as nid
    from q4e import piecewise_export as pwe
    cfg = pwe.real_config()
    table_rows0 = (dims(declared[table_ports[0]])[0] if table_ports else 1)

    ple_ord = nid.ple_ordinal(cfg, 1)                          # decoder layer 1 -> ordinal 0

    def id_feeds(tokens):
        g = nid.gen_row_ids(cfg, ple_ord, tokens)              # [1,T,Hn] i64
        c, l = nid.split_by_partition(g, table_rows0)
        return {"ngram_chunk_ids": ov.Tensor(c), "ngram_local_ids": ov.Tensor(l),
                "conv_mask": ov.Tensor(np.ones((1, len(tokens)), np.float32))}, g

    if args.no_pass:
        # CONTROL: every port the stateful graph declares, fed by name.
        extras, _ = id_feeds(ids)
        candidates = {
            "inputs_embeds": ov.Tensor(embeds_for(ids).reshape(1, n, H)),
            "position_ids": flat_or_2d("position_ids", list(range(n))),
            "attention_mask": i64([1] * n, (1, n)),
            "beam_idx": i32([0]),
            **extras,
        }
        for name, t in candidates.items():
            if name in declared:
                feed(name, t)
            else:
                say("forward", f"{name}: not declared by the compiled model")
    else:
        # THE SERVED PATH, in the C++'s own order (backend_ov.cpp:6153-6163).
        # the served C++ feeds position_ids at the port's own rank: the
        # mrope artifact's [sections, n]; this IR's post-pass [n]
        past, tot = 0, n
        pd = dims(declared["position_ids"]) if "position_ids" in declared else None
        if pd is not None and len(pd) == 1:
            pos = i64(list(range(past, tot)), (n,))
        else:
            sections = pd[0] if pd and pd[0] > 0 else 1
            pos = i64([p for _ in range(sections) for p in range(past, tot)],
                      (sections, n))
        served = [
            ("inputs_embeds", ov.Tensor(embeds_for(ids))),
            ("position_ids", pos),
            ("past_lens", i32([past])),
            ("subsequence_begins", i32([0, n])),
            ("block_indices", i32(list(range((tot + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS)))),
            ("block_indices_begins", i32([0, (tot + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS])),
            ("max_context_len", ov.Tensor(np.array(tot, dtype=np.int32))),
            ("la.block_indices", i32([0, 0])),
            ("la.block_indices_begins", i32([0, 2])),
            ("la.past_lens", i32([past])),
            ("la.cache_interval", i32([0])),
        ]
        first = None
        for name, t in served:
            exc = feed(name, t)
            if exc is not None and first is None:
                first = (name, exc)
        if first is not None:
            say("forward", f"SERVED PATH VERDICT: first refusal at set_tensor("
                           f"{first[0]}): {one_line(first[1])}")
            if not args.probe:
                arena.close()
                return 0
            say("probe", "LABELLED DEVIATION: continuing past the refusal")
        else:
            say("forward", "SERVED PATH: every name the C++ feeds was accepted")
        extras, g = id_feeds(ids)
        say("driver-feed", f"row ids for the prompt (q4e.ngram_ids, PLE at decoder "
                           f"layer 1 = ordinal {ple_ord}): "
                           f"min {int(g.min()):,} max {int(g.max()):,}; split at "
                           f"{table_rows0:,} rows/chunk; plus conv_mask=ones. "
                           f"THE DRIVER'S FEEDS, not the runtime's")
        for name, t in extras.items():
            feed(name, t)

    unfed = sorted(set(declared) - set(fed))
    say("forward", f"fed {len(fed)}: {fed}")
    say("forward", f"declared, never fed {len(unfed)}: {unfed}")
    t0 = time.time()
    try:
        req.infer()
    except Exception as exc:                                      # noqa: BLE001
        say("forward", f"INFER FAIL after {time.time() - t0:.3f}s " + one_line(exc))
        arena.close()
        return 0
    dt = time.time() - t0
    lg = (req.get_output_tensor(0).data if args.cut
          else req.get_tensor("logits").data)
    rows = lg.reshape(-1, lg.shape[-1])
    say("forward", f"INFER OK {dt:.3f}s out{tuple(lg.shape)} "
                   f"finite={bool(np.isfinite(lg).all())} "
                   f"absmax={float(np.abs(lg).max()):.4e}")
    say("forward", f"argmax per position: {[int(r.argmax()) for r in rows]}")
    say("forward", f"RAW OUTPUT (greedy, last position): {int(rows[-1].argmax())}")
    if feed_ is not None:
        # the token strings, from the GGUF's own vocabulary, so the id has a face
        toks = feed_.token_strings()
        top = np.argsort(-rows[-1])[:5]
        say("forward", "REAL-WEIGHT LOGITS, last position, top 5: " + ", ".join(
            f"{int(i)}={toks[int(i)]!r}:{float(rows[-1][int(i)]):.3f}" for i in top))
        say("forward", f"prompt tokens: {[toks[int(i)] for i in ids]}")
    gpu_mem("after infer")

    # ---- REPEATS: the same forward again, compared bit for bit ----------------
    # The first forward pays the kernel jit (feedback-first-request-compiles-
    # kernels); every later one is the rate AND the determinism reading. On
    # the served path the request is re-used (the runtime keeps one per lane);
    # --fresh binds a new request per repeat instead, so a difference that
    # appears only in one of the two modes names the request's own state.
    base = np.array(rows, dtype=np.float32, copy=True)
    served_feeds = {name: req.get_tensor(name) for name in fed}

    def fresh_request():
        r2 = compiled.create_infer_request()
        ctx2 = core.get_default_context(dev) if dev.startswith("GPU") else None
        for name, portp in declared.items():
            if name.startswith("conv_state_table."):
                sh = list(conv_proto[0]); sh[0] = ROWS_PER_LANE
                et2 = ov.Type.f16
            elif name.startswith("gated_delta_state_table."):
                sh = list(gdn_proto[0]); sh[0] = ROWS_PER_LANE
                et2 = ov.Type.f16
            elif name.startswith(("key_cache.", "value_cache.")):
                sh = list(served_feeds[name].get_shape())
                et2 = portp.get_element_type()
            else:
                if name in served_feeds:
                    r2.set_tensor(name, served_feeds[name])   # table ports, feeds: shared
                continue
            r2.set_tensor(name, ctx2.create_tensor(et2, ov.Shape(sh), {})
                          if ctx2 else ov.Tensor(et2, ov.Shape(sh)))
        return r2

    for k in range(2, int(args.repeat) + 1):
        rk = fresh_request() if args.fresh else req
        t0 = time.time()
        try:
            rk.infer()
        except Exception as exc:                                  # noqa: BLE001
            say("repeat", f"#{k} INFER FAIL after {time.time() - t0:.3f}s " + one_line(exc))
            break
        dt = time.time() - t0
        outk = rk.get_output_tensor(0).data if args.cut else rk.get_tensor("logits").data
        cur = np.array(outk, dtype=np.float32).reshape(-1, outk.shape[-1])
        if cur.shape != base.shape:
            say("repeat", f"#{k} shape {cur.shape} != #1 {base.shape}")
            break
        diff_rows = np.flatnonzero((cur != base).any(axis=1))
        if diff_rows.size == 0:
            say("repeat", f"#{k} ({'fresh' if args.fresh else 'same'} request) "
                          f"INFER OK {dt:.3f}s: BIT-IDENTICAL to #1 over "
                          f"{base.shape[0]} rows x {base.shape[1]}")
            continue
        moved = int((cur.argmax(axis=1) != base.argmax(axis=1)).sum())
        say("repeat", f"#{k} ({'fresh' if args.fresh else 'same'} request) "
                      f"INFER OK {dt:.3f}s: DIFFERS from #1 -- first row "
                      f"{int(diff_rows[0])}, {diff_rows.size} of {base.shape[0]} "
                      f"rows differ, max |diff| {float(np.abs(cur - base).max()):.4e}, "
                      f"max |#1| {float(np.abs(base).max()):.4e}, argmax moved "
                      f"in {moved} rows")

    # ---- the static-T probe: what a DECODE step would meet -----------------------
    # A decode step feeds one token. The query block is static in T inside the
    # graph even where the PORT is dynamic ([?] after the pass): this measures
    # what the runtime says to a one-token block, verbatim -- a set_tensor
    # refusal if the port is static, or an infer refusal at the first baked
    # reshape if it is not -- instead of predicting the text.
    port = "inputs_embeds" if "inputs_embeds" in declared else None
    if port is not None and n != 1:
        one = ov.Tensor(embeds_for([int(rows[-1].argmax())]))
        # the one token's companions: position, ids, mask, index ports
        nxt = [int(rows[-1].argmax())]
        feed("position_ids", flat_or_2d("position_ids", [n]))
        for name, t in id_feeds(nxt)[0].items():
            if name in declared:
                feed(name, t)
        if not args.no_pass:
            past1, tot1 = n, n + 1
            nb1 = (tot1 + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS
            for name, t in [("past_lens", i32([past1])),
                            ("subsequence_begins", i32([0, 1])),
                            ("block_indices", i32(list(range(nb1)))),
                            ("block_indices_begins", i32([0, nb1])),
                            ("max_context_len", ov.Tensor(np.array(tot1, dtype=np.int32))),
                            ("la.past_lens", i32([past1]))]:
                feed(name, t)
        exc = feed(port, one)
        if exc is None:
            say("decode-probe", f"set_tensor({port}) with a 1-token block was "
                                f"ACCEPTED against the {n}-token query block; "
                                f"infer:")
            t0 = time.time()
            try:
                req.infer()
                lg1 = req.get_output_tensor(0).data if args.cut else req.get_tensor("logits").data
                say("decode-probe", f"  INFER OK {time.time() - t0:.3f}s out{tuple(lg1.shape)} "
                                    f"finite={bool(np.isfinite(lg1).all())} -- a "
                                    f"1-token block after a {n}-token one: the "
                                    f"graph is dynamic in T")
            except Exception as exc2:                             # noqa: BLE001
                say("decode-probe", f"  INFER FAIL after {time.time() - t0:.3f}s "
                                    + one_line(exc2))
        else:
            say("decode-probe", f"a 1-token block against the {n}-token query "
                                f"block: refused (text above)")

    # ---- the LONG block: one forward of --long tokens from position 0 ------------
    if args.long and port is not None and not args.no_pass:
        L = int(args.long)
        toks = list(range(1000, 1000 + L))
        nbl = (L + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS
        # a fresh request so the KV pools are sized for L blocks
        req2 = compiled.create_infer_request()
        for name in table_ports:
            req2.set_tensor(name, req.get_tensor(name))
        la_i = 0
        for name, portp in declared.items():
            if name.startswith("conv_state_table."):
                sh = list(conv_proto[0]); sh[0] = ROWS_PER_LANE
            elif name.startswith("gated_delta_state_table."):
                sh = list(gdn_proto[0]); sh[0] = ROWS_PER_LANE
            elif name.startswith(("key_cache.", "value_cache.")):
                d = dims(portp); sh = [nbl + 1] + d[1:]
                req2.set_tensor(name, ctx.create_tensor(portp.get_element_type(), ov.Shape(sh), {})
                                if ctx else ov.Tensor(portp.get_element_type(), ov.Shape(sh)))
                continue
            else:
                continue
            req2.set_tensor(name, ctx.create_tensor(ov.Type.f16, ov.Shape(sh), {})
                            if ctx else ov.Tensor(ov.Type.f16, ov.Shape(sh)))
        req2.set_tensor("inputs_embeds", ov.Tensor(embeds_for(toks)))
        req2.set_tensor("position_ids", i64(list(range(L)), (L,)))
        for name, t in [("past_lens", i32([0])), ("subsequence_begins", i32([0, L])),
                        ("block_indices", i32(list(range(nbl)))),
                        ("block_indices_begins", i32([0, nbl])),
                        ("max_context_len", ov.Tensor(np.array(L, dtype=np.int32))),
                        ("la.block_indices", i32([0, 0])), ("la.block_indices_begins", i32([0, 2])),
                        ("la.past_lens", i32([0])), ("la.cache_interval", i32([0]))]:
            req2.set_tensor(name, t)
        for name, t in id_feeds(toks)[0].items():
            if name in declared:
                req2.set_tensor(name, t)
        t0 = time.time()
        try:
            req2.infer()
            lgL = req2.get_tensor("logits").data
            say("long-block", f"INFER OK {time.time() - t0:.3f}s for {L} tokens "
                              f"out{tuple(lgL.shape)} finite={bool(np.isfinite(lgL).all())} "
                              f"absmax={float(np.abs(lgL).max()):.4e} "
                              f"({'past' if L > 2051 else 'under'} the 2051 QSA boundary; "
                              f"rope positions up to {L - 1})")
        except Exception as exc:                                  # noqa: BLE001
            say("long-block", f"INFER FAIL after {time.time() - t0:.3f}s for {L} tokens "
                              + one_line(exc))
        gpu_mem("after long block")
    arena.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
