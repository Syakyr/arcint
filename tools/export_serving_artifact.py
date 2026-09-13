#!/usr/bin/env python3
"""FULL-DEPTH increment: the serving-shape IR as an ARTIFACT DIRECTORY the
served binary loads (`src/core/artifact.cpp` `load_artifact`), so the runtime's
own binding site (`bind_ngram_ports` / `feed_ngram_ports` in backend_ov.cpp)
runs on a card for the first time. Until this tool existed the only thing that
had ever fed the serving-shape IR on a card was the labelled probe
(`tools/boot_serving_shape.py`), and the France/KLD gates are SERVED-PATH
gates (window-050 §7, §8): they cannot be reached through a probe.

What lands in `<out>` (every name `load_artifact` requires, and nothing it
would silently ignore):

    openvino_language_model.xml/.bin   build_serving_shape_ir(feed=GgufFeed)
                                       at depth --layers, REAL weights, saved
                                       with compress_to_fp16=False (the bytes
                                       the probe ran are the bytes served)
    openvino_text_embeddings_model.*   token_embd.weight, dequantised f32
                                       [V, H], Gather by input_ids [1, T],
                                       T dynamic (the served `embed_paged`
                                       feeds [1, n] for any n)
    openvino_tokenizer.* / openvino_detokenizer.* / tokenizer.json /
    tokenizer_config.json              PASSTHROUGH from --tokenizer-from,
                                       admitted only after this tool has
                                       compared every id of tokenizer.json's
                                       vocab (+ added tokens) against the
                                       GGUF's own tokenizer.ggml.tokens --
                                       a mismatch refuses by count
    chat_template.jinja                the GGUF's tokenizer.chat_template
    config.json                        the real geometry (q4e.piecewise_export
                                       REAL_GEOMETRY) at num_hidden_layers =
                                       --layers, with the five n-gram keys
                                       artifact.cpp reads (FIX D) and the
                                       hash-boundary eos the GGUF declares
                                       under `qwen4exp.ple.eos_token_id` --
                                       the same key llama.cpp (the KLD
                                       reference) hashes with
    generation_config.json             eos_token_id [boundary, im_end]: the
                                       loader takes eos_ids.front() as the
                                       n-gram hash boundary, so the PLE eos
                                       is FIRST; both stop generation
    serving-shape.json                 the manifest: tree, depth, ports,
                                       fill census, shard identity, sha256 of
                                       the IR files -- what the allowlist
                                       entry pins

The directory's BASENAME is the allowlist alias (`models/allowlist-raw.json`,
`src/core/model_registry.cpp`); this tool does not consult the allowlist --
the served binary does, and refuses an unlisted name by name.

Nothing here is a test. One artifact per process:

    <venv>/bin/python tools/export_serving_artifact.py --layers 4 \\
        --shards /path/to/shards --tokenizer-from /path/to/qwen38-artifact \\
        --out /path/to/models/qwen38-flash-next-d4-ov --tree <sha>
"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

MODEL_TYPE = "qwen4_exp"
ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
# The GGUF's own hash-boundary key (llama.cpp gguf-py constants.py
# `{arch}.ple.eos_token_id`; llama.cpp's PLE hashes with it, so does the pin
# through config.eos_token_id). Read, never assumed: tokenizer.ggml.eos_token_id
# is a DIFFERENT token in this file (im_end), and feeding that one as the
# boundary would hash every first-position row wrongly.
PLE_EOS_KEY = "qwen4exp.ple.eos_token_id"
GENERATION_EOS_KEY = "tokenizer.ggml.eos_token_id"
CHAT_TEMPLATE_KEY = "tokenizer.chat_template"

TOKENIZER_PASSTHROUGH = (
    "openvino_tokenizer.xml", "openvino_tokenizer.bin",
    "openvino_detokenizer.xml", "openvino_detokenizer.bin",
    "tokenizer.json", "tokenizer_config.json",
)


def say(stage, text):
    print(f"EXPORT [{stage}] {text}", flush=True)


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def gguf_field(reader, key):
    f = reader.fields.get(key)
    if f is None:
        return None
    return f.contents()


def gguf_tokens(reader):
    f = reader.fields.get("tokenizer.ggml.tokens")
    if f is None:
        raise KeyError("tokenizer.ggml.tokens not in the shard")
    return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]


def vocab_mismatches(tokenizer_json, gguf_token_list):
    """Every (id, string) tokenizer.json defines -- the BPE vocab and the added
    tokens -- against the GGUF's token list at the same id. Returns the list of
    (id, tokenizer.json string, gguf string) that disagree, plus the count of
    ids tokenizer.json defines at all. Ids the GGUF has and tokenizer.json does
    not (the padded tail) are not a mismatch: the served tokenizer never
    produces them."""
    hf = json.loads(Path(tokenizer_json).read_text())
    defined = {}
    for s, i in hf["model"]["vocab"].items():
        defined[int(i)] = s
    for a in hf.get("added_tokens", []):
        defined[int(a["id"])] = a["content"]
    bad = []
    for i, s in defined.items():
        g = gguf_token_list[i] if i < len(gguf_token_list) else None
        if g != s:
            bad.append((i, s, g))
    return bad, len(defined)


def serving_config(n_layers, ple_eos_token_id, geometry=None):
    """config.json as the loader reads it (artifact.cpp: text_config or top
    level; num_hidden_layers, hidden_size, max_position_embeddings,
    num_experts, full_attention_interval, layer_types, the n-gram five,
    vocab_size, eos_token_id) plus the pin's own keys so the file is the
    real checkpoint's config at depth `n_layers` and not a stub."""
    from q4e import piecewise_export as pwe
    g = dict(geometry if geometry is not None else pwe.REAL_GEOMETRY)
    nl = int(n_layers)
    if nl < 1 or nl > g["num_hidden_layers"]:
        raise ValueError(f"--layers {nl} outside 1..{g['num_hidden_layers']}")
    # the same rule build_serving_shape_ir applies: layer i%4==3 is attention
    layer_types = ["qwen_sparse_attention" if i % 4 == 3 else "linear_attention"
                   for i in range(nl)]
    ple_ids = [i for i in g["ple_layer_ids"] if i - 1 < nl]   # 1-based ids
    if not ple_ids:
        raise ValueError(f"no PLE layer within {nl} layers (ple_layer_ids "
                         f"{g['ple_layer_ids']}); the IR carries one")
    cfg = {
        "architectures": [ARCHITECTURE],
        "model_type": MODEL_TYPE,
        "vocab_size": g["vocab_size"],
        "hidden_size": g["hidden_size"],
        "num_hidden_layers": nl,
        "num_attention_heads": g["num_attention_heads"],
        "num_key_value_heads": g["num_key_value_heads"],
        "head_dim": g["head_dim"],
        "max_position_embeddings": g["max_position_embeddings"],
        "rms_norm_eps": g["rms_norm_eps"],
        "linear_key_head_dim": g["linear_key_head_dim"],
        "linear_num_key_heads": g["linear_num_key_heads"],
        "linear_value_head_dim": g["linear_value_head_dim"],
        "linear_num_value_heads": g["linear_num_value_heads"],
        "linear_conv_kernel_dim": g["linear_conv_kernel_dim"],
        "num_experts": g["num_experts"],
        "num_experts_per_tok": g["num_experts_per_tok"],
        "moe_intermediate_size": g["moe_intermediate_size"],
        "shared_expert_intermediate_size": g["shared_expert_intermediate_size"],
        "hc_count": g["hc_count"],
        "hc_lowrank": g["hc_lowrank"],
        "full_attention_interval": 4,
        "layer_types": layer_types,
        # the n-gram table declaration artifact.cpp reads (FIX D)
        "ngram_size": g["ngram_size"],
        "heads_per_ngram": g["heads_per_ngram"],
        "ngram_vocab_size_base": g["ngram_vocab_size_base"],
        "make_ngram_vocab_size_divisible_by": g["make_ngram_vocab_size_divisible_by"],
        "ple_embed_dim": g["ple_embed_dim"],
        "ple_conv_kernel_size": g["ple_conv_kernel_size"],
        "ple_layer_ids": ple_ids,
        "eos_token_id": int(ple_eos_token_id),
        "indexer_n_heads": g["indexer_n_heads"],
        "indexer_kv_heads": g["indexer_kv_heads"],
        "indexer_head_dim": g["indexer_head_dim"],
        "indexer_budget": g["indexer_budget"],
        "indexer_compress_ratio": g["indexer_compress_ratio"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": g["rope_theta"],
            "partial_rotary_factor": g["partial_rotary_factor"],
            "mrope_section": g["mrope_section"],
        },
        "tie_word_embeddings": g["tie_word_embeddings"],
        "torch_dtype": "bfloat16",
    }
    return cfg


def build_embed_model(table_f32):
    """token_embd as its own model, T DYNAMIC: input_ids [1, T] i64 ->
    [1, T, H] f32. `pwe.build_embed_piece` is static in T; the served
    `embed_paged` feeds [1, n] for every n a chunk or a decode step has."""
    import openvino as ov
    from openvino import opset13 as op
    tbl = op.constant(np.ascontiguousarray(table_f32, dtype=np.float32))
    tbl.set_friendly_name("embed_tokens.weight")
    ids = op.parameter([1, -1], ov.Type.i64)
    ids.set_friendly_name("input_ids")
    ids.output(0).set_names({"input_ids"})
    out = op.gather(tbl, ids, op.constant(np.int64(0)))
    res = op.result(out)
    res.set_friendly_name("output")
    return ov.Model([res], [ids], "qwen4_exp_embed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--shards", required=True,
                    help="directory (or glob) of the model's GGUF shards")
    ap.add_argument("--tokenizer-from", required=True,
                    help="an artifact directory whose tokenizer files pass "
                         "through (admitted after the vocab comparison)")
    ap.add_argument("--out", required=True, help="the artifact directory to write")
    ap.add_argument("--tree", default="unrecorded",
                    help="the commit the emitter code came from (CF-MANIFESTSHA)")
    ap.add_argument("--arena", default=None,
                    help="path for the build's sparse arena file (default: "
                         "<out>/.arena.bin; the file is removed after the save "
                         "unless --keep-arena)")
    ap.add_argument("--keep-arena", action="store_true")
    ap.add_argument("--skip-hash", action="store_true",
                    help="do not sha256 the written IR files (the manifest "
                         "then says so)")
    args = ap.parse_args(argv)

    import openvino as ov
    from q4e import serving_shape as ss
    from q4e import gguf_feed as gf
    from q4e import expert_fill as ef
    from q4e import piecewise_export as pwe

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    say("env", f"openvino {ov.get_version()} layers={args.layers} out={out} "
               f"tree={args.tree}")

    # ---- the shards, the vocab comparison, the template, the boundary ------
    t0 = time.time()
    feed = gf.GgufFeed(args.shards)
    filler = ef.ExpertFiller(ef.gguf_expert_source(feed), ss.EXPERT_GROUP_SIZE)
    reader0 = feed._readers[0]
    ple_eos = gguf_field(reader0, PLE_EOS_KEY)
    gen_eos = gguf_field(reader0, GENERATION_EOS_KEY)
    template = gguf_field(reader0, CHAT_TEMPLATE_KEY)
    if ple_eos is None or gen_eos is None or not template:
        say("shards", f"REFUSED: the first shard lacks one of {PLE_EOS_KEY}="
                      f"{ple_eos!r} {GENERATION_EOS_KEY}={gen_eos!r} "
                      f"{CHAT_TEMPLATE_KEY}={'present' if template else None}")
        return 2
    toks = gguf_tokens(reader0)
    say("shards", f"feed over {args.shards} in {time.time() - t0:.1f}s; arch "
                  f"{feed.arch}; {len(toks):,} tokens; {PLE_EOS_KEY}={ple_eos} "
                  f"({toks[int(ple_eos)]!r}); {GENERATION_EOS_KEY}={gen_eos} "
                  f"({toks[int(gen_eos)]!r}); chat template {len(template)} chars")
    tok_src = Path(args.tokenizer_from)
    for name in TOKENIZER_PASSTHROUGH:
        if not (tok_src / name).is_file():
            say("tokenizer", f"REFUSED: {tok_src / name} missing")
            return 2
    bad, n_defined = vocab_mismatches(tok_src / "tokenizer.json", toks)
    if bad:
        say("tokenizer", f"REFUSED: {len(bad)} of {n_defined} ids in "
                         f"{tok_src / 'tokenizer.json'} disagree with the GGUF's "
                         f"tokens; first: {bad[:3]}")
        return 2
    say("tokenizer", f"{n_defined:,} ids of tokenizer.json agree with the GGUF's "
                     f"tokens, id for id ({len(toks) - n_defined} GGUF ids past "
                     f"the last defined one are the padded tail); passthrough "
                     f"from {tok_src}")

    # ---- the language model: build + fill + save ---------------------------
    arena_path = args.arena or str(out / ".arena.bin")
    arena = ss.SparseArena(path=arena_path)
    t0 = time.time()
    try:
        model, rep = ss.build_serving_shape_ir(arena=arena, n_layers=args.layers,
                                               filler=filler, feed=feed)
    except Exception as exc:                                      # noqa: BLE001
        say("build", f"FAIL {type(exc).__name__}: {exc}")
        arena.close()
        return 1
    dense = rep["dense_fill_census"]
    say("build", f"OK {time.time() - t0:.1f}s nodes={rep['nodes']} layers="
                 f"{rep['n_layers']} ({rep['gdn_layers']} GDN + {rep['attn_layers']} "
                 f"attn) dense fill {len(dense)} tensors "
                 f"{sum(b for _, b in dense) / 2 ** 30:.2f} GiB; expert fill "
                 f"{rep['fill_census']}; arena written "
                 f"{rep['arena_written_bytes'] / 2 ** 30:.2f} GiB; "
                 f"peak_host_GiB={peak_rss_gib():.2f}")
    say("build", f"ngram table: {rep['ngram_table_rows']:,} rows x "
                 f"{rep['ngram_row_bytes']} B over {len(rep['ngram_table_ports'])} "
                 f"port(s) under cap {rep['ngram_chunk_cap_bytes']:,}")
    lm_xml = out / "openvino_language_model.xml"
    t0 = time.time()
    ov.save_model(model, str(lm_xml), compress_to_fp16=False)
    lm_bin = lm_xml.with_suffix(".bin")
    say("save", f"{lm_xml.name} + .bin ({lm_bin.stat().st_size / 2 ** 30:.2f} GiB) "
                f"in {time.time() - t0:.1f}s, compress_to_fp16=False")
    del model
    arena.close()
    if not args.keep_arena and os.path.exists(arena_path):
        os.unlink(arena_path)

    # ---- the embedding model -----------------------------------------------
    t0 = time.time()
    V, H = pwe.REAL_GEOMETRY["vocab_size"], pwe.REAL_GEOMETRY["hidden_size"]
    table = feed.fitted("embed_tokens.weight", (V, H))
    emb = build_embed_model(table)
    del table
    emb_xml = out / "openvino_text_embeddings_model.xml"
    ov.save_model(emb, str(emb_xml), compress_to_fp16=False)
    del emb
    say("embed", f"{emb_xml.name} + .bin ({emb_xml.with_suffix('.bin').stat().st_size / 2 ** 30:.2f} "
                 f"GiB, f32 [{V}, {H}], T dynamic) in {time.time() - t0:.1f}s")

    # ---- passthrough + config + template + generation ---------------------
    for name in TOKENIZER_PASSTHROUGH:
        shutil.copyfile(tok_src / name, out / name)
    (out / "chat_template.jinja").write_text(template)
    cfg = serving_config(args.layers, ple_eos)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    gen = {
        # FIRST is the n-gram hash boundary (artifact.cpp takes eos_ids.front()
        # for it); the second is the chat turn end the served stop logic needs
        "eos_token_id": [int(ple_eos), int(gen_eos)],
        "pad_token_id": int(ple_eos),
        "temperature": 1.0, "top_k": 20, "top_p": 0.95,
    }
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
    say("files", "tokenizer passthrough, chat_template.jinja, config.json "
                 f"(num_hidden_layers {cfg['num_hidden_layers']}, layer_types "
                 f"{cfg['layer_types'].count('qwen_sparse_attention')} attn + "
                 f"{cfg['layer_types'].count('linear_attention')} GDN, ple_layer_ids "
                 f"{cfg['ple_layer_ids']}, eos {cfg['eos_token_id']}), "
                 f"generation_config.json (eos {gen['eos_token_id']})")

    # ---- the manifest ------------------------------------------------------
    hashes = {}
    if not args.skip_hash:
        t0 = time.time()
        for name in ("openvino_language_model.xml", "openvino_language_model.bin",
                     "openvino_text_embeddings_model.xml",
                     "openvino_text_embeddings_model.bin", "tokenizer.json",
                     "chat_template.jinja", "config.json"):
            hashes[name] = sha256_file(out / name)
        say("hash", f"sha256 of 7 files in {time.time() - t0:.1f}s; "
                    f"lm_xml_sha={hashes['openvino_language_model.xml'][:16]} "
                    f"template_sha={hashes['chat_template.jinja'][:16]} "
                    f"tokenizer_sha={hashes['tokenizer.json'][:16]}")
    shards = []
    for p in feed.paths:
        st = os.stat(p)
        shards.append({"name": os.path.basename(p), "bytes": st.st_size,
                       "mtime": int(st.st_mtime)})
    manifest = {
        "tree": args.tree,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "openvino": ov.get_version(),
        "layers": rep["n_layers"], "gdn_layers": rep["gdn_layers"],
        "attn_layers": rep["attn_layers"], "nodes": rep["nodes"],
        "rope_span": rep["rope_span"],
        "ngram_table_rows": rep["ngram_table_rows"],
        "ngram_row_bytes": rep["ngram_row_bytes"],
        "ngram_chunk_cap_bytes": rep["ngram_chunk_cap_bytes"],
        "ngram_table_ports": [list(p) for p in rep["ngram_table_ports"]],
        "inputs": [[n, d, t] for n, d, t in rep["inputs"]],
        "dense_fill_tensors": len(dense),
        "dense_fill_bytes": int(sum(b for _, b in dense)),
        "expert_fill": rep["fill_census"],
        "arena_written_bytes": int(rep["arena_written_bytes"]),
        "lm_bin_bytes": lm_bin.stat().st_size,
        "ple_eos_token_id": int(ple_eos), "generation_eos_token_id": int(gen_eos),
        "tokenizer_from": str(tok_src),
        "tokenizer_ids_compared": n_defined,
        "shards": shards, "arch": feed.arch,
        "sha256": hashes if hashes else "skipped (--skip-hash)",
        "compress_to_fp16": False,
    }
    (out / "serving-shape.json").write_text(json.dumps(manifest, indent=2) + "\n")
    say("done", f"{out} peak_host_GiB={peak_rss_gib():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
