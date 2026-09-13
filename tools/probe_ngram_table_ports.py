#!/usr/bin/env python3
"""Increment 5, the on-card probe for the n-gram table PORTS -- run before the
depth-4 boot so the boot measures the graph and not the transport.

Three things the depth-4 boot cannot tell apart, because every weight there
is an unwritten page and a wrong row gathers the same zeros as the right one:

  1. does a USM-host tensor of a chunk's size share with the graph WITHOUT a
     device copy (the pinned plugin source says it does: sync_infer_request.cpp
     `prepare_input`, `is_usm_host_tensor && !convert_needed`), and does the
     per-object cap refuse the whole table as one USM-host object (the same
     source says it does: engine.cpp `check_allocatable` runs before the
     allocation type is looked at);
  2. does the Gather kernel read the RIGHT row at the top of a chunk that is
     4,294,901,760 B long -- `gather_ref.cl` indexes in `uint`, and the last
     byte of that chunk sits 65,536 B under 2**32;
  3. does the chunk Select pick the right port on the card, across every
     boundary, for ids the host computed.

Sentinel rows are written into the USM-host tensors through `.data` at both
edges of every chunk and at a few interior rows; each row's bytes are a
function of its GLOBAL row index, so a gather that lands one row off, or in
the wrong chunk, produces the wrong bytes rather than plausible ones. The
graph under test is `q4e.serving_shape.ngram_chunked_gather` itself, over
ports from `ngram_table_ports` -- the same code the serving-shape IR emits.

One leg per process:

    <venv>/bin/python tools/probe_ngram_table_ports.py --device GPU.1 --rows 4096 --chunks 3
    <venv>/bin/python tools/probe_ngram_table_ports.py --device GPU.1 --rows 53686272 --chunks 1
    <venv>/bin/python tools/probe_ngram_table_ports.py --device GPU.1 --over-cap
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

ROW_BYTES = 80                      # 160 nibbles, the real row
HN, T = 16, 5


def one_line(exc):
    return f"{type(exc).__name__}: " + " | ".join(
        s.strip() for s in str(exc).splitlines() if s.strip())


def say(stage, text):
    print(f"PROBE [{stage}] {text}", flush=True)


def sentinel_row(global_row, row_bytes):
    """Bytes of row r: (r * 2654435761 + j * 40503 + 17) mod 256 per byte j --
    every row distinct from its neighbours in every byte, and dependent on
    the GLOBAL index so a chunk mix-up is visible."""
    j = np.arange(row_bytes, dtype=np.uint64)
    v = (np.uint64(global_row) * np.uint64(2654435761) + j * np.uint64(40503)
         + np.uint64(17)) & np.uint64(0xFF)
    return v.astype(np.uint8)


def unpack(rows_u8):
    out = np.empty(rows_u8.shape[:-1] + (2 * rows_u8.shape[-1],), np.float32)
    out[..., 0::2] = rows_u8 & 0x0F
    out[..., 1::2] = rows_u8 >> 4
    return out


def gpu_mem(core, dev, label):
    if not dev.startswith("GPU"):
        return
    try:
        st = dict(core.get_property(dev, "GPU_MEMORY_STATISTICS"))
        say("gpu-mem", f"{label}: " + " ".join(
            f"{k}={v / 2 ** 30:.3f}GiB" for k, v in sorted(st.items()) if v))
    except Exception as exc:                                      # noqa: BLE001
        say("gpu-mem", f"{label}: unavailable ({one_line(exc)})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True)
    ap.add_argument("--rows", type=int, default=4096, help="rows per chunk")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--over-cap", action="store_true",
                    help="only: try the WHOLE real table as one USM-host "
                         "object and print the verdict verbatim")
    args = ap.parse_args(argv)

    import openvino as ov
    from openvino import opset13 as op
    from q4e import serving_shape as ss
    from q4e import piecewise_export as pwe

    core = ov.Core()
    dev = args.device
    say("env", f"openvino {ov.get_version()} device={dev}")
    ctx = core.get_default_context(dev) if dev.startswith("GPU") else None

    if args.over_cap:
        V = pwe.REAL_GEOMETRY["ngram_total_vocab"]
        say("over-cap", f"create_host_tensor(u8, [{V}, {ROW_BYTES}]) = "
                        f"{V * ROW_BYTES:,} B as ONE object")
        t0 = time.time()
        try:
            t = ctx.create_host_tensor(ov.Type.u8, ov.Shape([V, ROW_BYTES]))
            say("over-cap", f"ALLOCATED in {time.time() - t0:.2f}s -- the cap "
                            f"did NOT apply to a USM-host object of "
                            f"{t.get_byte_size():,} B")
        except Exception as exc:                                  # noqa: BLE001
            say("over-cap", f"REFUSED after {time.time() - t0:.2f}s " + one_line(exc))
        return 0

    rows, nchunks = args.rows, args.chunks
    n_rows = rows * nchunks
    cap = rows * ROW_BYTES
    parts = ss.ngram_table_chunks(n_rows, ROW_BYTES, cap, align=1)
    assert parts == [rows] * nchunks, parts
    say("shape", f"{nchunks} chunk(s) x {rows:,} rows x {ROW_BYTES} B = "
                 f"{cap:,} B each ({cap / 2 ** 30:.3f} GiB); "
                 f"last byte index of a chunk {cap - 1:,} "
                 f"({'under' if cap - 1 < 2 ** 32 else 'OVER'} 2**32)")

    # the graph: exactly the serving-shape IR's gather over its own ports
    ports = []
    for k in range(nchunks):
        p = op.parameter([rows, ROW_BYTES], ss.NGRAM_PORT_TYPE)
        p.set_friendly_name(f"ngram_table.{k}")
        p.output(0).set_names({f"ngram_table.{k}"})
        ports.append(p)
    row_ids = op.parameter([1, T, HN], ov.Type.i64)
    row_ids.set_friendly_name("ngram_row_ids")
    row_ids.output(0).set_names({"ngram_row_ids"})
    out = ss.ngram_chunked_gather(row_ids, ports, 2 * ROW_BYTES)
    model = ov.Model([op.result(out)], [row_ids] + ports, "ngram_port_probe")

    t0 = time.time()
    try:
        compiled = core.compile_model(model, dev)
    except Exception as exc:                                      # noqa: BLE001
        say("compile", f"FAIL after {time.time() - t0:.2f}s " + one_line(exc))
        return 0
    say("compile", f"OK {time.time() - t0:.2f}s ports: " + ", ".join(
        f"{p.get_any_name()}:{p.get_element_type().get_type_name()}"
        for p in compiled.inputs))
    req = compiled.create_infer_request()
    gpu_mem(core, dev, "after request, before table")

    # the sentinel rows: both edges of every chunk, the middle, a few random
    rng = np.random.default_rng(11)
    probe_rows = set()
    for k in range(nchunks):
        base = k * rows
        probe_rows |= {base, base + 1, base + rows // 2, base + rows - 2,
                       base + rows - 1}
    while len(probe_rows) < T * HN:
        probe_rows.add(int(rng.integers(0, n_rows)))
    ids = np.array(sorted(probe_rows)[:T * HN], np.int64).reshape(1, T, HN)

    tensors = []
    t0 = time.time()
    for k in range(nchunks):
        try:
            t = (ctx.create_host_tensor(ov.Type.u8, ov.Shape([rows, ROW_BYTES]))
                 if ctx else ov.Tensor(ov.Type.u8, ov.Shape([rows, ROW_BYTES])))
        except Exception as exc:                                  # noqa: BLE001
            say("alloc", f"ngram_table.{k}: FAIL " + one_line(exc))
            return 0
        tensors.append(t)
    say("alloc", f"{nchunks} x {cap:,} B of {'USM host' if ctx else 'host'} "
                 f"memory in {time.time() - t0:.2f}s")
    t0 = time.time()
    for r in ids.ravel():
        k, local = divmod(int(r), rows)
        tensors[k].data[local, :] = sentinel_row(int(r), ROW_BYTES)
    say("alloc", f"{ids.size} sentinel rows written through .data in "
                 f"{time.time() - t0:.3f}s")
    for k, t in enumerate(tensors):
        req.set_tensor(f"ngram_table.{k}", t)
    req.set_tensor("ngram_row_ids", ov.Tensor(ids))
    gpu_mem(core, dev, "after set_tensor")

    for i in range(2):
        t0 = time.time()
        try:
            req.infer()
        except Exception as exc:                                  # noqa: BLE001
            say("infer", f"#{i + 1} FAIL after {time.time() - t0:.3f}s " + one_line(exc))
            return 0
        say("infer", f"#{i + 1} OK {time.time() - t0:.3f}s")
    gpu_mem(core, dev, "after infer")

    got = req.get_output_tensor(0).data
    ref = unpack(np.stack([sentinel_row(int(r), ROW_BYTES)
                           for r in ids.ravel()]).reshape(1, T, HN, ROW_BYTES))
    bad = np.argwhere((got != ref).any(axis=-1))
    say("verdict", f"out{tuple(got.shape)} {got.dtype}; rows wrong "
                   f"{len(bad)} of {ids.size}; values wrong "
                   f"{int((got != ref).sum())} of {ref.size}")
    for b in bad[:8]:
        r = int(ids[tuple(b)])
        say("verdict", f"  row {r} (chunk {r // rows}, local {r % rows}): "
                       f"got {got[tuple(b)][:8].astype(int).tolist()} "
                       f"want {ref[tuple(b)][:8].astype(int).tolist()}")
    say("verdict", "EXACT" if len(bad) == 0 else "MISMATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
