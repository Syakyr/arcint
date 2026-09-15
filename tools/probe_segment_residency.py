#!/usr/bin/env python3
"""B.3: what a SEGMENTED artifact actually costs to hold, pool by pool.

window-051 row (a) prices the 4 x 12 cut by adding four terms into one
budget -- the n-gram table (26.82 GiB), one expert buffer set (14.06 GiB),
a segment's dense staging, and the process -- and predicts the sum does
NOT fit "the 48 GiB host". This probe exists because that sentence hides
two questions the arithmetic cannot answer:

  1. WHICH POOL. The table is USM host: window-050 §4.10 F2 measured that
     driver / USM-host memory is NOT charged to the container's cgroup --
     the `MemoryMax=44G` fence never engaged while the PHYSICAL host ran
     out. The buffer set is ordinary process pages and IS charged. So
     "the 48 GiB host" is two budgets, not one, and a sum across them is
     not a budget at all. This probe reports both columns at every step
     and never adds them.

  2. WHICH ORDER. Row (a)'s peak assumes a segment compiles while the
     table is resident. Whether any compile ever does is a property of
     the export: only the segment carrying the PLE declares
     `ngram_table.*` ports, and the table can only be bound to a request,
     which only exists after that segment is compiled. `--order` runs
     both orders so the premise is measured instead of assumed:

       compile-then-table   the serving order: compile every segment,
                            create the requests, THEN bind the table
       table-then-compile   row (a)'s premise: allocate the table's
                            USM-host tensors FIRST, hold them, and
                            compile every segment beside them

Nothing here is a test and nothing here serves: no forward is run, no
logits are read, and the bodies are whatever the allocation holds unless
--shards is given. It measures residency and nothing else. One leg per
process:

    <venv>/bin/python tools/probe_segment_residency.py --dry-run \\
        --artifact /models/ov/qwen38-flash-next-seg12-ov
    <venv>/bin/python tools/probe_segment_residency.py --device GPU.0 \\
        --artifact /models/ov/qwen38-flash-next-seg12-ov \\
        --order compile-then-table --buffer-set usm

The PHYSICAL host's column cannot be read from inside the container --
/proc/meminfo there is the cgroup's view. The host-side sampler (root on
the physical host, A.1's wpA1-sampler.sh) supplies it, and this probe
prints the timestamps that align the two logs.
"""
import argparse
import json
import re
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

SEG_DIR = re.compile(r"^segment(\d+)$")
TABLE_PREFIX = "ngram_table."
EXPERT_RE = re.compile(r"^layer(\d+)/moe/experts_(gate|up|down)/weight_u8$")


def say(stage, text):
    print(f"RESID [{stage}] {text}", flush=True)


def one_line(exc):
    return f"{type(exc).__name__}: " + " | ".join(
        s.strip() for s in str(exc).splitlines() if s.strip())


def gib(n):
    return n / 2 ** 30


def dims(port):
    ps = port.get_partial_shape()
    if ps.rank.is_dynamic:
        return None
    return [d.get_length() if d.is_static else -1 for d in ps]


# ---- the plan, read off the xml: pure arithmetic, no device ------------------

def discover_segments(root):
    """The segment directories of an artifact, in segment order.

    A gap or a non-contiguous index is a REFUSAL, not a sort: the chain's
    identity is the segment order (84b63b8's chain arch_hash), so a
    missing segment must not silently become a shorter chain.
    """
    root = Path(root)
    found = {}
    for child in sorted(root.iterdir()):
        m = SEG_DIR.match(child.name)
        if m and child.is_dir():
            found[int(m.group(1))] = child
    if not found:
        raise ValueError(f"{root}: no segmentN/ directory -- not a segmented artifact")
    want = list(range(len(found)))
    if sorted(found) != want:
        raise ValueError(f"{root}: segment indices {sorted(found)} are not "
                         f"contiguous from 0 ({want} expected)")
    return [found[i] for i in want]


def port_census(model):
    """The three port classes this probe prices, by name, off ONE model."""
    table, experts, other = [], [], []
    for p in model.get_parameters():
        name = p.get_friendly_name()
        shape = dims(p.output(0))
        nbytes = 0
        if shape and all(d >= 0 for d in shape):
            nbytes = p.get_element_type().size
            for d in shape:
                nbytes *= d
        row = (name, shape, nbytes)
        if name.startswith(TABLE_PREFIX):
            table.append(row)
        elif EXPERT_RE.match(name):
            experts.append(row)
        else:
            other.append(row)
    table.sort(key=lambda r: int(r[0].split(".")[1]))
    return {"table": table, "experts": experts, "other": other}


def plan(censuses):
    """The residency plan of a chain of segments.

    buffer_set_bytes is the MAX over segments, not the sum: the design's
    one host buffer set is refilled per segment, so the chain needs the
    largest segment's expert bytes resident, once.
    """
    table_bytes = sum(b for _, _, b in censuses[0]["table"]) if censuses else 0
    for k, c in enumerate(censuses[1:], start=1):
        if c["table"]:
            raise ValueError(f"segment {k} declares {len(c['table'])} ngram_table "
                             f"port(s); only the PLE-carrying segment may")
    per_seg = [sum(b for _, _, b in c["experts"]) for c in censuses]
    slots = [len(c["experts"]) for c in censuses]
    return {
        "segments": len(censuses),
        "table_bytes": table_bytes,
        "table_ports": len(censuses[0]["table"]) if censuses else 0,
        "expert_bytes_per_segment": per_seg,
        "expert_slots_per_segment": slots,
        "buffer_set_bytes": max(per_seg) if per_seg else 0,
        "blob_bytes": sum(per_seg),
    }


# ---- the two pools, sampled -------------------------------------------------

def _proc_status():
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                out[line.split(":")[0]] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return out


def _cgroup_meminfo():
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k = line.split(":")[0]
            if k in ("MemTotal", "MemAvailable", "MemFree", "Shmem"):
                out[k] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return out


def _fdinfo_drm():
    """drm-resident-gtt / -vram0 summed over this process's unique DRM
    clients -- the driver's own accounting, which `free` cannot see
    (mneme 372, window-050 §4.10 F2). Values are KiB in the xe fdinfo."""
    gtt, vram = {}, {}
    d = Path("/proc/self/fdinfo")
    if not d.is_dir():
        return {}
    for f in d.iterdir():
        try:
            cid = None
            for line in f.read_text().splitlines():
                if line.startswith("drm-client-id:"):
                    cid = line.split(":", 1)[1].strip()
                elif line.startswith("drm-resident-gtt:") and cid is not None:
                    gtt[cid] = int(line.split(":", 1)[1].split()[0]) * 1024
                elif line.startswith("drm-resident-vram0:") and cid is not None:
                    vram[cid] = int(line.split(":", 1)[1].split()[0]) * 1024
        except (OSError, ValueError):
            continue
    return {"drm_gtt": sum(gtt.values()), "drm_vram0": sum(vram.values()),
            "drm_clients": len(gtt)}


def sampler(core, dev):
    def sample(label):
        st = _proc_status()
        mi = _cgroup_meminfo()
        drm = _fdinfo_drm()
        gpu = {}
        if dev and dev.startswith("GPU"):
            try:
                gpu = {k: v for k, v in dict(core.get_property(dev, "GPU_MEMORY_STATISTICS")).items() if v}
            except Exception:                                      # noqa: BLE001
                gpu = {}
        say("mem", f"{label} t={time.strftime('%H:%M:%SZ', time.gmtime())} "
                   f"rss={gib(st.get('VmRSS', 0)):.2f}GiB "
                   f"hwm={gib(st.get('VmHWM', 0)):.2f}GiB "
                   f"cgroup_avail={gib(mi.get('MemAvailable', 0)):.2f}GiB "
                   f"drm_gtt={gib(drm.get('drm_gtt', 0)):.2f}GiB "
                   f"drm_vram0={gib(drm.get('drm_vram0', 0)):.2f}GiB "
                   + " ".join(f"{k}={gib(v):.2f}GiB" for k, v in sorted(gpu.items())))
        return {"label": label, "rss": st.get("VmRSS", 0), "hwm": st.get("VmHWM", 0),
                "cgroup_avail": mi.get("MemAvailable", 0), **drm,
                **{f"gpu_{k}": v for k, v in gpu.items()}}
    return sample


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True,
                    help="a SEGMENTED artifact directory (segment0/, segment1/, ...)")
    ap.add_argument("--device", default=None,
                    help="GPU.0 / GPU.1 / CPU; absent = --dry-run only")
    ap.add_argument("--dry-run", action="store_true",
                    help="read every segment's xml and print the plan; no compile, "
                         "no allocation, no device")
    ap.add_argument("--segments", default=None,
                    help="comma-separated segment indices to use (default: all). "
                         "A SHORTER CHAIN IS NOT THE MODEL -- the output says so.")
    ap.add_argument("--order", default="compile-then-table",
                    choices=("compile-then-table", "table-then-compile"),
                    help="compile-then-table = the serving order; "
                         "table-then-compile = window-051 row (a)'s premise")
    ap.add_argument("--buffer-set", default="none", choices=("none", "usm", "heap"),
                    help="allocate one expert buffer set after the compiles: "
                         "usm = the device context's host tensors (the table's "
                         "own class), heap = ordinary process pages "
                         "(src/exec/segment_runtime.h's std::vector today)")
    ap.add_argument("--table", default="alloc", choices=("alloc", "skip"),
                    help="skip = do not allocate the n-gram table at all (the "
                         "control that prices the table's own share)")
    ap.add_argument("--shards", default=None,
                    help="fill the table from the GGUF's own IQ4_NL bytes "
                         "(q4e.gguf_feed); absent = the allocation's own pages")
    ap.add_argument("--paged-kv", default="u8")
    ap.add_argument("--json", default=None, help="write the samples to this path")
    args = ap.parse_args(argv)

    t_start = time.time()
    say("env", f"artifact={args.artifact} device={args.device} order={args.order} "
               f"buffer_set={args.buffer_set} table={args.table} "
               f"start_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    try:
        seg_dirs = discover_segments(args.artifact)
    except ValueError as exc:
        say("plan", "REFUSED " + one_line(exc))
        return 2
    if args.segments:
        want = [int(x) for x in args.segments.split(",") if x.strip()]
        bad = [i for i in want if i >= len(seg_dirs)]
        if bad:
            say("plan", f"REFUSED segment(s) {bad} do not exist "
                        f"({len(seg_dirs)} in the artifact)")
            return 2
        seg_dirs = [seg_dirs[i] for i in want]
        say("plan", f"PARTIAL CHAIN {want} of {len(want)} -- NOT the model, "
                    f"a residency leg over a subset")

    import openvino as ov
    core = ov.Core()
    say("env", f"openvino {ov.get_version()}")

    models, censuses = [], []
    for d in seg_dirs:
        xml = d / "openvino_language_model.xml"
        t0 = time.time()
        try:
            m = core.read_model(str(xml))
        except Exception as exc:                                   # noqa: BLE001
            say("plan", f"READ FAIL {xml}: " + one_line(exc))
            return 2
        c = port_census(m)
        models.append(m)
        censuses.append(c)
        say("plan", f"{d.name}: read {time.time() - t0:.2f}s "
                    f"nodes={len(m.get_ordered_ops())} "
                    f"table_ports={len(c['table'])} "
                    f"expert_ports={len(c['experts'])} "
                    f"({gib(sum(b for _, _, b in c['experts'])):.2f}GiB) "
                    f"other={len(c['other'])} "
                    f"bin={(d / 'openvino_language_model.bin').stat().st_size:,}B")

    try:
        p = plan(censuses)
    except ValueError as exc:
        say("plan", "REFUSED " + one_line(exc))
        return 2
    say("plan", f"CHAIN {p['segments']} segment(s); "
                f"table {p['table_bytes']:,}B ({gib(p['table_bytes']):.2f}GiB) "
                f"over {p['table_ports']} port(s); "
                f"buffer set {p['buffer_set_bytes']:,}B "
                f"({gib(p['buffer_set_bytes']):.2f}GiB) = max over segments; "
                f"blob {p['blob_bytes']:,}B ({gib(p['blob_bytes']):.2f}GiB) read per forward")
    say("plan", "THE TWO POOLS, never summed: the table is USM host (physical "
                "host, invisible to the cgroup -- window-050 §4.10 F2); the "
                "heap buffer set is process pages (cgroup). Which pool the USM "
                "buffer set lands in is what --buffer-set usm measures.")

    if args.dry_run or not args.device:
        say("done", f"DRY RUN, nothing allocated, {time.time() - t_start:.1f}s")
        return 0

    dev = args.device
    sample = sampler(core, dev)
    samples = [sample("baseline")]

    tctx = core.get_default_context(dev) if dev.startswith("GPU") else None
    table_tensors = []
    raw = None
    if args.shards and args.table == "alloc":
        from q4e import gguf_feed as gf
        t0 = time.time()
        feed_ = gf.GgufFeed(args.shards)
        import numpy as np
        raw = np.asarray(feed_.raw_table())
        say("table", f"feed over {args.shards} in {time.time() - t0:.1f}s; "
                     f"REAL table {raw.shape} {raw.dtype}")

    def alloc_table():
        """The table as the served path binds it: one USM-host tensor per
        `ngram_table.K` port, from the device's own context."""
        if args.table == "skip" or not censuses[0]["table"]:
            say("table", "SKIPPED" if args.table == "skip"
                         else "no ngram_table port on the first segment of this chain")
            return
        import numpy as np
        t0, off, total = time.time(), 0, 0
        for name, shape, nbytes in censuses[0]["table"]:
            et = next(p.get_element_type() for p in models[0].get_parameters()
                      if p.get_friendly_name() == name)
            try:
                t = (tctx.create_host_tensor(et, ov.Shape(shape)) if tctx
                     else ov.Tensor(et, ov.Shape(shape)))
            except Exception as exc:                               # noqa: BLE001
                say("table", f"{name}{shape}: ALLOC FAIL " + one_line(exc))
                raise
            if raw is not None:
                t1 = time.time()
                np.copyto(t.data, raw[off:off + shape[0]])
                say("table", f"{name}: rows {off:,}..{off + shape[0]:,} "
                             f"copied in {time.time() - t1:.1f}s")
                off += shape[0]
            table_tensors.append((name, t))
            total += nbytes
        say("table", f"allocated {len(table_tensors)} port(s), {total:,}B "
                     f"({gib(total):.2f}GiB) of {'USM host' if tctx else 'host'} "
                     f"in {time.time() - t0:.2f}s")

    if args.order == "table-then-compile":
        say("order", "row (a)'s premise: the table is allocated BEFORE any "
                     "compile and held across all of them")
        alloc_table()
        samples.append(sample("after-table"))

    compiled = []
    props = {"KV_CACHE_PRECISION": getattr(ov.Type, args.paged_kv)}
    for d, m in zip(seg_dirs, models):
        t0 = time.time()
        try:
            cm = core.compile_model(m, dev, props)
        except Exception as exc:                                   # noqa: BLE001
            say("compile", f"{d.name} FAIL after {time.time() - t0:.1f}s "
                           f"peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f}GiB "
                           + one_line(exc))
            samples.append(sample(f"compile-FAIL-{d.name}"))
            if args.json:
                Path(args.json).write_text(json.dumps({"plan": p, "samples": samples}, indent=1))
            return 1
        compiled.append(cm)
        say("compile", f"{d.name} OK {time.time() - t0:.1f}s")
        samples.append(sample(f"after-compile-{d.name}"))

    requests = []
    for d, cm in zip(seg_dirs, compiled):
        requests.append(cm.create_infer_request())
    say("request", f"{len(requests)} infer request(s) created")
    samples.append(sample("after-requests"))

    if args.order == "compile-then-table":
        say("order", "the serving order: every segment compiled and its "
                     "request created, THEN the table bound")
        alloc_table()
        samples.append(sample("after-table"))

    if table_tensors:
        declared = {pp.get_any_name() for pp in compiled[0].inputs}
        for name, t in table_tensors:
            if name not in declared:
                say("bind", f"{name} is not an input of segment 0's compiled model")
                continue
            try:
                requests[0].set_tensor(name, t)
            except Exception as exc:                               # noqa: BLE001
                say("bind", f"set_tensor({name}) THREW " + one_line(exc))
                return 1
        say("bind", f"{len(table_tensors)} table port(s) bound to segment 0's request")
        samples.append(sample("after-table-bind"))

    buffers = []
    if args.buffer_set != "none":
        import numpy as np
        big = max(range(len(censuses)), key=lambda i: sum(b for _, _, b in censuses[i]["experts"]))
        t0, total = time.time(), 0
        for name, shape, nbytes in censuses[big]["experts"]:
            et = next(pp.get_element_type() for pp in models[big].get_parameters()
                      if pp.get_friendly_name() == name)
            try:
                if args.buffer_set == "usm" and tctx:
                    t = tctx.create_host_tensor(et, ov.Shape(shape))
                else:
                    t = ov.Tensor(et, ov.Shape(shape))
                # the pages must be REAL: an untouched mapping is not resident,
                # and a buffer set that is never written is not a buffer set.
                t.data[..., :1] = 0
            except Exception as exc:                               # noqa: BLE001
                say("buffer", f"{name}{shape}: ALLOC FAIL " + one_line(exc))
                samples.append(sample("buffer-FAIL"))
                if args.json:
                    Path(args.json).write_text(json.dumps({"plan": p, "samples": samples}, indent=1))
                return 1
            buffers.append(t)
            total += nbytes
        say("buffer", f"ONE buffer set: {len(buffers)} slot(s) of "
                      f"{seg_dirs[big].name}, {total:,}B ({gib(total):.2f}GiB) "
                      f"{'USM host' if args.buffer_set == 'usm' and tctx else 'process heap'} "
                      f"in {time.time() - t0:.1f}s")
        samples.append(sample("after-buffer-set"))

    say("done", f"held: {len(compiled)} compiled segment(s), "
                f"{len(table_tensors)} table port(s), {len(buffers)} buffer slot(s); "
                f"{time.time() - t_start:.1f}s total")
    if args.json:
        Path(args.json).write_text(json.dumps({"plan": p, "samples": samples}, indent=1))
        say("done", f"samples -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
