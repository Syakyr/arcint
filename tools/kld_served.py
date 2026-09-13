#!/usr/bin/env python3
"""THE KLD GATE'S SERVED HALF: arcint's served-path logits against the pinned
llama.cpp reference capture, below AND above the QSA boundary. REPORT ONLY.

`tools/kld_capture.py` builds the reference half: `llama-perplexity
--kl-divergence-base` over the corpus at n_ctx = 2735, two windows, and the
capture's own token ids in its header. This tool is the other half:

  1. `--replay`: read the capture's windows and POST each one to a running
     arcint as a TOKEN-ID prompt (`/v1/completions`, `prompt: [ids]`, one
     greedy token) while the server writes `ARCINT_LOGITS_DUMP`. The ids are
     the capture's own -- no text round trip, so the served forward sees
     exactly the tokens the reference saw. The server must run with
     `--no-logits-slice` (every prefill row) and a prefill chunk that
     covers the window in as many forwards as it likes: the records are
     stitched by `past`.
  2. `--compare`: read the capture and the dump, reconstruct the reference
     log-probs row by row (the capture stores, per recorded row, an f32
     scale, an f32 min log-prob and n_vocab uint16 quantised logits -- the
     writer is `log_softmax(int, const float*, uint16_t*, int)` in
     llama.cpp's tools/perplexity/perplexity.cpp), log-softmax the served
     rows, and report mean per-token KL(P_ref || P_served) over the
     recorded rows, split at the boundary: rows at 0-based index < 2051 and
     rows >= 2051. The bar (tools/kld_harness.py THRESHOLD_NATS, 0.0599
     nats, PROVISIONAL) is printed beside both means with its provenance;
     nothing here decides.

What the capture records (kld_capture.py's correction, measured): the SECOND
HALF of each window only -- rows `n_ctx/2 .. n_ctx-2` (0-based), which is
`n_ctx - 1 - n_ctx/2` rows a window. Row p of window w is the distribution
over token p+1 given tokens 0..p of that window; the served dump's row p of
the same window is arcint's, so they align by index and nothing is shifted.

The dump: "ARCLGT01" + five uint64 (lane, past, n, rows, vocab) + rows x
vocab f32 per paged forward, in forward order. A window is every record from
a `past == 0` record up to (not including) the next one; its prefill rows are
the records with n > 1 concatenated in order (rows == n under
--no-logits-slice); the greedy decode record (past == window length) is
ignored. A sliced dump (rows < n) is refused by name.

A depth-4 artifact through this tool is the instrument's RED PROBE, not a
measurement: forty-four layers are missing and the KL must be far above the
bar. A full-depth artifact is the measurement.

THE INSTRUMENT'S OWN FLOOR (REVIEW ba2d5de F1): served logits at 2,735
tokens are not run-to-run deterministic on this backend (chunked prefill,
f16 kernels), so a reading needs its noise floor beside it. `--replay
--repeat N` posts every window N times; `--compare` then pairs every earlier
replay with the last one and reports KL(A||B) mean/max, argmax agreement and
max |logit diff| over the same recorded rows -- the floor the gate's means
are read against, and with N > 2 how often the event occurs. A floor near
the bar makes the bar unreadable; a floor far below it does not make a
reading pass. The floor's mechanism is measured by
tools/boot_serving_shape.py --repeat/--cut (the per-node bisect), not here.

THE BAR (REVIEW ba2d5de F2; PROVISIONAL): 0.0599 nats is NOT derived from
Flash-Next. It is kld_harness.THRESHOLD_NATS = 1.5 x 0.0399, the R0 of a
DIFFERENT model (Qwen3.6-35B-A3B, UD-Q3_K_XL against BF16 on wikitext-2,
-c 512, 64 chunks, 2026-08-11); the derivation and its three conditions live
in that module and its provenance sentence is printed beside every reading
here. The 0.5.1 acceptance commit re-derives the bar from this model's own
reference round-trip; until then this tool prints the inherited number,
tagged, and decides nothing.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from kld_harness import BAR_PROVENANCE, BAR_STATUS, THRESHOLD_NATS  # noqa: E402

QSA_BOUNDARY_TOKENS = 2051
DUMP_MAGIC = b"ARCLGT01"


# ---------------------------------------------------------------- the capture
def read_capture(path):
    """The llama.cpp `--kl-divergence-base` file. Returns (n_ctx, n_vocab,
    n_chunk, tokens [n_chunk, n_ctx] int32, rows [n_chunk, n_rows, nv]
    uint16 memmap) with nv = 2 * ((n_vocab + 1) // 2) + 4 and n_rows =
    n_ctx - 1 - n_ctx // 2."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"_logits_":
            raise ValueError(f"{path}: not a llama.cpp logits capture (magic {magic!r})")
        n_ctx = struct.unpack("<i", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunk = struct.unpack("<i", f.read(4))[0]
        tokens = np.fromfile(f, dtype=np.int32, count=n_chunk * n_ctx).reshape(n_chunk, n_ctx)
        off = f.tell()
    first = n_ctx // 2
    n_rows = n_ctx - 1 - first
    nv = 2 * ((n_vocab + 1) // 2) + 4
    rows = np.memmap(path, dtype=np.uint16, mode="r", offset=off,
                     shape=(n_chunk, n_rows, nv))
    return n_ctx, n_vocab, n_chunk, tokens, rows


def reference_log_probs(row_u16, n_vocab):
    """One recorded row -> log P_ref over the vocab, f64. The writer stores
    scale (f32), min_log_prob (f32), then q[i] = round((logit_i - min_logit)
    / scale) for logit_i > min_logit else 0, with min_logit clamped to
    max_logit - 16; so log P(i) = min_log_prob + q[i] * scale, and a q of 0
    is the floor (16 nats below the max, or lower)."""
    hdr = np.asarray(row_u16[:4]).view(np.float32)
    scale, min_log_prob = float(hdr[0]), float(hdr[1])
    q = np.asarray(row_u16[4:4 + n_vocab], dtype=np.float64)
    return min_log_prob + q * scale


def log_softmax(x):
    z = x - x.max()
    return z - np.log(np.exp(z).sum())


def kl_ref_vs_served(ref_logp, served_logits):
    """KL(P_ref || P_served) in nats for one row. P_ref from the capture's
    reconstructed log-probs (renormalised: the uint16 floor makes them sum to
    slightly under 1), P_served from a log-softmax of the served f32 row."""
    ref = log_softmax(np.asarray(ref_logp, dtype=np.float64))
    cand = log_softmax(np.asarray(served_logits, dtype=np.float64))
    p = np.exp(ref)
    return float((p * (ref - cand)).sum())


# ------------------------------------------------------------------- the dump
def read_dump(path):
    """Every record of an ARCINT_LOGITS_DUMP file, in order:
    [(lane, past, n, rows, vocab, offset_bytes)], the f32 data left on disk."""
    recs = []
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        while f.tell() < size:
            magic = f.read(8)
            if magic != DUMP_MAGIC:
                raise ValueError(f"{path}: bad record magic {magic!r} at {f.tell() - 8}")
            lane, past, n, rows, vocab = struct.unpack("<5Q", f.read(40))
            off = f.tell()
            f.seek(rows * vocab * 4, 1)
            recs.append((lane, past, n, rows, vocab, off))
    return recs


def dump_windows(recs):
    """Group records into windows: each window starts at a past == 0 record.
    Returns [[rec, ...], ...] in order."""
    windows = []
    for r in recs:
        if r[1] == 0:
            windows.append([r])
        elif windows:
            windows[-1].append(r)
        else:
            raise ValueError(f"dump starts mid-window (first record has past {r[1]})")
    return windows


def window_rows(path, window, n_ctx, vocab):
    """The served [n_ctx, vocab] f32 logits of one window, stitched from its
    prefill records by `past`. Refuses a sliced record (rows < n) and a gap."""
    out = np.empty((n_ctx, vocab), dtype=np.float32)
    have = 0
    for lane, past, n, rows, v, off in window:
        if past >= n_ctx:
            continue                                   # the greedy decode step
        if v != vocab:
            raise ValueError(f"record vocab {v} != {vocab}")
        if rows != n:
            raise ValueError(f"record at past {past} has {rows} rows for {n} tokens: a sliced "
                             f"dump (run the server with --no-logits-slice)")
        if past != have:
            raise ValueError(f"record at past {past} but {have} rows stitched so far")
        data = np.memmap(path, dtype=np.float32, mode="r", offset=off, shape=(rows, v))
        out[past:past + rows] = data
        have += rows
    if have != n_ctx:
        raise ValueError(f"window has {have} served rows, the capture's n_ctx is {n_ctx}")
    return out


# --------------------------------------------------------------------- stages
def replay(args):
    import urllib.request
    n_ctx, n_vocab, n_chunk, tokens, _rows = read_capture(args.ref)
    print(f"capture {args.ref}: n_ctx {n_ctx} n_vocab {n_vocab} windows {n_chunk}", flush=True)
    for rep in range(int(args.repeat)):
      for w in range(n_chunk):
        if args.windows and w not in args.windows:
            continue
        ids = [int(t) for t in tokens[w]]
        body = json.dumps({"model": "x", "prompt": ids, "max_tokens": 1,
                           "temperature": 0}).encode()
        req = urllib.request.Request(args.url.rstrip("/") + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            reply = json.loads(resp.read())
        usage = reply.get("usage", {})
        print(f"replay {rep} window {w}: {len(ids)} ids -> HTTP OK in {time.time() - t0:.1f}s; "
              f"usage {usage}; text {reply.get('choices', [{}])[0].get('text', '')!r}", flush=True)
    return 0


def compare(args):
    n_ctx, n_vocab, n_chunk, tokens, rows = read_capture(args.ref)
    recs = read_dump(args.dump)
    all_windows = dump_windows(recs)
    # The server's own load-time probe forwards are in the dump too (every
    # paged forward is): windows of 128/512/... zero tokens. The replayed
    # windows are the ones whose prefill rows sum to the capture's n_ctx; of
    # those, the LAST n_chunk are the replay (the tool posts them in order).
    windows = [w for w in all_windows
               if sum(r[3] for r in w if r[1] < n_ctx and r[1] + r[3] <= n_ctx) == n_ctx]
    replays = windows
    windows = replays[-n_chunk:]
    print(f"capture: n_ctx {n_ctx} n_vocab {n_vocab} windows {n_chunk}; dump: {len(recs)} "
          f"records in {len(all_windows)} window(s), {len(replays)} of them {n_ctx}-token "
          f"replays (the rest: the server's load-time probes)", flush=True)
    if len(windows) != n_chunk:
        raise ValueError(f"dump holds {len(replays)} replayed window(s) of {n_ctx} tokens, "
                         f"the capture has {n_chunk}")
    first = n_ctx // 2
    n_rows = n_ctx - 1 - first
    report = {"ref": args.ref, "dump": args.dump, "n_ctx": n_ctx, "n_vocab": n_vocab,
              "boundary": QSA_BOUNDARY_TOKENS, "threshold_nats": THRESHOLD_NATS,
              "threshold_status": BAR_STATUS, "threshold_provenance": BAR_PROVENANCE,
              "windows": []}
    all_below, all_above = [], []
    for w, win in enumerate(windows[:n_chunk]):
        served = window_rows(args.dump, win, n_ctx, n_vocab)
        if not np.isfinite(served).all():
            raise ValueError(f"window {w}: served logits are not all finite")
        kl_below, kl_above = [], []
        argmax_agree = 0
        for i in range(n_rows):
            p = first + i                                  # 0-based row index
            ref_lp = reference_log_probs(rows[w, i], n_vocab)
            kl = kl_ref_vs_served(ref_lp, served[p])
            (kl_below if p < QSA_BOUNDARY_TOKENS else kl_above).append(kl)
            argmax_agree += int(np.argmax(ref_lp) == np.argmax(served[p]))
        entry = {"window": w, "rows": n_rows, "rows_below": len(kl_below),
                 "rows_above": len(kl_above),
                 "mean_kl_below": float(np.mean(kl_below)) if kl_below else None,
                 "mean_kl_above": float(np.mean(kl_above)) if kl_above else None,
                 "max_kl": float(max(kl_below + kl_above)),
                 "argmax_agreement": argmax_agree / n_rows}
        all_below += kl_below
        all_above += kl_above
        report["windows"].append(entry)
        print(f"window {w}: rows {n_rows} (below {len(kl_below)}, above {len(kl_above)}); "
              f"mean KL below {entry['mean_kl_below']:.6e} above "
              f"{entry['mean_kl_above'] if entry['mean_kl_above'] is None else format(entry['mean_kl_above'], '.6e')}; "
              f"max {entry['max_kl']:.4e}; argmax agreement {entry['argmax_agreement']:.4f}",
              flush=True)
    report["mean_kl_below"] = float(np.mean(all_below)) if all_below else None
    report["mean_kl_above"] = float(np.mean(all_above)) if all_above else None
    # THE FLOOR: every earlier replay (A) against the last one (B), same rows.
    # `windows` is the last pair's per-window list (the shape the first floor
    # reading was recorded in); `pairs` carries every pair, so a run with
    # --repeat N says how many of N-1 pairs moved, not just whether the last did.
    report["floor"] = None
    n_rep = len(replays) // n_chunk
    if n_rep >= 2:
        pairs = []
        for a_idx in range(n_rep - 1):
            prev = replays[a_idx * n_chunk:(a_idx + 1) * n_chunk]
            fl = []
            for w in range(n_chunk):
                a = window_rows(args.dump, prev[w], n_ctx, n_vocab)
                b = window_rows(args.dump, windows[w], n_ctx, n_vocab)
                kls, agree, maxdiff = [], 0, 0.0
                for i in range(n_rows):
                    p = first + i
                    kls.append(kl_ref_vs_served(log_softmax(a[p].astype(np.float64)), b[p]))
                    agree += int(np.argmax(a[p]) == np.argmax(b[p]))
                    maxdiff = max(maxdiff, float(np.abs(a[p] - b[p]).max()))
                fl.append({"window": w, "mean_kl_a_b": float(np.mean(kls)),
                           "max_kl_a_b": float(max(kls)), "argmax_agreement": agree / n_rows,
                           "max_abs_logit_diff": maxdiff})
                print(f"FLOOR replay {a_idx} vs {n_rep - 1}, window {w}: KL(A||B) mean "
                      f"{fl[-1]['mean_kl_a_b']:.6e} max {fl[-1]['max_kl_a_b']:.4e}; argmax "
                      f"agreement {fl[-1]['argmax_agreement']:.4f}; max |logit diff| "
                      f"{maxdiff:.3f}", flush=True)
            pairs.append({"a": a_idx, "b": n_rep - 1, "windows": fl,
                          "mean_kl_a_b": float(np.mean([f["mean_kl_a_b"] for f in fl])),
                          "max_kl_a_b": float(max(f["max_kl_a_b"] for f in fl)),
                          "windows_moved": sum(int(f["max_abs_logit_diff"] > 0) for f in fl)})
        last = pairs[-1]
        moved = sum(p["windows_moved"] for p in pairs)
        report["floor"] = {"windows": last["windows"], "mean_kl_a_b": last["mean_kl_a_b"],
                           "max_kl_a_b": last["max_kl_a_b"], "replays": n_rep,
                           "pairs": pairs,
                           "window_pairs_moved": moved,
                           "window_pairs": (n_rep - 1) * n_chunk}
        print(f"FLOOR ALL: KL(A||B) mean {last['mean_kl_a_b']:.6e} max {last['max_kl_a_b']:.4e} "
              f"(last pair); {moved} of {(n_rep - 1) * n_chunk} window pairs moved across "
              f"{n_rep} replays of the same windows in the same process; the means above are "
              f"read against this", flush=True)
    else:
        print("FLOOR: not measured (one replay; --replay --repeat 2 gives it)", flush=True)
    print(f"ALL: mean KL below {report['mean_kl_below']:.6e} above "
          f"{report['mean_kl_above'] if report['mean_kl_above'] is None else format(report['mean_kl_above'], '.6e')} "
          f"(bar {THRESHOLD_NATS} nats, {BAR_STATUS} -- {BAR_PROVENANCE}; REPORT ONLY)",
          flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="the llama.cpp --kl-divergence-base capture")
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--replay", action="store_true")
    sub.add_argument("--compare", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--windows", type=int, nargs="*", default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="replay every window this many times (2 = the floor)")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--dump", default=None, help="the ARCINT_LOGITS_DUMP file")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args(argv)
    if args.replay:
        return replay(args)
    if not args.dump:
        ap.error("--compare needs --dump")
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
