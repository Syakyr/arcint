#!/usr/bin/env python3
"""REFERENCE-LOGIT CAPTURE for the 0.5.0 KLD gate, and only the capture.

The KLD gate is the quality adjudicator for Flash-Next serving: arcint's served
logits against a reference implementation's, below AND above the QSA boundary.
THIS TOOL BUILDS THE REFERENCE HALF. The comparison is deliberately not here --
it needs served logits, and there are none yet.

--------------------------------------------------------------------------
THE REFERENCE IS NOT A FORK ANY MORE
--------------------------------------------------------------------------

The roadmap recorded the reference as "the llama.cpp qwen4exp fork", carrying a
companion PR. That is superseded: `qwen4exp` is IN UPSTREAM MASTER. Verified
2026-09-13 -- `LLM_ARCH_QWEN4EXP` is registered in `src/llama-arch.cpp` and the
graph is its own translation unit, `src/models/qwen4exp.cpp`; the shipped GGUF
declares `general.architecture = "qwen4exp"`, so they are the same arch string.
Pin a master commit, not a PR branch.

Upstream's own `llama-perplexity --kl-divergence-base FNAME` (alias
`--save-all-logits`) IS the capture, and its `--kl-divergence` is the
comparison half this tool does not invoke. Nothing here reimplements either;
what this adds is the part a bare invocation cannot carry:

  * THE REFUSAL BELOW, which is the whole reason the tool exists rather than a
    shell line;
  * a MANIFEST beside the capture -- every input hashed, so a capture found on
    disk in three weeks can be attributed to a reference, a model and a corpus
    rather than to a memory.

--------------------------------------------------------------------------
WHY A CONTEXT THAT DOES NOT CROSS THE BOUNDARY IS REFUSED
--------------------------------------------------------------------------

The gate's whole shape is a comparison on BOTH sides of the QSA boundary:
below it the served logits must sit inside the noise floor, above it they must
diverge by the measured QSA price and no more. The boundary and the price are
measured facts, not settings:

    T <= 2051   price 0.0, exact          (the boundary is DERIVED, and it is
                                           not the 2048 budget)
    T == 2052   2.307817e-06 over 1/2052 rows
    T == 2080   2.385560e-02 over 29/2080 rows = T - 2051

The count of touched rows is `T - 2051`, and it is DERIVED FROM THOSE TWO
MEASURED POINTS rather than from reasoning about position indices -- which is
how the first draft of this file got it wrong, by one, in both the arithmetic
and the test that was supposed to pin it. `T - 1 - 2051` gives 0 and 28 where
the measurement says 1 and 29. So the smallest window the price touches at all
is 2052, not 2053.

A capture taken at a context of 2051 or less touches no row. It is not a worse
capture; it is a capture the gate CANNOT USE for half its job, and it looks
exactly like a usable one on disk. So it is refused by name here rather than
discovered later -- arcint's standing pattern for the class.

--------------------------------------------------------------------------
WHAT THIS TOOL DOES NOT KNOW
--------------------------------------------------------------------------

Every path is an argument. The reference binary, the model and the corpus live
on the measurement host and none of their locations belong in a public
repository; the tool hashes whatever it is handed and records the hashes.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

# THE QSA BOUNDARY, measured, not chosen: the largest T at which the
# QSA-to-dense price is exactly 0.0. The budget is 2048; the boundary is 2051,
# and the difference is the point -- see this module's docstring for the three
# measured rows.
QSA_BOUNDARY_TOKENS = 2051

# `llama-perplexity` evaluates `--chunks` windows of `--ctx-size` tokens each
# and needs at least two windows' worth of tokens before it will start.
MIN_CHUNKS = 2


class CaptureRefusal(RuntimeError):
    """A capture that would be taken but could not serve the gate."""


def rows_past_boundary(n_ctx, boundary=QSA_BOUNDARY_TOKENS):
    """How many rows of a window of `n_ctx` tokens the QSA price touches.

    DERIVED FROM THE TWO MEASURED POINTS, not from reasoning about position
    indices -- which is how this got written wrong once. The recorded rows are

        T = 2052   1 / 2052 rows
        T = 2080  29 / 2080 rows = T - 2051

    so the count is `T - boundary`, and the first draft's `T - 1 - boundary`
    gives 0 and 28 against a measured 1 and 29. The smallest window that
    touches the price at all is therefore `boundary + 1` = 2052, not 2053.
    Negative counts are clamped to zero; the caller decides what to do.
    """
    return max(0, n_ctx - boundary)


def check_context_crosses_boundary(n_ctx, boundary=QSA_BOUNDARY_TOKENS):
    """Refuse a context whose window the QSA price never touches."""
    touched = rows_past_boundary(n_ctx, boundary)
    if touched:
        return touched
    raise CaptureRefusal(
        f"--n-ctx {n_ctx} cannot serve the KLD gate: the QSA price touches "
        f"T - {boundary} rows of a T-token window, which is "
        f"{n_ctx - boundary} here, so NO row in this capture carries it. The "
        f"gate needs both sides -- below the boundary the served logits are "
        f"judged against the noise floor, above it against the measured price "
        f"(0.0 at T<={boundary}, 1/2052 rows at T=2052, 2.385560e-02 over "
        f"29/2080 rows at T=2080). Use --n-ctx {boundary + 1} or more; a "
        f"capture taken here would look usable on disk and be half-useless.")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_head(path):
    """The commit of the checkout `path` sits in, or None. A capture whose
    reference cannot be pinned is still taken -- the manifest says `null` and
    the reader knows what they have."""
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def build_command(args):
    """The `llama-perplexity` invocation, as a list. Separate from running it
    so the tests can read the command without a model on disk."""
    return [
        args.reference_bin,
        "-m", args.model,
        "-f", args.corpus,
        "-c", str(args.n_ctx),
        "-b", str(args.batch),
        "--chunks", str(args.chunks),
        "-t", str(args.threads),
        "--kl-divergence-base", args.out,
    ]


def manifest(args, past_boundary, started, elapsed, returncode, stderr_tail):
    """Everything needed to attribute this capture later, hashes included.

    The capture file itself is hashed too. It is the artifact; a manifest that
    describes inputs and not the output cannot detect a truncated run.
    """
    return {
        "tool": "tools/kld_capture.py",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime(started)),
        "elapsed_seconds": round(elapsed, 1),
        "returncode": returncode,
        "qsa_boundary_tokens": QSA_BOUNDARY_TOKENS,
        "n_ctx": args.n_ctx,
        "chunks": args.chunks,
        "batch": args.batch,
        "threads": args.threads,
        "rows_past_boundary_per_window": past_boundary,
        "reference": {
            "binary": os.path.abspath(args.reference_bin),
            "binary_sha256": sha256_file(args.reference_bin),
            "checkout_head": git_head(args.reference_checkout)
                             if args.reference_checkout else None,
        },
        "model": {
            "path": os.path.abspath(args.model),
            "sha256": sha256_file(args.model) if args.hash_model else None,
        },
        "corpus": {
            "path": os.path.abspath(args.corpus),
            "sha256": sha256_file(args.corpus),
            "bytes": os.path.getsize(args.corpus),
        },
        "capture": {
            "path": os.path.abspath(args.out),
            "bytes": os.path.getsize(args.out) if os.path.exists(args.out) else 0,
            "sha256": sha256_file(args.out) if os.path.exists(args.out) else None,
        },
        "stderr_tail": stderr_tail,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference-bin", required=True,
                   help="upstream llama.cpp `llama-perplexity`")
    p.add_argument("--reference-checkout",
                   help="the checkout it was built from, to pin its commit")
    p.add_argument("--model", required=True, help="the qwen4exp GGUF (shard 1)")
    p.add_argument("--corpus", required=True, help="the fixed prompt set")
    p.add_argument("--out", required=True, help="capture file to write")
    p.add_argument("--n-ctx", type=int, default=4096,
                   help="evaluated window; must cross the QSA boundary")
    p.add_argument("--chunks", type=int, default=MIN_CHUNKS)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--hash-model", action="store_true",
                   help="sha256 the model too; it is tens of GiB, so off by "
                        "default and the manifest records null rather than "
                        "a figure nobody waited for")
    p.add_argument("--dry-run", action="store_true",
                   help="run every check and print the command, run nothing")
    args = p.parse_args(argv)

    past = check_context_crosses_boundary(args.n_ctx)
    if args.chunks < MIN_CHUNKS:
        raise CaptureRefusal(
            f"--chunks {args.chunks}: `llama-perplexity` evaluates at least "
            f"{MIN_CHUNKS} windows and refuses a corpus shorter than that; "
            f"asking for fewer produces no capture at all.")

    cmd = build_command(args)
    print(f"[kld-capture] window {args.n_ctx} tokens; the QSA price touches "
          f"{past} of its rows (T - {QSA_BOUNDARY_TOKENS}), and "
          f"{args.n_ctx - past} sit below the boundary")
    print("[kld-capture] " + " ".join(cmd))
    if args.dry_run:
        return 0

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-12:])
    print(tail)

    man = manifest(args, past, started, elapsed, proc.returncode, tail)
    with open(args.out + ".manifest.json", "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[kld-capture] {man['capture']['bytes']:,} B in "
          f"{man['elapsed_seconds']} s -> {args.out}")
    print(f"[kld-capture] manifest -> {args.out}.manifest.json")
    return proc.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CaptureRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
